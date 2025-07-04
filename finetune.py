import fire
import jax
import jax.numpy as jnp
import optax
import wandb
import qwix
from flax import nnx
from tqdm.auto import tqdm
from functools import partial
from optax import tree_utils as otu
from jax.sharding import NamedSharding, PartitionSpec as P
import gemma
import data


def finetune(
    model_variant = 'gemma3-1b', # ['1b', '4b', '12b', '27b']
    lora_rank = None,
    optimizer_name = 'adam', # ['adam', 'adafactor']
    peak_lr = 1e-6,
    lr_schedule = 'const',
    b2 = 0.997,
    n_epochs = 1,
    batch_size = 1,
    microbatch_size = 1,
    n_eval_samples = 128,
    eval_batch_size = 128,
    n_data_devices = 1,
    train_parallelism = 'seq', # ['seq', 'batch']
    temperature = 1,
    log_every_steps = 100,
    logging = False,
    remat = False,
    seed = 0,
    **kwargs,
):
    # check if any unrecognized arguments were passed
    if len(kwargs) > 0: raise NameError(f'Unrecognized arguments: {kwargs}')

    # get config
    train_config = locals()
    if jax.process_index() == 0:
        print(f'{train_config=}')
        if logging: wandb.init(project='picodo-finetune', config=train_config)

    # load model
    print('loading model…')
    n_tensor_devices = jax.device_count() // n_data_devices
    mesh = jax.make_mesh((n_data_devices, n_tensor_devices), ('data', 'model'))
    model, vocab = gemma.load_pretrained(model_variant, mesh)

    # use lora for all layers except normalization layers
    use_lora = lora_rank is not None
    if use_lora:
        lora_provider = qwix.lora.LoraProvider(module_path='^((?!scale).)*$', rank=lora_rank, alpha=2)
        dummy_input = jnp.ones([1, 128], dtype=jnp.int32)
        model = qwix.lora.apply_lora_to_model(model, lora_provider, dummy_input)

    # enable gradient chekpointing (currently doesn't work with Lora)
    if remat: model.layers = [jax.remat(layer) for layer in model.layers]

    # load datasets
    tokens_train, train_loss_mask, tokens_eval, problems_eval, solutions_eval = data.load_datasets(vocab)
    
    # optimizer
    warmup_frac = 0.05
    n_train_samples = len(tokens_train)
    n_batches = n_train_samples // batch_size
    grad_acc_steps = batch_size // microbatch_size
    n_optimizer_steps = n_epochs * n_batches
    warmup_steps = int(warmup_frac * n_optimizer_steps)
    if lr_schedule == 'const': lr = peak_lr
    if lr_schedule == 'cosine': lr = optax.schedules.warmup_cosine_decay_schedule(0, peak_lr, warmup_steps, max(1, n_optimizer_steps))
    if optimizer_name == 'adam': tx = optax.adam(lr, 0.9, b2)
    if optimizer_name == 'adafactor': tx = optax.adafactor(lr, decay_rate=b2)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.LoRAParam) if use_lora else nnx.Optimizer(model, tx)
    
    # pring number of parameters
    n_model_params = jax.tree.reduce(lambda x, y: x + jnp.size(y), nnx.state(model), 0)
    n_opt_params = jax.tree.reduce(lambda x, y: x + jnp.size(y), nnx.state(optimizer.opt_state), 0)
    print(f'{n_model_params=:_}')
    print(f'{n_opt_params=:_}')

    # training loop
    step = 0
    train_loss = 0
    key = jax.random.PRNGKey(seed)
    key, key_eval = jax.random.split(key)
    del model
    with mesh:
        # iterate over epochs
        if ((n_epochs > 0) and (jax.process_index() == 0)): pbar = tqdm(total=n_optimizer_steps, desc='Training')
        for epoch in range(n_epochs):

            # train for 1 epoch
            key, key_train = jax.random.split(key)
            idxs = jax.random.choice(key_train, n_train_samples, shape=[n_batches, grad_acc_steps, microbatch_size], replace=False)
            for idx in idxs:

                # load batch
                if train_parallelism == 'seq': train_data_pspec = P(None, None, 'data') # [grad_acc_steps, microbatch_size, seq_len]
                if train_parallelism == 'batch': train_data_pspec = P(None, 'data', None) # [grad_acc_steps, microbatch_size, seq_len]
                tokens_batch = jax.device_put(tokens_train[idx], NamedSharding(mesh, train_data_pspec)) # [grad_acc_steps, microbatch_size, seq_len]
                loss_mask_batch = jax.device_put(train_loss_mask[idx], NamedSharding(mesh, train_data_pspec)) # [grad_acc_steps, microbatch_size, seq_len]
                
                # training step
                if grad_acc_steps == 1:
                    optimizer, batch_loss = train_step(optimizer, tokens_batch[0], loss_mask_batch[0], use_lora)
                else:
                    optimizer, batch_loss = train_step_grad_acc(optimizer, tokens_batch, loss_mask_batch, use_lora)
                    
                # logging
                train_loss += batch_loss
                if (step+1) % log_every_steps == 0:
                    avg_loss = train_loss / log_every_steps
                    if jax.process_index() == 0:
                        if logging: wandb.log({'train_loss': float(avg_loss)}, step)
                        pbar.set_postfix_str(f'loss={float(avg_loss):.2f}')
                    train_loss = 0
                step += 1
                if jax.process_index() == 0: pbar.update(1)
        if (n_epochs > 0) and (jax.process_index() == 0): pbar.close()
        
        # eval
        eval_metrics = data.benchmark_model(key_eval, optimizer.model, tokens_eval, problems_eval, solutions_eval, vocab, eval_batch_size, n_eval_samples, temperature)
        if logging and (jax.process_index() == 0):
            wandb.log(eval_metrics, step)
            wandb.finish()


@partial(nnx.jit)
def loss_fn(model, x, loss_mask): # [B, T]
    B, T = x.shape
    y = jnp.roll(x, -1, axis=1)
    logits, _ = model(x) # [B, T, V]
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y) # [B, T]
    return (losses * loss_mask).sum() / loss_mask.sum()


@partial(nnx.jit, static_argnames=('lora'))
def train_step(optimizer, tokens, loss_mask, lora=False):
    argnums = nnx.DiffState(0, nnx.LoRAParam) if lora else 0
    loss, grads = nnx.value_and_grad(loss_fn, argnums=argnums)(optimizer.model, tokens, loss_mask)
    optimizer.update(grads)
    return optimizer, loss


@partial(nnx.jit, static_argnames=('lora'))
def train_step_grad_acc(optimizer, tokens, loss_mask, lora=False):
    argnums = nnx.DiffState(0, nnx.LoRAParam) if lora else 0
    loss_mean = 0
    grad_mean = otu.tree_zeros_like(nnx.state(optimizer.model))
    def step_fn(i, args):
        grad_mean, loss_mean = args
        batch_loss, batch_grads = nnx.value_and_grad(loss_fn, argnums=argnums)(optimizer.model, tokens[i], loss_mask[i])
        grad_mean = jax.tree.map(lambda m, g: (i*m + g) / (i+1), grad_mean, batch_grads)
        loss_mean = (i*loss_mean + batch_loss) / (i+1)
        return grad_mean, loss_mean
    grad_mean, loss_mean = jax.lax.fori_loop(0, len(tokens), step_fn, (grad_mean, loss_mean))
    optimizer.update(grad_mean)
    return optimizer, loss_mean


if __name__ == '__main__':
    fire.Fire(finetune)
