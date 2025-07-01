import fire
import jax
import jax.numpy as jnp
import optax
import wandb
from flax import nnx
from tqdm.auto import tqdm
from functools import partial
from optax import tree_utils as otu
from jax.sharding import NamedSharding, PartitionSpec as P
import gemma
import data


@partial(jax.jit, static_argnames=('model_graphdef'))
def loss_fn(model_state, model_graphdef, x, loss_mask): # [B, T]
    B, T = x.shape
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits, _ = model(x) # [B, T, V]
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y) # [B, T]
    return (losses * loss_mask).sum() / loss_mask.sum()


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef'))
def train_step(opt_state, opt_graphdef, model_graphdef, tokens, loss_mask):
    loss, grads = jax.value_and_grad(loss_fn)(opt_state.model, model_graphdef, tokens, loss_mask)
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    return opt_state, loss


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef'))
def train_step_grad_acc(opt_state, opt_graphdef, model_graphdef, tokens, loss_mask):
    loss_mean = 0
    grad_mean = otu.tree_zeros_like(opt_state.model)
    def step_fn(i, args):
        grad_mean, loss_mean = args
        batch_loss, batch_grads = jax.value_and_grad(loss_fn)(opt_state.model, model_graphdef, tokens[i], loss_mask[i])
        grad_mean = jax.tree.map(lambda m, g: (i*m + g) / (i+1), grad_mean, batch_grads)
        loss_mean = (i*loss_mean + batch_loss) / (i+1)
        return grad_mean, loss_mean
    grad_mean, loss_mean = jax.lax.fori_loop(0, len(tokens), step_fn, (grad_mean, loss_mean))
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grad_mean)
    opt_state = nnx.state(optimizer)
    return opt_state, loss_mean


def finetune(
    model_variant = 'gemma3-4b-it', # ['1b', '4b', '12b', '27b']
    eval_dataset = 'aime_2024', # ['aime_2024', 'MATH-500']
    use_lora = False,
    optimizer_name = 'adafactor', # ['adam', 'adafactor']
    peak_lr = 1e-6,
    b2 = 0.997,
    n_epochs = 1,
    batch_size = 1,
    microbatch_size = 1,
    n_eval_samples = 30,
    eval_batch_size = 16,
    log_every_steps = 1,
    train_seq_len = 9216,
    eval_seq_len = 32768,
    n_data_devices = 1,
    train_parallelism = 'seq', # ['seq', 'batch']
    logging = False,
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
    n_tensor_devices = jax.device_count() // n_data_devices
    mesh = jax.make_mesh((n_data_devices, n_tensor_devices), ('data', 'model'))
    model, vocab = gemma.load_pretrained(model_variant, mesh)
    model_graphdef = nnx.graphdef(model)

    # load datasets
    tokens_train, train_loss_mask, tokens_eval, problems_eval, answers_eval = data.load_datasets(eval_dataset, vocab, train_seq_len, eval_seq_len, eval_batch_size)
    
    # optimizer
    warmup_frac = 0.05
    n_train_samples = len(tokens_train)
    n_batches = n_train_samples // batch_size
    grad_acc_steps = batch_size // microbatch_size
    n_optimizer_steps = n_epochs * n_batches
    warmup_steps = int(warmup_frac * n_optimizer_steps)
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, peak_lr, warmup_steps, max(1, n_optimizer_steps))
    if optimizer_name == 'adam': tx = optax.adam(lr_schedule, 0.9, b2)
    if optimizer_name == 'adafactor': tx = optax.adafactor(lr_schedule, decay_rate=b2)
    optimizer = nnx.Optimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)
    del model, optimizer

    # training loop
    step = 0
    train_loss = 0
    key = jax.random.PRNGKey(seed)
    key, key_eval = jax.random.split(key)
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
                if grad_acc_steps == 1: opt_state, batch_loss = train_step(opt_state, opt_graphdef, model_graphdef, tokens_batch[0], loss_mask_batch[0])
                else: opt_state, batch_loss = train_step_grad_acc(opt_state, opt_graphdef, model_graphdef, tokens_batch, loss_mask_batch)
                    
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
        model = nnx.merge(model_graphdef, opt_state.model)
        del opt_state
        eval_metrics = data.benchmark_model(key_eval, model, tokens_eval, problems_eval, answers_eval, vocab, eval_batch_size, n_eval_samples)
        if logging and (jax.process_index() == 0):
            wandb.log(eval_metrics, step)
            wandb.finish()


if __name__ == '__main__':
    fire.Fire(finetune)
