import fire
import jax
import jax.numpy as jnp
import optax
import wandb
from flax import nnx
from tqdm.auto import tqdm
from functools import partial
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


def finetune(
    model_variant = 'gemma3-1b-it', # '1b', '4b', '12b', '27b'
    eval_dataset = 'MATH-500', # 'MATH-500', 'aime_2024'
    use_lora = False,
    n_epochs = 5,
    learning_rate = 1e-6,
    n_eval_samples = 30,
    eval_batch_size = 10,
    log_every_steps = 20,
    train_seq_len = 9216,
    eval_seq_len = 16384,
    sequence_devices = 1,
    seed = 0,
):
    train_config = locals()
    print(f'{train_config=}')

    # load model
    tensor_devices = jax.device_count() // sequence_devices
    mesh = jax.make_mesh((sequence_devices, tensor_devices), ('data', 'model'))
    model, vocab = gemma.load_pretrained(model_variant, mesh)
    model_graphdef = nnx.graphdef(model)

    # load datasets
    tokens_train, train_loss_mask, tokens_eval, answers_eval = data.load_datasets(eval_dataset, vocab, train_seq_len, eval_seq_len)
    data_sharding = NamedSharding(mesh, P(None, 'data')) # [B, T]
    tokens_train = jax.device_put(tokens_train, data_sharding)
    train_loss_mask = jax.device_put(train_loss_mask, data_sharding)
    tokens_eval = jax.device_put(tokens_eval, data_sharding)

    # optimizer
    tx = optax.adafactor(learning_rate, decay_rate=0.997)
    optimizer = nnx.Optimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project='picodo-finetune', config=train_config)

    # training loop
    step = 0
    epoch = 0
    train_loss = 0
    key = jax.random.PRNGKey(seed)
    n_train_steps = n_epochs * len(tokens_train)
    pbar = tqdm(total=n_train_steps) if jax.process_index() == 0 else None
    with mesh:
        # iterate over epochs
        while True:
            key, key_train, key_eval = jax.random.split(key, 3)
            
            # eval
            model = nnx.merge(model_graphdef, opt_state.model)
            eval_metrics = data.benchmark_model(key_eval, model, tokens_eval, answers_eval, vocab, eval_batch_size, n_eval_samples)
            if jax.process_index() == 0:
                wandb.log(eval_metrics, step)
            if epoch == n_epochs: break

            # train for 1 epoch
            idxs = jax.random.permutation(key_train, len(tokens_train))
            for idx in idxs:

                # train on a single example
                opt_state, batch_loss = train_step(opt_state, opt_graphdef, model_graphdef, tokens_train[idx, None], train_loss_mask[idx, None])
                train_loss += batch_loss
                
                # logging
                if (step+1) % log_every_steps == 0:
                    avg_loss = train_loss / log_every_steps
                    if jax.process_index() == 0:
                        wandb.log({'train_loss': float(avg_loss)}, step)
                        pbar.set_postfix_str(f'loss={float(avg_loss):.2f}')
                    train_loss = 0
                step += 1
                if jax.process_index() == 0: pbar.update(1)
            epoch += 1
    pbar.close()


if __name__ == '__main__':
    fire.Fire(finetune)
