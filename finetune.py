import fire
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import nnx
from tqdm.auto import tqdm
from functools import partial
from contextlib import nullcontext 
from datasets import load_dataset
from math_verify import parse, verify
import gemma
import data
from sampler import sample


@partial(jax.jit, static_argnames=('model_graphdef'))
def loss_fn(model_state, model_graphdef, x, pad_id=0): # [B, T]
    B, T = x.shape
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    seq_lengths = (x == pad_id).argmax(1) # [B]
    loss_mask = jnp.arange(T)[None, :] < seq_lengths[:, None] # [B, T]
    loss_mask = loss_mask.at[:, -1].set(False)
    logits, _ = model(x) # [B, T, V]
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y) # [B, T]
    return (losses * loss_mask).sum() / loss_mask.sum()


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef'))
def train_step(opt_state, opt_graphdef, model_graphdef, batch):
    loss, grads = jax.value_and_grad(loss_fn)(opt_state.model, model_graphdef, batch)
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    return opt_state, loss


def eval_step(key, model_state, model_graphdef, tokens_eval, answers_eval, vocab, eval_batch_size=5, n_eval_samples=20):
    model = nnx.merge(model_graphdef, model_state)
    n_eval_examples_total = len(tokens_eval)
    sample_idxs = jax.random.choice(key, n_eval_examples_total, shape=[n_eval_samples//eval_batch_size, eval_batch_size])
    correct = []
    for idx in sample_idxs:
        predictions_tokens = sample(model, tokens_eval[idx])
        predictions_text = vocab.DecodeIds(predictions_tokens)
        batch_correct = [verify(gold, parse(answer)) for gold, answer in zip(answers_eval[idx], predictions_text)]
        correct += batch_correct

    accuracy = sum(correct) / len(correct)
    return accuracy


def finetune(
    model_variant = 'gemma3-4b-it',
    use_lora = False,
    n_epochs = 10,
    learning_rate = 1e-7,
    n_eval_samples = 30,
    eval_batch_size = 10,
    log_every_steps = 10,
    max_seq_len = 8600,
    eval_dataset = 'MATH-500', # 'aime_2024', 'MATH-500'
    seed = 0,
):
    train_config = locals()
    print(f'{train_config=}')
    
    # load datasets
    ds_train = load_dataset('simplescaling/s1K-1.1', split='train') # ['question', 'solution', 'gemini_thinking_trajectory', 'gemini_attempt']
    ds_eval = load_dataset('HuggingFaceH4/MATH-500', split='test') # ['problem', 'answer']

    # load model
    mesh = jax.make_mesh((1, jax.device_count()), ('data', 'model'))
    model, vocab = gemma.load_pretrained(model_variant, mesh)
    model_graphdef, model_state = nnx.split(model)

    # tokenize trainig dataset
    examples_train = []
    for i, example in enumerate(ds_train):
        question, thinking_trajectory, answer = example['question'], example['gemini_thinking_trajectory'], example['gemini_attempt']
        text = f'<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{thinking_trajectory}\n**Final Answer**\n{answer}<end_of_turn>'
        examples_train += [text]
    tokens_train = jnp.array(data.tokenize(examples_train, vocab, max_seq_len))
    print(f'train dataset max. length: {jnp.argmax(tokens_train==0, axis=1).max()}')

    # tokenize eval dataset
    problems_eval = []
    answers_eval = []
    for i, example in enumerate(ds_eval):
        text = f'<start_of_turn>user\n{example["problem"]}<end_of_turn>\n<start_of_turn>model\n'
        problems_eval += [text]
        answers_eval += [example['answer']]
    answers_eval = np.array(answers_eval)
    tokens_eval = data.tokenize(problems_eval, vocab, max_seq_len)

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project='picodo-finetune', config=train_config)
    
    # training loop
    key = jax.random.PRNGKey(seed)
    tx = optax.adamw(learning_rate, 0.9, 0.999)
    optimizer = nnx.Optimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)

    with mesh:
        step = 0
        train_loss = 0
        n_train_steps = n_epochs * len(tokens_train)
        pbar = tqdm(total=n_train_steps) if jax.process_index() == 0 else nullcontext 
        with pbar:

            # iter over epochs
            for epoch in range(n_epochs):
                key, key_train, key_eval = jax.random.split(key, 3)
                
                # train for 1 epoch
                idxs = jax.random.permutation(key_train, len(tokens_train))
                for idx in idxs:

                    # train on a single example
                    batch = tokens_train[idx, None] # [1, T]
                    opt_state, batch_loss = train_step(opt_state, opt_graphdef, model_graphdef, batch)
                    train_loss += batch_loss
                    
                    # logging
                    if step % log_every_steps == 0:
                        avg_loss = train_loss / log_every_steps
                        if jax.process_index() == 0:
                            wandb.log({'train_loss': float(avg_loss)}, step)
                            pbar.set_postfix_str(f'loss={float(avg_loss):.2f}')
                        train_loss = 0
                    step += 1
                    if jax.process_index() == 0: pbar.update(1)
        
                # eval
                accuracy = eval_step(key_eval, opt_state.model, model_graphdef, tokens_eval, answers_eval, vocab, eval_batch_size, n_eval_samples)
                if jax.process_index() == 0:
                    wandb.log({'eval_acc': float(accuracy)}, step)


if __name__ == '__main__':
    fire.Fire(finetune)
