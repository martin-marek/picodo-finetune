import math
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from math_verify import parse, verify
from sampler import sample


def tokenize(sequences, vocab, seq_len, add_eos=False):
    sequences_tokenized = vocab.EncodeAsIds(sequences)
    eos_id = vocab.eos_id()
    tokens = np.full([len(sequences), seq_len], vocab.pad_id(), dtype=jnp.int32)
    tokens[:, 0] = vocab.bos_id()
    for i, seq_tok in enumerate(sequences_tokenized):
        if add_eos: seq_tok += [eos_id]
        tokens[i, 1:1+len(seq_tok)] = seq_tok[:seq_len-1]
    return jnp.array(tokens, dtype=jnp.int32)


def load_datasets(vocab, seq_len=1024):
    pad_id = vocab.pad_id()
    eos_id = vocab.eos_id()

    # load MATH dataset
    print('loading datasets…')
    ds_name = 'EleutherAI/hendrycks_math'
    configs = get_dataset_config_names(ds_name) # ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    ds_train = concatenate_datasets([load_dataset(ds_name, config, split='train') for config in configs]) # ['problem', 'solution']
    ds_valid = concatenate_datasets([load_dataset(ds_name, config, split='test') for config in configs]) # ['problem', 'solution']

    # tokenize training dataset
    print('tokenizing training dataset…')
    prompts_train = []
    examples_train = []
    for example in ds_train:
        prompt = f'Problem: {example["problem"]}\nSolution: '
        solution = f'{example["solution"]}'
        prompts_train += [prompt]
        examples_train += [prompt+solution]
    prompts_tokenized_train = tokenize(prompts_train, vocab, seq_len)
    examples_tokenized_train = tokenize(examples_train, vocab, seq_len, add_eos=True)
    within_seq_length = examples_tokenized_train[:, -1] == pad_id
    prompts_tokenized_train, examples_tokenized_train = prompts_tokenized_train[within_seq_length], examples_tokenized_train[within_seq_length]
    print(f'fraction within seq. length: {within_seq_length.mean():.1%}')
    first_solution_token = jnp.argmax(prompts_tokenized_train == pad_id, axis=1) - 1 # [B]
    last_solution_token = jnp.argmax(examples_tokenized_train == eos_id, axis=1) - 1 # [B]
    train_loss_mask = (first_solution_token[:, None] <= jnp.arange(seq_len)[None, :]) & (jnp.arange(seq_len)[None, :] <= last_solution_token[:, None])  # [B, T]
    print(f'{float(train_loss_mask.mean())=:.1%}')

    # tokenize eval dataset
    print('tokenizing eval dataset…')
    prompts_eval = []
    problems_eval = []
    solutions_eval = []
    for example in ds_valid:
        prompts_eval += [f'Problem: {example["problem"]}\nSolution: ']
        problems_eval += [example['problem']]
        solutions_eval += [example['solution']]
    problems_tokenized_eval = tokenize(prompts_eval, vocab, seq_len)
    
    return examples_tokenized_train, train_loss_mask, problems_tokenized_eval, np.array(problems_eval), np.array(solutions_eval)


def benchmark_model(key, model, tokens, problems_eval, solutions_eval, vocab, batch_size, n_eval_samples=None, temperature=1):
    pad_id = vocab.pad_id()
    eos_id = vocab.eos_id()
    key_decoding, key_questions = jax.random.split(key)
    mesh = model.in_embed.embedding.value.sharding.mesh
    if n_eval_samples is None: n_eval_samples = len(tokens)
    n_batches = int(math.ceil(n_eval_samples / batch_size))
    n_eval_samples = n_batches * batch_size
    sample_idxs = jax.random.choice(key_questions, max(n_eval_samples, len(tokens)), shape=[n_batches, batch_size], replace=False)
    lengths_list = []
    correct_list = []
    finished_list = []
    for batch_idx in sample_idxs:
        tokens_batch = jax.device_put(tokens[batch_idx], NamedSharding(mesh, P('data', None)))
        completions_tokens = sample(key_decoding, model, tokens_batch, temperature)
        completions_text = vocab.DecodeIds(completions_tokens)
        for sample_idx, completion_tokens, completion_text in zip(batch_idx, completions_tokens, completions_text):
            if sample_idx < len(problems_eval):
                problem = problems_eval[sample_idx]
                gold = solutions_eval[sample_idx]
                parsed = parse(completion_text)
                finished = eos_id in completion_tokens
                correct = verify(parse(gold), parsed)
                lengths_list += [len(completion_tokens)]
                finished_list += [finished]
                correct_list += [correct]
                print('------------')
                print(f'PROMPT:\n{problem}\nCOMPLETION:\n{completion_text}\nPARSED: {parsed}\nGOLD: {gold}\nCORRECT: {correct}')

    return dict(length=np.mean(lengths_list), finished=np.mean(finished_list), accuracy=np.mean(correct_list))
