import math
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from math_verify import parse, verify
from sampler import sample


def tokenize(sequences, vocab, seq_len, pad_id=0):
    sequences_tokenized = vocab.EncodeAsIds(sequences)
    within_seq_length = [len(seq) <= seq_len for seq in sequences_tokenized]
    sequences_tokenized = [seq for seq, keep in zip(sequences_tokenized, within_seq_length) if keep]
    print(f'fraction within seq. length: {np.mean(within_seq_length):.1%}')
    B, T = len(sequences_tokenized), seq_len
    print(f'{B=}, {T=}')
    tokens = np.full([B, T], pad_id, dtype=jnp.int32)
    tokens[:, 0] = vocab.bos_id()
    for i, seq_tok in enumerate(sequences_tokenized):
        tokens[i, 1:1+len(seq_tok)] = seq_tok
    return jnp.array(tokens, dtype=jnp.int32)


def load_datasets(vocab, seq_len=1024):
    sot_token = vocab.EncodeAsIds('<start_of_turn>')[0]
    eot_token = vocab.EncodeAsIds('<end_of_turn>')[0]

    # load MATH dataset
    print('loading datasets…')
    ds_name = 'EleutherAI/hendrycks_math'
    configs = get_dataset_config_names(ds_name) # ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    ds_train = concatenate_datasets([load_dataset(ds_name, config, split='train') for config in configs]) # ['problem', 'solution']
    ds_valid = concatenate_datasets([load_dataset(ds_name, config, split='test') for config in configs]) # ['problem', 'solution']

    # tokenize training dataset
    print('tokenizing training dataset…')
    examples_train = []
    for example in ds_train:
        text = (f'<start_of_turn>user\n'
                f'{example["problem"]}<end_of_turn>\n'
                f'<start_of_turn>model\n'
                f'{example["solution"]}<end_of_turn>')
        examples_train += [text]
    tokens_train = jnp.array(tokenize(examples_train, vocab, seq_len))
    first_model_token = jnp.array([np.where(x==sot_token)[0][1] for x in tokens_train]) + 1
    last_model_token = jnp.array([np.where(x==eot_token)[0][1] for x in tokens_train])
    train_loss_mask = (first_model_token[:, None] <= jnp.arange(seq_len)[None, :]) & (jnp.arange(seq_len)[None, :] <= last_model_token[:, None])  # [B, T]
    print(f'{float(train_loss_mask.mean())=:.1%}')

    # tokenize eval dataset
    print('tokenizing eval dataset…')
    prompts_eval = []
    problems_eval = []
    solutions_eval = []
    for example in ds_valid:
        prompt = (f'<start_of_turn>user\n'
                  f'{example["problem"]}<end_of_turn>\n'
                  f'<start_of_turn>model\n')
        prompts_eval += [prompt]
        problems_eval += [example['problem']]
        solutions_eval += [example['solution']]
    tokens_eval = tokenize(prompts_eval, vocab, seq_len)
    
    return tokens_train, train_loss_mask, tokens_eval, np.array(problems_eval), np.array(solutions_eval)


def benchmark_model(key, model, tokens, problems_eval, solutions_eval, vocab, batch_size, n_eval_samples=None, temperature=1, pad_id=0, eot_id=106):
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
                finished = eot_id in completion_tokens
                correct = verify(parse(gold), parsed)
                lengths_list += [len(completion_tokens)]
                finished_list += [finished]
                correct_list += [correct]
                print('------------')
                print(f'PROMPT:\n{problem}\nCOMPLETION:\n{completion_text}\nPARSED: {parsed}\nGOLD: {gold}\nCORRECT: {correct}')

    return dict(length=np.mean(lengths_list), finished=np.mean(finished_list), accuracy=np.mean(correct_list))
