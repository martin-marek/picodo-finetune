import math
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from itertools import chain
from datasets import load_dataset
from math_verify import parse, verify
from sampler import sample


def tokenize(sequences, vocab, seq_len, pad_id=0):
    sequences_tokenized = vocab.EncodeAsIds(sequences)
    assert max(map(len, sequences_tokenized)) <= seq_len
    B, T = len(sequences), seq_len
    tokens = np.full([B, T], pad_id, dtype=jnp.int32)
    tokens[:, 0] = vocab.bos_id()

    for i, seq_tok in enumerate(sequences_tokenized):
        tokens[i, 1:1+len(seq_tok)] = seq_tok

    return jnp.array(tokens, dtype=jnp.int32)


def load_datasets(vocab, train_seq_len, eval_seq_len, pad_id=0):
    # load datasets
    ds_train = load_dataset('simplescaling/s1K-1.1', split='all') # ['question', 'solution', 'gemini_thinking_trajectory', 'gemini_attempt']
    aime24 = load_dataset(f'HuggingFaceH4/aime_2024', split='all') # ['problem', 'answer']
    aime25 = load_dataset(f'MathArena/aime_2025', split='all') # ['problem', 'answer']

    # tokenize training dataset
    examples_train = []
    for i, example in enumerate(ds_train):
        text = (f'<start_of_turn>user\n'
                f'{example["question"]}<end_of_turn>\n'
                f'<start_of_turn>model\n'
                f'<think>\n'
                f'{example["gemini_thinking_trajectory"]}\n'
                f'<\\think>\n'
                f'{example["gemini_attempt"]}<end_of_turn>')
        examples_train += [text]
    tokens_train = jnp.array(tokenize(examples_train, vocab, train_seq_len))
    sot_token = vocab.EncodeAsIds('<start_of_turn>')[0]
    first_model_token = jnp.array([np.where(x==sot_token)[0][1] for x in tokens_train]) + 1
    last_model_token = jnp.array([np.where(x==pad_id)[0][0] for x in tokens_train])
    train_loss_mask = (first_model_token[:, None] <= jnp.arange(train_seq_len)[None, :]) & (jnp.arange(train_seq_len)[None, :] <= last_model_token[:, None])  # [B, T]
    print(f'train dataset max. length: {jnp.argmax(tokens_train==0, axis=1).max()}')

    # tokenize eval dataset
    prompts_eval = []
    problems_eval = []
    answers_eval = []
    for i, example in enumerate(chain(aime24, aime25)):
        prompt = (f'<start_of_turn>user\n'
                  f'{example["problem"]} Hint: the answer is an integer between $0$ and $999$, inclusive.<end_of_turn>\n'
                  f'<start_of_turn>model\n')
        prompts_eval += [prompt]
        problems_eval += [example["problem"]]
        answers_eval += [example['answer']]
    tokens_eval = tokenize(prompts_eval, vocab, eval_seq_len)
    print(f'{tokens_eval.shape=}')
    
    return tokens_train, train_loss_mask, tokens_eval, np.array(problems_eval), np.array(answers_eval)


def benchmark_model(key, model, tokens, problems_eval, answers_eval, vocab, batch_size, n_eval_samples=None, temperature=1, pad_id=0, eot_id=106):
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
                gold = answers_eval[sample_idx]
                parsed = parse(completion_text)
                finished = eot_id in completion_tokens
                correct = verify(parse(gold), parsed)
                lengths_list += [len(completion_tokens)]
                finished_list += [finished]
                correct_list += [correct]
                print('------------')
                print(f'PROMPT:\n{problem}\nCOMPLETION:\n{completion_text}\nPARSED: {parsed}\nGOLD: {gold}\nCORRECT: {correct}')

    return dict(length=np.mean(lengths_list), finished=np.mean(finished_list), accuracy=np.mean(correct_list))
