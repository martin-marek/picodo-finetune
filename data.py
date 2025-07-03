import math
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from datasets import load_dataset
from math_verify import parse, verify
from sampler import sample


def tokenize(sequences, vocab, seq_len, pad_id=0):
    sequences_tokenized = vocab.EncodeAsIds(sequences)
    assert (max_length := max(map(len, sequences_tokenized))) <= seq_len, f'{max_length=}'
    B, T = len(sequences), seq_len
    tokens = np.full([B, T], pad_id, dtype=jnp.int32)
    tokens[:, 0] = vocab.bos_id()

    for i, seq_tok in enumerate(sequences_tokenized):
        tokens[i, 1:1+len(seq_tok)] = seq_tok

    return jnp.array(tokens, dtype=jnp.int32)


def load_datasets(teacher, eval_ds_name, vocab, eval_seq_len, force_think, pad_id=0):
    # load datasets
    ds_train = load_dataset('simplescaling/s1K-1.1', split='all') # ['question', 'solution', 'gemini_thinking_trajectory', 'gemini_attempt']
    if eval_ds_name == 'aime':
        eval_datasets = [
            load_dataset(f'HuggingFaceH4/aime_2024', split='all'), # ['problem', 'answer']
            load_dataset(f'MathArena/aime_2025', split='all'), # ['problem', 'answer']
        ]
    if eval_ds_name == 'math500':
        eval_datasets = [load_dataset(f'HuggingFaceH4/MATH-500', split='all')]  # ['problem', 'answer']

    # tokenize training dataset
    if teacher == 'gemini': train_seq_len = 9216
    if teacher == 'deepseek': train_seq_len = 28672
    examples_train = []
    for example in ds_train:
        question = example['question']
        thinking_trajectory = example[f'{teacher}_thinking_trajectory']
        attempt = example[f'{teacher}_attempt']
        text = (f'<start_of_turn>user\n'
                f'{question}<end_of_turn>\n'
                f'<start_of_turn>model\n'
                f'<think>\n'
                f'{thinking_trajectory}\n'
                f'<\\think>\n'
                f'{attempt}<end_of_turn>')
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
    for eval_dataset in eval_datasets:
        for example in eval_dataset:
            problem = example["problem"]
            if eval_ds_name == 'aime': problem += ' Hint: the answer is an integer between $0$ and $999$, inclusive.'
            prompt = (f'<start_of_turn>user\n'
                      f'{problem}<end_of_turn>\n'
                      f'<start_of_turn>model\n')
            if force_think: prompt += f'<think>\n'
            prompts_eval += [prompt]
            problems_eval += [problem]
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
