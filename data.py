import math
import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from math_verify import parse, verify
from sampler import sample


def tokenize(sequences, vocab, seq_len, batch_divisor=1, pad_id=0):
    sequences_tokenized = vocab.EncodeAsIds(sequences)
    assert max(map(len, sequences_tokenized)) <= seq_len
    B, T = len(sequences), seq_len
    B = int(batch_divisor * math.ceil(B / batch_divisor)) # round T up to be divisible by `batch_divisor`
    print(f'{B=}')
    tokens = np.full([B, T], pad_id, dtype=jnp.int32)
    tokens[:, 0] = vocab.bos_id()

    for i, seq_tok in enumerate(sequences_tokenized):
        tokens[i, 1:1+len(seq_tok)] = seq_tok

    return jnp.array(tokens, dtype=jnp.int32)


def load_datasets(eval_ds_name, vocab, train_seq_len, eval_seq_len, batch_divisor=1):
    # load datasets
    ds_train = load_dataset('simplescaling/s1K-1.1', split='all') # ['question', 'solution', 'gemini_thinking_trajectory', 'gemini_attempt']
    ds_eval = load_dataset(f'HuggingFaceH4/{eval_ds_name}', split='all') # ['problem', 'answer']
    
    # tokenize training dataset
    examples_train = []
    for i, example in enumerate(ds_train):
        text = (f'<start_of_turn>user\n'
                f'{example["question"]}<end_of_turn>\n'
                f'<start_of_turn>model\n'
                f'{example["gemini_thinking_trajectory"]}\n'
                f'**Final Answer**\n'
                f'{example["gemini_attempt"]}<end_of_turn>\n')
        examples_train += [text]
    tokens_train = jnp.array(tokenize(examples_train, vocab, train_seq_len))
    sot_token = vocab.EncodeAsIds('<start_of_turn>')[0]
    eot_token = vocab.EncodeAsIds('<end_of_turn>')[0]
    first_model_token = jnp.array([jnp.where(x==sot_token)[0][-1] for x in tokens_train]) + 1
    last_model_token = jnp.array([jnp.where(x==eot_token)[0][-1] for x in tokens_train]) - 1
    train_loss_mask = (first_model_token[:, None] <= jnp.arange(train_seq_len)[None, :]) & (jnp.arange(train_seq_len)[None, :] <= last_model_token[:, None])  # [B, T]
    print(f'train dataset max. length: {jnp.argmax(tokens_train==0, axis=1).max()}')

    # tokenize eval dataset
    problems_eval = []
    answers_eval = []
    for i, example in enumerate(ds_eval):
        text = (f'<start_of_turn>user\n'
                f'{example["problem"]}\n'
                f'<end_of_turn>\n'
                f'<start_of_turn>model\n')
        problems_eval += [text]
        answers_eval += [example['answer']]
    answers_eval = np.array(answers_eval)
    tokens_eval = tokenize(problems_eval, vocab, eval_seq_len, batch_divisor)
    print(f'{tokens_eval.shape=}')
    
    return tokens_train, train_loss_mask, tokens_eval, answers_eval


def benchmark_model(key, model, tokens_eval, answers_eval, vocab, eval_batch_size, n_eval_samples):
    key_decoding, key_questions = jax.random.split(key)
    n_eval_examples_total = len(answers_eval)
    eot_token = vocab.EncodeAsIds('<end_of_turn>')[0]
    sample_idxs = jax.random.choice(key_questions, n_eval_examples_total, shape=[n_eval_samples//eval_batch_size, eval_batch_size])
    lengths = []
    correct = []
    finished = []
    for idx in sample_idxs:
        completions_tokens = sample(key_decoding, model, tokens_eval[idx])
        completions_text = vocab.DecodeIds(completions_tokens)
        lengths += [len(seq) for seq in completions_tokens]
        finished += [eot_token in seq for seq in completions_tokens]
        correct += [verify(gold, parse(completion)) for gold, completion in zip(answers_eval[idx], completions_text)]
        for gold, completion, corr in zip(answers_eval[idx], completions_text, correct):
            print('------------')
            print(f'COMPLETION:\n{completion}\nPARSED: {parse(completion)}\nGOLD: {gold}\nCORRECT:{corr}')

    mean = lambda x: sum(x) / len(x)
    return dict(length=mean(lengths), finished=mean(finished), accuracy=mean(correct))
