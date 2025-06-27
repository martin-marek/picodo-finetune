import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from math_verify import parse, verify
from sampler import sample


def load_datasets(eval_ds_name):
    ds_train = load_dataset('simplescaling/s1K-1.1', split='all') # ['question', 'solution', 'gemini_thinking_trajectory', 'gemini_attempt']
    ds_eval = load_dataset(f'HuggingFaceH4/{eval_ds_name}', split='all') # ['problem', 'answer']
    return ds_train, ds_eval


def tokenize(sequences, vocab, seq_len, pad_id=0):
    sequences_tokenized = vocab.EncodeAsIds(sequences)
    assert max(map(len, sequences_tokenized)) <= seq_len
    B, T = len(sequences), seq_len
    tokens = np.full([B, T], pad_id, dtype=jnp.int32)
    tokens[:, 0] = vocab.bos_id()

    for i, seq_tok in enumerate(sequences_tokenized):
        tokens[i, 1:1+len(seq_tok)] = seq_tok

    return jnp.array(tokens, dtype=jnp.int32)


def tokenize_training_dataset(dataset, vocab, max_seq_len):
    examples_train = []
    for i, example in enumerate(dataset):
        text = (f'<start_of_turn>user\n'
                f'{example["question"]}<end_of_turn>\n'
                f'<start_of_turn>model\n'
                f'{example["gemini_thinking_trajectory"]}\n'
                f'**Final Answer**\n'
                f'{example["gemini_attempt"]}<end_of_turn>\n')
        examples_train += [text]
    tokens_train = jnp.array(tokenize(examples_train, vocab, max_seq_len))
    sot_token = vocab.EncodeAsIds('<start_of_turn>')[0]
    eot_token = vocab.EncodeAsIds('<end_of_turn>')[0]
    first_model_token = jnp.array([jnp.where(x==sot_token)[0][-1] for x in tokens_train]) + 1
    last_model_token = jnp.array([jnp.where(x==eot_token)[0][-1] for x in tokens_train]) - 1
    train_loss_mask = (first_model_token[:, None] <= jnp.arange(max_seq_len)[None, :]) & (jnp.arange(max_seq_len)[None, :] <= last_model_token[:, None])  # [B, T]
    print(f'train dataset max. length: {jnp.argmax(tokens_train==0, axis=1).max()}')
    return tokens_train, train_loss_mask


def tokenize_eval_dataset(dataset, vocab, max_seq_len):
    problems_eval = []
    answers_eval = []
    for i, example in enumerate(dataset):
        text = (f'<start_of_turn>user\n'
                f'{example["problem"]}\n'
                f'<end_of_turn>\n'
                f'<start_of_turn>model\n')
        problems_eval += [text]
        answers_eval += [example['answer']]
    answers_eval = np.array(answers_eval)
    tokens_eval = tokenize(problems_eval, vocab, max_seq_len)
    print(f'{tokens_eval.shape=}')
    eval_ds = (tokens_eval, answers_eval)
    return eval_ds


def benchmark_model(key, model, eval_ds, vocab, eval_batch_size, n_eval_samples):
    tokens_eval, answers_eval = eval_ds
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
