import math
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from datasets import load_dataset
from math_verify import parse, verify
from sampler import sample


def tokenize(sequences, vocab, seq_len, batch_divisor=1, pad_id=0):
    sequences_tokenized = vocab.EncodeAsIds(sequences)
    assert max(map(len, sequences_tokenized)) <= seq_len
    B, T = len(sequences), seq_len
    B = int(batch_divisor * math.ceil(B / batch_divisor)) # round B up to be divisible by `batch_divisor`
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
    prompts_eval = []
    problems_eval = []
    answers_eval = []
    for i, example in enumerate(ds_eval):
        prompt = (f'<start_of_turn>user\n'
                  f'{example["problem"]}\n'
                  f'<end_of_turn>\n'
                  f'<start_of_turn>model\n')
        prompts_eval += [prompt]
        problems_eval += [example["problem"]]
        answers_eval += [example['answer']]
    tokens_eval = tokenize(prompts_eval, vocab, eval_seq_len, batch_divisor)
    print(f'{tokens_eval.shape=}')
    
    return tokens_train, train_loss_mask, tokens_eval, np.array(problems_eval), np.array(answers_eval)


def benchmark_model(key, model, tokens, problems_eval, answers_eval, vocab, batch_size, n_eval_samples):
    key_decoding, key_questions = jax.random.split(key)
    eot_token = vocab.EncodeAsIds('<end_of_turn>')[0]
    mesh = model.in_embed.embedding.value.sharding.mesh
    n_batches = len(tokens) // batch_size
    sample_idxs = jax.random.choice(key_questions, len(tokens), shape=[n_batches, batch_size], replace=False)
    lengths_list = []
    correct_list = []
    finished_list = []
    for batch_idx in sample_idxs:
        tokens_batch = jax.device_put(tokens[batch_idx], NamedSharding(mesh, P('data', None)))
        completions_tokens = sample(key_decoding, model, tokens_batch)
        completions_text = vocab.DecodeIds(completions_tokens)
        for sample_idx, completion_tokens, completion_text in zip(batch_idx, completions_tokens, completions_text):
            if sample_idx < len(problems_eval):
                problem = problems_eval[sample_idx]
                gold = answers_eval[sample_idx]
                parsed = parse(completion_text)
                finished = eot_token in completion_tokens
                correct = verify(gold, parsed)
                lengths_list += [len(completion_tokens)]
                finished_list += [finished]
                correct_list += [correct]
                print('------------')
                print(f'PROMPT:\n{problem}\nCOMPLETION:\n{completion_text}\nPARSED: {parsed}\nGOLD: {gold}\nCORRECT: {correct}')

    return dict(length=np.median(lengths_list), finished=np.mean(finished_list), accuracy=np.mean(correct_list))
