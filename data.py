import numpy as np
import jax.numpy as jnp
import sentencepiece as spm


def tokenize(sequences, vocab, seq_len, pad_id=0):
    sequences_tokenized = vocab.EncodeAsIds(sequences)
    assert max(map(len, sequences_tokenized)) <= seq_len
    B, T = len(sequences), seq_len
    tokens = np.full([B, T], pad_id, dtype=jnp.int32)

    for i, seq_tok in enumerate(sequences_tokenized):
        tokens[i, :len(seq_tok)] = seq_tok

    return tokens
