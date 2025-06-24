"""based on https://github.com/google/flax/tree/main/examples/gemma"""

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece as spm


@flax.struct.dataclass
class _SamplingState:
    decoding_step: jnp.int32
    num_input_tokens: jnp.ndarray # [B] Number of tokens in given prompt.
    token_buffer: jnp.ndarray # [B, T], Fixed-size buffer for accumulating output tokens.
    kv_cache: dict
    done: jnp.ndarray # [B] # Is decoding done on the given sequence?


class Sampler:
    def __init__(self, transformer, vocab, max_seq_len=1024):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.transformer = transformer
        self.transformer_graphdef = nnx.graphdef(transformer)

    def _sample_step(self, params, state):
        """Samples next token in sequence."""

        # get logits for next token
        last_token = state.token_buffer[:, state.decoding_step, None] # [B, 1]
        transformer = nnx.merge(self.transformer_graphdef, params)
        logits, kv_cache = transformer(last_token, state.kv_cache) # [B, 1, V]

        # sample tokens
        next_token = jnp.argmax(logits, axis=-1)[:, 0] # [B]
        next_token = jnp.where(
            state.done | (state.decoding_step < state.num_input_tokens - 1), # done or insisde input
            state.token_buffer[:, state.decoding_step+1], # do not update
            next_token, # update
        )
        token_buffer = state.token_buffer.at[:, state.decoding_step + 1].set(next_token)

        # check if sampling is done
        eos_ids = jnp.array([self.vocab.eos_id(), self.vocab.EncodeAsIds('<end_of_turn>')[0]])
        done = state.done | jnp.isin(next_token, eos_ids)

        return _SamplingState(
            decoding_step=state.decoding_step + 1,
            num_input_tokens=state.num_input_tokens,
            token_buffer=token_buffer,
            kv_cache=kv_cache,
            done=done,
        )

    def init_sample_state(self, all_input_ids: list[jax.Array]):
        """Initializes the sampling state given input prompts."""
        B = len(all_input_ids)
        num_input_tokens = jnp.array([len(ids) for ids in all_input_ids], dtype=jnp.int32)
        token_buffer = np.full((B, self.max_seq_len), self.vocab.pad_id(), dtype=jnp.int32)
        for i, input_ids in enumerate(all_input_ids):
            token_buffer[i, :len(input_ids)] = input_ids

        return _SamplingState(
            decoding_step=0,
            num_input_tokens=num_input_tokens,
            token_buffer=jnp.array(token_buffer),
            kv_cache=self.transformer.init_kv_cache(B, self.max_seq_len),
            done=jnp.zeros((B,), dtype=jnp.bool_),
        )

    def tokenize(self, input_string):
        input_ids = self.vocab.EncodeAsIds(input_string)
        input_ids = jnp.array([self.vocab.bos_id()] + input_ids, dtype=jnp.int32)
        return input_ids

    def __call__(self, input_strings):
        # tokenize inputs
        all_input_ids = [self.tokenize(x) for x in input_strings]

        # sample tokens
        max_input_length = max(len(input_ids) for input_ids in all_input_ids)
        state = self.init_sample_state(all_input_ids)
        params = nnx.state(self.transformer) # we pass params to _sample_step, to avoid compiling them
        step_fn = lambda state: self._sample_step(params, state)
        cond_fn = lambda state: (state.decoding_step < self.max_seq_len) & jnp.any(~state.done)
        state = jax.lax.while_loop(cond_fn, step_fn, state)

        # convert tokens to text
        out_text = []
        for token_buffer, num_in_tokens in zip(state.token_buffer, state.num_input_tokens):
            out_text += [self.vocab.DecodeIds(token_buffer[num_in_tokens:].tolist())]

        return out_text
