"""based on https://github.com/google/flax/tree/main/examples/gemma"""

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import sentencepiece as spm


def _compute_attention_masks(time_step, seq_len, input_mask):
  """Computes causal attention mask."""
  batch_size = input_mask.shape[0]
  batch_time_step = jnp.full((batch_size, 1), time_step, dtype=jnp.uint32)
  causal_padding = jnp.greater(jnp.expand_dims(jnp.arange(seq_len), 0), batch_time_step)
  max_seq_len = min(input_mask.shape[-1], seq_len)
  input_mask = jax.lax.dynamic_slice(input_mask,(0, jnp.maximum(time_step - seq_len + 1, 0)),(batch_size, max_seq_len))
  input_mask = (jnp.zeros((batch_size, seq_len), dtype=jnp.bool_).at[:, :max_seq_len].set(input_mask))
  causal_padding = jnp.logical_or(causal_padding, input_mask)
  attention_mask = causal_padding[:, jnp.newaxis, :].astype(jnp.bool_)
  return ~attention_mask


@flax.struct.dataclass
class _SamplingState:
  decoding_step: jnp.int32 # Decoding step.
  num_input_tokens: jnp.ndarray  # [B], Number of tokens in the prompt.
  token_buffer: jnp.ndarray  # [B, T], Fixed-size buffer for accumulating the output tokens.
  positions: jnp.ndarray  # [B, T], # Position indices, based on ignoring pad tokens.
  cache: dict # Model state for conditioning the model on autoregressively.
  done: jnp.ndarray  # [B], # Is decoding done on the given sequence?


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
    step_positions = state.positions[:, state.decoding_step, None] # [B, 1]
    input_mask = state.token_buffer == self.vocab.pad_id() # [B, T]
    attention_mask = _compute_attention_masks(state.decoding_step, self.max_seq_len, input_mask) # [B, 1, T]
    transformer = nnx.merge(self.transformer_graphdef, params)
    logits, cache = transformer(last_token, step_positions, attention_mask, state.cache) # [B, 1, V]

    # sample tokens
    next_token = jnp.argmax(logits, axis=-1)[:, 0] # [B]
    next_token = jnp.where(
        state.done | (state.decoding_step < state.num_input_tokens - 1), # done or input seq
        state.token_buffer[:, state.decoding_step+1], # do not update
        next_token, # update
    )

    token_buffer = state.token_buffer.at[:, state.decoding_step + 1].set(next_token)
    eos_ids = jnp.array([self.vocab.eos_id(), self.vocab.EncodeAsIds('<end_of_turn>')[0]])
    done = state.done | jnp.isin(next_token, eos_ids)

    return _SamplingState(
        decoding_step=state.decoding_step + 1,
        num_input_tokens=state.num_input_tokens,
        token_buffer=token_buffer,
        positions=state.positions,
        cache=cache,
        done=done,
    )

  def init_sample_state(self, all_input_ids: list[jax.Array]):
    """Initializes the sampling state given input prompts."""
    B = len(all_input_ids)
    num_input_tokens = jnp.array([len(ids) for ids in all_input_ids], dtype=jnp.int32)
    token_buffer = jnp.full((B, self.max_seq_len), self.vocab.pad_id(), dtype=jnp.int32)
    input_mask = jnp.ones_like(token_buffer, dtype=jnp.bool_)
    for i, (input_ids, num_tokens) in enumerate(zip(all_input_ids, num_input_tokens)):
      token_buffer = token_buffer.at[i, :num_tokens].set(input_ids)
      input_mask = input_mask.at[i, :num_tokens].set(input_ids != self.vocab.pad_id())
    positions = jnp.cumsum(input_mask, axis=-1)
    positions -= (positions >= 1) # Subtract one for all positions from the first valid one as they are 0-indexed

    return _SamplingState(
        decoding_step=0,
        num_input_tokens=num_input_tokens,
        token_buffer=token_buffer,
        positions=positions,
        cache=self.transformer.init_cache(self.max_seq_len, B),
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
