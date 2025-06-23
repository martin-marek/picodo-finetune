"""based on https://github.com/google/flax/tree/main/examples/gemma"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P, NamedSharding
from .rope import apply_rope


class Embedder(nnx.Module):
  """Embedder module."""

  def __init__(self, vocab_size, embed_dim, rngs):
    self.input_embedding = nnx.Param(jax.nn.initializers.normal()(rngs.params(), (vocab_size, embed_dim)))

  def encode(self, x):
    x = self.input_embedding[(x,)]
    x *= jnp.sqrt(x.shape[-1]).astype(x.dtype)
    return x

  def decode(self, x):
    return jnp.dot(x, self.input_embedding.value.T)


class Attention(nnx.Module):
  """Attention module."""

  def __init__(
      self,
      num_heads: int,
      num_kv_heads: int,
      embed_dim: int,
      head_dim: int,
      query_pre_attn_scalar: float,
      rngs: nnx.Rngs,
      rope_base_frequency: int,
      rope_scale_factor: float,
      sliding_window_size: int | None = None,
  ):
    self.query_pre_attn_scalar = query_pre_attn_scalar
    self.sliding_window_size = sliding_window_size
    self.rope_base_frequency = rope_base_frequency
    self.rope_scale_factor = rope_scale_factor
    self.attn_vec_einsum = nnx.Einsum(einsum_str='BTNH,NHD->BTD', kernel_shape=(num_heads, head_dim, embed_dim), rngs=rngs)
    self.q_einsum = nnx.Einsum(einsum_str='BTD,NDH->BTNH', kernel_shape=(num_heads, embed_dim, head_dim), rngs=rngs)
    self.kv_einsum = nnx.Einsum(einsum_str='BSD,CKDH->CBSKH', kernel_shape=(2, num_kv_heads, embed_dim, head_dim), rngs=rngs)
    self._query_norm = nnx.RMSNorm(head_dim, rngs=rngs)
    self._key_norm = nnx.RMSNorm(head_dim, rngs=rngs)

  def __call__(self, x, segment_pos, cache, attn_mask):
    batch_size, in_len, embed_dim = x.shape
    num_heads, embed_dim, head_dim = self.q_einsum.kernel.value.shape

    # qkv projection
    query = self.q_einsum(x) # [batch_size, in_len, num_heads, head_dim]
    key, value = self.kv_einsum(x) # [batch_size, seq_len, num_kv_head, head_dim]

    # qk norm
    query = self._query_norm(query)
    key = self._key_norm(key)

    # positional embedding
    query = apply_rope(query, segment_pos, self.rope_base_frequency, self.rope_scale_factor)
    key = apply_rope(key, segment_pos, self.rope_base_frequency, self.rope_scale_factor)

    # cache (left aligned)
    if cache is not None:
      end_index = cache['end_index'][0]
      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      key = jax.lax.dynamic_update_slice(cache['k'], key, slice_indices)
      value = jax.lax.dynamic_update_slice(cache['v'], value, slice_indices)

    # add window to attention mask
    if self.sliding_window_size is not None:
      all_ones = jnp.ones_like(attn_mask) # [batch_size, in_len, seq_len]
      sliding_mask = jnp.triu(all_ones, -1 * self.sliding_window_size + 1) * jnp.tril(all_ones, self.sliding_window_size - 1)
      attn_mask = sliding_mask * attn_mask

    # gqa attention
    # print(query.shape, key.shape, value.shape, attn_mask.shape)
    # query *= self.query_pre_attn_scalar
    # logits = jnp.einsum('BTNH,BSNH->BTNS', query, key)
    # padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, -1e38)
    # probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
    # encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value)
    attn_mask = attn_mask[:, None, :, :].repeat(num_heads, 1) # [batch_size, num_heads, in_len, seq_len]
    encoded = jax.nn.dot_product_attention(query, key, value, mask=attn_mask, is_causal=False, scale=self.query_pre_attn_scalar)
    
    # output projection
    attn_output = self.attn_vec_einsum(encoded)

    if cache is not None:
      cache = {'k': key, 'v': value, 'end_index': cache['end_index'] + in_len}

    return cache, attn_output

  def init_cache(self, cache_size, batch_size):
    mesh = self.kv_einsum.kernel.value.sharding.mesh
    _, num_kv_heads, _, head_dim = self.kv_einsum.kernel.value.shape
    kv_sharding = NamedSharding(mesh, P('data', None, 'model', None))
    cache = {
        'k': jnp.zeros((batch_size, cache_size, num_kv_heads, head_dim), dtype=jnp.float32, device=kv_sharding),
        'v': jnp.zeros((batch_size, cache_size, num_kv_heads, head_dim), dtype=jnp.float32, device=kv_sharding),
        'end_index': jnp.zeros((batch_size,), dtype=jnp.int32, device=NamedSharding(mesh, P('data'))),
    }
    return cache


class FeedForward(nnx.Module):
  """Feed forward module."""

  def __init__(self, embed_dim, hidden_dim, rngs):
    self.gate_proj = nnx.Linear(embed_dim, hidden_dim, use_bias=False, rngs=rngs, kernel_init=jax.nn.initializers.normal())
    self.up_proj = nnx.Linear(embed_dim, hidden_dim, use_bias=False, rngs=rngs, kernel_init=jax.nn.initializers.normal())
    self.down_proj = nnx.Linear(hidden_dim, embed_dim, use_bias=False, rngs=rngs, kernel_init=jax.nn.initializers.normal())

  def __call__(self, x):
    activations = nnx.gelu(self.gate_proj(x)) * self.up_proj(x)
    outputs = self.down_proj(activations)
    return outputs


class Block(nnx.Module):
  """Transformer block."""

  def __init__(
      self,
      num_heads: int,
      num_kv_heads: int,
      embed_dim: int,
      head_dim: int,
      hidden_dim: int,
      query_pre_attn_scalar: float,
      rope_base_frequency: int,
      rope_scale_factor: float,
      sliding_window_size: int | None = None,
      *, rngs: nnx.Rngs,
  ):
    self.attn = Attention(
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        embed_dim=embed_dim, head_dim=head_dim, query_pre_attn_scalar=query_pre_attn_scalar,
        rope_base_frequency=rope_base_frequency, rope_scale_factor=rope_scale_factor,
        sliding_window_size=sliding_window_size, rngs=rngs,
    )
    self.mlp = FeedForward(embed_dim, hidden_dim, rngs=rngs)
    self.pre_attention_norm = nnx.RMSNorm(embed_dim, rngs=rngs)
    self.post_attention_norm = nnx.RMSNorm(embed_dim, rngs=rngs)
    self.pre_ffw_norm = nnx.RMSNorm(embed_dim, rngs=rngs)
    self.post_ffw_norm = nnx.RMSNorm(embed_dim, rngs=rngs)

  def __call__(self, x, segment_pos, attn_mask, cache=None):

    # Attention.
    attn_inputs = self.pre_attention_norm(x)
    cache, attn_output = self.attn(attn_inputs, segment_pos, cache, attn_mask)
    attn_output = self.post_attention_norm(attn_output)
    x += attn_output

    # Feed forward.
    ffw_inputs = self.pre_ffw_norm(x)
    ffw_outputs = self.mlp(ffw_inputs)
    ffw_outputs = self.post_ffw_norm(ffw_outputs)
    x += ffw_outputs

    return cache, x
