"""based on https://github.com/google/flax/tree/main/examples/gemma"""

import dataclasses
from itertools import cycle

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P, NamedSharding
from .rope import apply_rope


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
  """Configuration for the gemma transformer."""
  num_layers: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  query_pre_attn_scalar: float
  num_embed: int = 262_144
  local_base_frequency: int = 10_000
  global_base_frequency: int = 1_000_000
  local_scale_factor: float = 1.0
  global_scale_factor: float = 1.0
  sliding_window_size: int | None = None
  attention_pattern: tuple[str] = ('sliding', 'sliding', 'sliding', 'sliding', 'sliding', 'global')


  @classmethod
  def gemma3_1b(cls):
    return cls(
        num_layers=26,
        embed_dim=1152,
        hidden_dim=6 * 1152,
        num_heads=4,
        head_dim=256,
        num_kv_heads=1,
        query_pre_attn_scalar=256**-0.5, # 1/sqrt(head_dim)
        sliding_window_size=512,
        global_scale_factor=1.0,
    )

  @classmethod
  def gemma3_4b(cls):
    return cls(
        num_layers=34,
        embed_dim=2560,
        hidden_dim=2560 * 8 // 2,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        query_pre_attn_scalar=256**-0.5, # 1/sqrt(head_dim)
        sliding_window_size=1024,
        global_scale_factor=8.0,
    )

  @classmethod
  def gemma3_12b(cls):
    return cls(
        num_layers=48,
        embed_dim=30 * 128,
        hidden_dim=8 * 30 * 128 // 2,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        query_pre_attn_scalar=256**-0.5, # 1/sqrt(head_dim)
        sliding_window_size=1024,
        global_scale_factor=8.0,
    )

  @classmethod
  def gemma3_27b(cls):
    return cls(
        num_layers=62,
        embed_dim=5376,
        hidden_dim=5376 * 8 // 2,
        num_heads=32,
        head_dim=128,
        num_kv_heads=16,
        query_pre_attn_scalar=(5376/32)**-0.5, # 1/sqrt(embed_dim / num_heads)
        sliding_window_size=1024,
        global_scale_factor=8.0,
    )


class Transformer(nnx.Module):
  """Gemma transformer."""

  def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
    self.embedder = Embedder(config.num_embed, config.embed_dim, rngs=rngs)
    self.layers = [
        Block(
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            embed_dim=config.embed_dim,
            head_dim=config.head_dim,
            hidden_dim=config.hidden_dim,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            sliding_window_size=config.sliding_window_size if attn_type == 'sliding' else None,
            rope_base_frequency=config.local_base_frequency if attn_type == 'sliding' else config.global_base_frequency,
            rope_scale_factor=config.local_scale_factor if attn_type == 'sliding' else config.global_scale_factor,
            rngs=rngs,
        ) for _, attn_type in zip(range(config.num_layers), cycle(config.attention_pattern))
    ]
    self.final_norm = nnx.RMSNorm(config.embed_dim, rngs=rngs)

  def __call__(
      self,
      tokens, # [B, T]
      kv_cache = None, # [S]
  ):
    x = self.embedder.encode(tokens) # [B, T, D]

    for i, layer in enumerate(self.layers):
      x, kv_cache[i] = layer(x, kv_cache.get(i)) # [B, T, D]

    x = self.final_norm(x)
    logits = self.embedder.decode(x) # [B, T, V]

    return logits, kv_cache


  def init_kv_cache(self, batch_size, max_seq_len):
    kv_cache = {
        i: layer.attn.init_kv_cache(batch_size, max_seq_len)
        for i, layer in enumerate(self.layers)
    }
    return kv_cache


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

  def __call__(self,
    x, # [B, T, D]
    kv_cache=None # [B, S]
  ):
    B, T, D = x.shape
    N, D, H = self.q_einsum.kernel.value.shape
    # K, D, H = self.kv_einsum.kernel.value.shape
    S = T if kv_cache is None else kv_cache['v'].shape[1]

    # qkv projection
    query = self.q_einsum(x) # [B, T, N, H]
    key, value = self.kv_einsum(x) # [B, T, N, H]

    # qk norm
    query = self._query_norm(query)
    key = self._key_norm(key)

    # get token indices
    if kv_cache is None: # training
      positions = jnp.arange(T)[None, :].repeat([B, 1]) # [B, S]
    else: # sampling
      positions = jnp.full([B, 1], kv_cache['end_idx']) # [B, 1]

    # apply positional embedding
    query = apply_rope(query, positions, self.rope_base_frequency, self.rope_scale_factor)
    key = apply_rope(key, positions, self.rope_base_frequency, self.rope_scale_factor)

    # load kv cache
    if kv_cache is not None:
      key = kv_cache['k'].at[:, kv_cache['end_idx'], :, :].set(key[:, 0]) # [B, S, N, H]
      value = kv_cache['v'].at[:, kv_cache['end_idx'], :, :].set(value[:, 0]) # [B, S, N, H]

    # compute attention mask [B, T, S]
    if kv_cache is None: # if training, use trinagular mask
      attn_mask = jnp.tri(T, dtype=jnp.bool_) # [T, S]
    else: # if sampling, all cached tokens should be visible
      attn_mask = (jnp.arange(S) <= kv_cache['end_idx'])[None] # [1, S]

    # add window to attention mask (TODO)
    if self.sliding_window_size is not None:
      all_ones = jnp.ones_like(attn_mask) # [T, S]
      sliding_mask = jnp.triu(all_ones, -1 * self.sliding_window_size + 1) & jnp.tril(all_ones, self.sliding_window_size - 1)
      attn_mask &= sliding_mask

    # gqa attention
    attn_mask = jnp.broadcast_to(attn_mask[None, None, :, :], [B, N, T, S])
    encoded = jax.nn.dot_product_attention(query, key, value, mask=attn_mask, scale=self.query_pre_attn_scalar)
    
    # output projection
    attn_output = self.attn_vec_einsum(encoded)

    # update kv cache
    if kv_cache is not None:
      kv_cache = {'k': key, 'v': value, 'end_idx': kv_cache['end_idx'] + T}

    return attn_output, kv_cache

  def init_kv_cache(self, batch_size, max_seq_len):
    mesh = self.kv_einsum.kernel.value.sharding.mesh
    _, num_kv_heads, _, head_dim = self.kv_einsum.kernel.value.shape
    sharding = NamedSharding(mesh, P('data', None, 'model', None))
    kv_cache = {
        'k': jnp.zeros((batch_size, max_seq_len, num_kv_heads, head_dim), dtype=jnp.float32, device=sharding),
        'v': jnp.zeros((batch_size, max_seq_len, num_kv_heads, head_dim), dtype=jnp.float32, device=sharding),
        'end_idx': jnp.array(0, dtype=jnp.int32),
    }
    return kv_cache


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

  def __call__(self, x, kv_cache=None):

    # Attention.
    attn_inputs = self.pre_attention_norm(x)
    attn_output, kv_cache = self.attn(attn_inputs, kv_cache)
    attn_output = self.post_attention_norm(attn_output)
    x += attn_output

    # Feed forward.
    ffw_inputs = self.pre_ffw_norm(x)
    ffw_outputs = self.mlp(ffw_inputs)
    ffw_outputs = self.post_ffw_norm(ffw_outputs)
    x += ffw_outputs

    return x, kv_cache
