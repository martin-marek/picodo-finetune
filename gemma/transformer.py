"""based on https://github.com/google/flax/tree/main/examples/gemma"""

import dataclasses
from itertools import cycle

import jax
import jax.numpy as jnp
from flax import nnx
from . import modules


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
    self.embedder = modules.Embedder(config.num_embed, config.embed_dim, rngs=rngs)
    self.layers = [
        modules.Block(
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
      tokens,  # [B, T]
      positions,  # [B, T]
      attention_mask,  # [B, T, T']
      cache = None,  # [T']
  ):
    x = self.embedder.encode(tokens) # [B, T, D]

    for i, layer in enumerate(self.layers):
      layer_cache = cache[f'layer_{i}'] if cache else None
      layer_cache, x = layer(x, positions, attention_mask, layer_cache) # [B, T, D]
      cache[f'layer_{i}'] = layer_cache

    x = self.final_norm(x)
    logits = self.embedder.decode(x) # [B, T, V]

    return logits, cache


  def init_cache(self, cache_size, batch_size):
    return {
        f'layer_{i}': layer.attn.init_cache(cache_size, batch_size)
        for i, layer in enumerate(self.layers)
    }
