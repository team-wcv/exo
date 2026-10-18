"""Project full-model prefill state onto an already sharded decode model."""

from dataclasses import dataclass
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.worker.engines.mlx.types import KVCacheType, Model


@dataclass(frozen=True)
class CacheLayout:
    """Per-rank dimensions after the model's tensor-sharding transformation."""

    kv_heads: int = 0
    linear_key_dim: int = 0
    linear_value_dim: int = 0
    linear_value_heads: int = 0


def _positive_attribute(module: object, *names: str) -> int:
    for name in names:
        value = cast(object, getattr(module, name, None))
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise ValueError(
        f"Remote tensor prefill requires a positive dimension {names} on "
        f"{type(module).__name__}"
    )


def tensor_cache_layouts(model: Model, caches: KVCacheType) -> list[CacheLayout]:
    """Read cache dimensions from the local model, never from checkpoint globals.

    Qwen's convolution state concatenates Q, K and V channels. A contiguous
    split of that concatenation assigns different projections to each rank.
    Its PLE convolution/history and sparse-attention indexer remain replicated.
    """
    if len(model.layers) != len(caches):
        raise ValueError("Remote tensor prefill requires one cache per model layer")
    layouts: list[CacheLayout] = []
    for layer, cache in zip(model.layers, caches, strict=True):
        if isinstance(cache, (KVCache, RotatingKVCache)):
            attention = cast(object, getattr(layer, "self_attn", None))
            layouts.append(
                CacheLayout(
                    kv_heads=_positive_attribute(
                        attention, "n_kv_heads", "num_key_value_heads"
                    )
                )
            )
        elif isinstance(cache, ArraysCache):
            attention = cast(object, getattr(layer, "linear_attn", None))
            layouts.append(
                CacheLayout(
                    linear_key_dim=_positive_attribute(attention, "key_dim"),
                    linear_value_dim=_positive_attribute(attention, "value_dim"),
                    linear_value_heads=_positive_attribute(
                        attention, "n_v", "num_v_heads"
                    ),
                )
            )
        else:
            raise ValueError(
                f"Remote tensor prefill does not support {type(cache).__name__}"
            )
    return layouts


def project_kv_heads(
    tensor: mx.array, local_heads: int, rank: int, world_size: int
) -> mx.array:
    """Select this rank's heads, including replicated GQA heads."""
    heads = int(tensor.shape[1])
    if heads == local_heads * world_size:
        start = rank * local_heads
    elif local_heads == 1 and 0 < heads < world_size and world_size % heads == 0:
        start = rank // (world_size // heads)
    else:
        raise ValueError(
            f"Remote KV has {heads} heads; rank {rank}/{world_size} expects "
            f"{local_heads} local heads"
        )
    return mx.contiguous(tensor[:, start : start + local_heads, :])


def project_linear_state(
    state: list[mx.array | None], layout: CacheLayout, rank: int, world_size: int
) -> list[mx.array | None]:
    """Shard Gated DeltaNet state and preserve replicated auxiliary slots."""
    if len(state) < 2:
        raise ValueError(
            "Remote linear attention state needs convolution and SSM slots"
        )
    result = list(state)
    convolution, recurrent = state[:2]
    key_width = layout.linear_key_dim
    value_width = layout.linear_value_dim
    if convolution is not None:
        expected = (2 * key_width + value_width) * world_size
        if convolution.ndim != 3 or convolution.shape[-1] != expected:
            raise ValueError(
                f"Remote convolution shape {convolution.shape} does not match "
                f"{expected} full Q/K/V channels"
            )
        global_key_width = key_width * world_size
        result[0] = mx.contiguous(
            mx.concatenate(
                [
                    convolution[..., rank * key_width : (rank + 1) * key_width],
                    convolution[
                        ...,
                        global_key_width + rank * key_width : global_key_width
                        + (rank + 1) * key_width,
                    ],
                    convolution[
                        ...,
                        2 * global_key_width + rank * value_width : 2 * global_key_width
                        + (rank + 1) * value_width,
                    ],
                ],
                axis=-1,
            )
        )
    if recurrent is not None:
        heads = layout.linear_value_heads
        if recurrent.ndim != 4 or recurrent.shape[1] != heads * world_size:
            raise ValueError(
                f"Remote SSM shape {recurrent.shape} does not match "
                f"{heads * world_size} full value heads"
            )
        result[1] = mx.contiguous(recurrent[:, rank * heads : (rank + 1) * heads])
    return result
