"""Opt-in YaRN for Qwen4-Exp's shared attention/indexer rotary embedding.

The remote MLX model currently reads rope_theta but ignores rope_type. Replace
only its rotary object when the checkpoint explicitly requests YaRN; ordinary
checkpoints and their caches retain the original implementation. The frequency
blend and attention factor follow Transformers' modeling_rope_utils YaRN.
"""

import math
from collections.abc import Mapping
from typing import Protocol, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _positive(parameters: Mapping[str, object], name: str, default: float) -> float:
    value = parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Qwen YaRN {name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Qwen YaRN {name} must be a positive finite number")
    return result


class QwenYarnRotaryEmbedding:
    """Return float32 cos/sin at absolute scalar or batched cache positions."""

    def __init__(self, dimension: int, parameters: Mapping[str, object]) -> None:
        if dimension <= 0 or dimension % 2:
            raise ValueError("Qwen YaRN rotary dimension must be positive and even")
        self.dim = dimension
        base = _positive(parameters, "rope_theta", 10_000_000)
        factor = _positive(parameters, "factor", 1)
        original = parameters.get("original_max_position_embeddings")
        if isinstance(original, bool) or not isinstance(original, int) or original <= 0:
            raise ValueError("Qwen YaRN requires original_max_position_embeddings > 0")
        if base <= 1 or factor < 1:
            raise ValueError("Qwen YaRN requires rope_theta > 1 and factor >= 1")
        beta_fast = _positive(parameters, "beta_fast", 32)
        beta_slow = _positive(parameters, "beta_slow", 1)
        if beta_fast < beta_slow:
            raise ValueError("Qwen YaRN beta_fast must be >= beta_slow")
        truncate = parameters.get("truncate", True)
        if not isinstance(truncate, bool):
            raise ValueError("Qwen YaRN truncate must be boolean")

        def correction(rotations: float) -> float:
            return (
                dimension
                * math.log(original / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low, high = correction(beta_fast), correction(beta_slow)
        if truncate:
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dimension - 1)
        if low == high:
            high += 0.001
        ramp = mx.clip(
            (mx.arange(dimension // 2, dtype=mx.float32) - low) / (high - low), 0, 1
        )
        # Metal fast-pow differs by an ULP on some frequencies. Absolute
        # positions near 512K magnify that into phase error. This tiny,
        # one-time CPU table matches the reference float32 initialization.
        frequencies = mx.array(
            np.float32(base)
            ** (np.arange(0, dimension, 2, dtype=np.float32) / dimension)
        )
        self.inv_freq = (1 / (factor * frequencies)) * ramp + (1 / frequencies) * (
            1 - ramp
        )
        amplitude = 1 + 0.1 * math.log(factor)
        if "mscale" in parameters and "mscale_all_dim" in parameters:
            numerator = 1 + 0.1 * _positive(parameters, "mscale", 1) * math.log(factor)
            denominator = 1 + 0.1 * _positive(
                parameters, "mscale_all_dim", 1
            ) * math.log(factor)
            amplitude = numerator / denominator
        self.attention_factor = _positive(parameters, "attention_factor", amplitude)

    def __call__(self, positions: mx.array) -> tuple[mx.array, mx.array]:
        frequencies = positions.astype(mx.float32)[..., None] * self.inv_freq
        embedding = mx.concatenate([frequencies, frequencies], axis=-1)
        return mx.cos(embedding) * self.attention_factor, mx.sin(
            embedding
        ) * self.attention_factor


class _RotaryOwner(Protocol):
    rope: object


def apply_qwen_yarn(model: nn.Module, config: Mapping[str, object]) -> bool:
    """Install explicit Qwen4-Exp YaRN before sharding or allocating KV caches."""
    if config.get("model_type") != "qwen4_exp":
        return False
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("Qwen4-Exp text_config must be an object")
    text_config = cast(Mapping[str, object], text)
    raw_parameters = text_config.get("rope_parameters", {})
    if not isinstance(raw_parameters, dict):
        raise ValueError("Qwen4-Exp rope_parameters must be an object")
    parameters = cast(Mapping[str, object], raw_parameters)
    rope_type = parameters.get("rope_type", "default")
    if rope_type == "default":
        return False
    if rope_type != "yarn":
        raise ValueError(f"Unsupported Qwen4-Exp rope_type: {rope_type!r}")
    if "factor" not in parameters:
        raise ValueError("Qwen YaRN requires an explicit factor")
    head = _positive(text_config, "head_dim", 256)
    partial = _positive(
        parameters,
        "partial_rotary_factor",
        _positive(text_config, "partial_rotary_factor", 0.25),
    )
    dimension = head * partial
    if partial > 1 or not dimension.is_integer():
        raise ValueError("Qwen YaRN partial rotary dimension is invalid")
    inner = cast(object, getattr(model, "model", None))
    existing = cast(object, getattr(inner, "rope", None))
    if inner is None or cast(object, getattr(existing, "dim", None)) != int(dimension):
        raise ValueError(
            "Qwen4-Exp model does not expose the expected shared rotary object"
        )
    cast(_RotaryOwner, inner).rope = QwenYarnRotaryEmbedding(int(dimension), parameters)
    return True
