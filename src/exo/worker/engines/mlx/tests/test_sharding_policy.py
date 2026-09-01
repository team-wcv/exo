"""Tests for MLX tensor parameter replication policy."""

import pytest

from exo.worker.engines.mlx.sharding_policy import (
    should_replicate_tensor_parameter,
)


@pytest.mark.parametrize(
    "path",
    [
        "model.layers.0.mlp.experts.codebook",
        "language_model.layers.3.experts.17.codebook",
        "codebook",
    ],
)
def test_vector_quantized_codebooks_are_replicated(path: str) -> None:
    assert should_replicate_tensor_parameter(path)


@pytest.mark.parametrize(
    "path",
    [
        "model.layers.0.mlp.experts.codes",
        "model.layers.0.mlp.experts.scales",
        "model.layers.0.mlp.codebook.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ],
)
def test_other_tensor_parameters_remain_shardable(path: str) -> None:
    assert not should_replicate_tensor_parameter(path)
