import math
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from exo.worker.engines.mlx.qwen_yarn import (
    QwenYarnRotaryEmbedding,
    apply_qwen_yarn,
    qwen_yarn_prefill_limits,
    thin_prefill_snapshots,
)


def parameters(factor: float = 2.0) -> dict[str, object]:
    return {
        "rope_type": "yarn",
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.25,
        "original_max_position_embeddings": 262144,
        "factor": factor,
    }


@pytest.mark.parametrize("factor", [1.0, 2.0, 4.0])
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"attention_factor": 1.2},
        {"truncate": False},
        {"mscale": 0.5, "mscale_all_dim": 1.0},
    ],
)
def test_yarn_matches_transformers(factor: float, overrides: dict[str, object]) -> None:
    values = {**parameters(factor), **overrides}
    config = PretrainedConfig()
    config.head_dim = 256
    config.hidden_size = 2560
    config.num_attention_heads = 24
    config.max_position_embeddings = 524288
    config.rope_parameters = values
    reference, amplitude = ROPE_INIT_FUNCTIONS["yarn"](config, torch.device("cpu"))
    rotary = QwenYarnRotaryEmbedding(64, values)
    np.testing.assert_allclose(
        np.array(rotary.inv_freq), reference.numpy(), rtol=2e-6, atol=1e-9
    )
    assert math.isclose(rotary.attention_factor, amplitude, rel_tol=1e-8)
    positions = mx.array([[0, 1, 2048, 262143, 262144, 524287]])
    cos, sin = rotary(positions)
    phase = (
        torch.tensor(np.array(positions), dtype=torch.float32)[..., None] * reference
    )
    phase = torch.cat([phase, phase], dim=-1)
    np.testing.assert_allclose(
        np.array(cos), (phase.cos() * amplitude).numpy(), atol=0.005, rtol=1e-4
    )
    np.testing.assert_allclose(
        np.array(sin), (phase.sin() * amplitude).numpy(), atol=0.005, rtol=1e-4
    )


def test_cached_positions_match_one_shot_past_native_limit() -> None:
    rotary = QwenYarnRotaryEmbedding(64, parameters())
    positions = mx.arange(262130, 262180)[None, :]
    whole = rotary(positions)
    chunks = [rotary(positions[:, :17]), rotary(positions[:, 17:])]
    for index in range(2):
        np.testing.assert_array_equal(
            np.array(whole[index]),
            np.array(mx.concatenate([chunk[index] for chunk in chunks], axis=1)),
        )
    cos, _ = rotary(mx.array([[524287], [262144]]))
    assert cos.shape == (2, 1, 64)


class Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rope: object = SimpleNamespace(dim=64)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Inner()


def test_default_and_unrelated_models_are_untouched() -> None:
    model = Model()
    original = model.model.rope
    assert not apply_qwen_yarn(model, {"model_type": "qwen3_5"})
    assert not apply_qwen_yarn(
        model,
        {
            "model_type": "qwen4_exp",
            "text_config": {"rope_parameters": {"rope_type": "default"}},
        },
    )
    assert model.model.rope is original


def test_yarn_replaces_shared_rotary_without_changing_weights() -> None:
    model = Model()
    before = model.parameters()
    assert apply_qwen_yarn(
        model,
        {
            "model_type": "qwen4_exp",
            "text_config": {"head_dim": 256, "rope_parameters": parameters()},
        },
    )
    assert isinstance(model.model.rope, QwenYarnRotaryEmbedding)
    assert model.model.rope.dim == 64
    assert model.parameters() == before
    assert qwen_yarn_prefill_limits(model) == (512, 64)


def test_checkpoint_thinning_preserves_replay_and_final_rollback_positions() -> None:
    snapshots: list[int] = []
    for position in range(1000):
        snapshots.append(position)
        thin_prefill_snapshots(snapshots, 64)
        assert len(snapshots) <= 64
        assert snapshots[0] == 0
        assert snapshots[-1] == position
        if position:
            assert snapshots[-2] == position - 1
        assert sorted(set(snapshots)) == snapshots
    assert qwen_yarn_prefill_limits(Model()) == (4096, None)


@pytest.mark.parametrize(
    "override",
    [
        {"factor": 0},
        {"factor": float("nan")},
        {"original_max_position_embeddings": 0},
        {"beta_fast": 0},
        {"truncate": "yes"},
    ],
)
def test_invalid_scaling_is_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        QwenYarnRotaryEmbedding(64, {**parameters(), **override})


def test_unsupported_rotary_mode_fails_instead_of_silent_unscaled_context() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        apply_qwen_yarn(
            Model(),
            {
                "model_type": "qwen4_exp",
                "text_config": {"rope_parameters": {"rope_type": "dynamic"}},
            },
        )
