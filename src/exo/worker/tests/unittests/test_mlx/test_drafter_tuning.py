"""Tests for drafter tuning knobs (num_draft_tokens, short-skip, env helpers).

End-to-end MLX inference can't run in unit tests (no GPUs/weights), so we
test the *policy* helpers that decide whether speculative decoding is active
and how many draft tokens to issue per round.
"""

from typing import cast

import pytest

from exo.worker.engines.mlx.generator.generate import resolve_speculative_decoding
from exo.worker.engines.mlx.types import Model
from exo.worker.runner.llm_inference.batch_generator import (
    DEFAULT_DRAFTER_MIN_OUTPUT_TOKENS,
    DEFAULT_NUM_DRAFT_TOKENS,
    EXO_DRAFTER_MIN_OUTPUT_TOKENS,
    EXO_NUM_DRAFT_TOKENS,
    parse_env_int,
)


def test_parse_env_int_returns_default_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_FAKE_VAR_FOR_TEST", raising=False)
    assert parse_env_int("EXO_FAKE_VAR_FOR_TEST", 5) == 5


def test_parse_env_int_clamps_to_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_FAKE_VAR_FOR_TEST", "0")
    assert parse_env_int("EXO_FAKE_VAR_FOR_TEST", 5, minimum=1) == 1


def test_parse_env_int_falls_back_on_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_FAKE_VAR_FOR_TEST", "not-a-number")
    assert parse_env_int("EXO_FAKE_VAR_FOR_TEST", 5) == 5


def test_parse_env_int_accepts_valid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_FAKE_VAR_FOR_TEST", "9")
    assert parse_env_int("EXO_FAKE_VAR_FOR_TEST", 5) == 9


def test_default_constants_are_sane() -> None:
    assert DEFAULT_NUM_DRAFT_TOKENS >= 2
    assert DEFAULT_DRAFTER_MIN_OUTPUT_TOKENS > 0
    assert EXO_NUM_DRAFT_TOKENS == "EXO_NUM_DRAFT_TOKENS"
    assert EXO_DRAFTER_MIN_OUTPUT_TOKENS == "EXO_DRAFTER_MIN_OUTPUT_TOKENS"


def _fake_model() -> Model:
    return cast(Model, object())


def test_resolve_speculative_decoding_distributed_drops_drafter() -> None:
    """Multi-device runs never pass the drafter through."""
    import mlx.core as mx

    drafter = _fake_model()
    fake_group = cast(mx.distributed.Group, object())
    eff, kwargs = resolve_speculative_decoding(
        draft_model=drafter,
        group=fake_group,
        max_tokens=128,
        num_draft_tokens=5,
        drafter_min_output_tokens=16,
    )
    assert eff is None
    assert kwargs == {}


def test_resolve_speculative_decoding_no_drafter_returns_empty_kwargs() -> None:
    eff, kwargs = resolve_speculative_decoding(
        draft_model=None,
        group=None,
        max_tokens=128,
        num_draft_tokens=5,
        drafter_min_output_tokens=16,
    )
    assert eff is None
    assert kwargs == {}


def test_resolve_speculative_decoding_short_max_tokens_drops_drafter() -> None:
    """Item 8: short generations skip the drafter."""
    drafter = _fake_model()
    eff, kwargs = resolve_speculative_decoding(
        draft_model=drafter,
        group=None,
        max_tokens=8,
        num_draft_tokens=5,
        drafter_min_output_tokens=16,
    )
    assert eff is None
    assert kwargs == {}


def test_resolve_speculative_decoding_threshold_boundary_drops_drafter() -> None:
    """``<=`` threshold means equality also skips the drafter."""
    drafter = _fake_model()
    eff, _ = resolve_speculative_decoding(
        draft_model=drafter,
        group=None,
        max_tokens=16,
        num_draft_tokens=5,
        drafter_min_output_tokens=16,
    )
    assert eff is None


def test_resolve_speculative_decoding_passes_k_through() -> None:
    """Item 1: num_draft_tokens flows into stream_generate kwargs."""
    drafter = _fake_model()
    eff, kwargs = resolve_speculative_decoding(
        draft_model=drafter,
        group=None,
        max_tokens=512,
        num_draft_tokens=5,
        drafter_min_output_tokens=16,
    )
    assert eff is drafter
    assert kwargs == {"num_draft_tokens": 5}


def test_resolve_speculative_decoding_no_k_means_no_kwarg() -> None:
    """If caller doesn't override K, mlx_lm uses its default (currently 2)."""
    drafter = _fake_model()
    eff, kwargs = resolve_speculative_decoding(
        draft_model=drafter,
        group=None,
        max_tokens=512,
        num_draft_tokens=None,
        drafter_min_output_tokens=16,
    )
    assert eff is drafter
    assert kwargs == {}
