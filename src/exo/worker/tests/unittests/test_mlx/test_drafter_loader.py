"""Tests for ``_maybe_load_drafter`` and the surrounding load path.

These tests exercise the policy-only branches of drafter loading so they can
run in CI without GPUs or downloaded model weights:

- Cards with no drafter return ``None``.
- Drafter weights missing from disk falls back to ``None`` (warned, not
  errored).
- ``EXO_DISABLE_DRAFTER`` short-circuits even when weights are present.

The "actually call ``mlx_lm.utils.load_model``" branch is exercised by the
end-to-end smoke harness, not unit tests.
"""

from pathlib import Path
from typing import cast

import pytest

from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.memory import Memory
from exo.worker.engines.mlx import utils_mlx
from exo.worker.engines.mlx.types import Model


def _card_with_drafter(drafter_id: ModelId | None) -> ModelCard:
    return ModelCard(
        model_id=ModelId("mlx-community/test-target"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
        drafter_model_id=drafter_id,
    )


def test_maybe_load_drafter_returns_none_when_no_drafter_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    card = _card_with_drafter(None)

    def fail_resolve(*_args: object, **_kwargs: object) -> Path | None:
        raise AssertionError("resolve_existing_model should not be called")

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", fail_resolve)

    assert utils_mlx._maybe_load_drafter(card) is None  # pyright: ignore[reportPrivateUsage]


def test_maybe_load_drafter_returns_none_when_drafter_weights_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    card = _card_with_drafter(ModelId("mlx-community/missing-drafter"))

    def missing_resolve(_model_id: ModelId) -> Path | None:
        return None

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", missing_resolve)

    def fail_load(*_args: object, **_kwargs: object) -> tuple[Model, dict[str, object]]:
        raise AssertionError("load_model must not run when weights are missing")

    monkeypatch.setattr(utils_mlx, "load_model", fail_load)

    assert utils_mlx._maybe_load_drafter(card) is None  # pyright: ignore[reportPrivateUsage]


def test_maybe_load_drafter_disabled_by_env_skips_filesystem_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, "1")
    card = _card_with_drafter(ModelId("mlx-community/some-drafter"))

    def fail_resolve(*_args: object, **_kwargs: object) -> Path | None:
        raise AssertionError("resolve_existing_model must not run when disabled")

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", fail_resolve)

    assert utils_mlx._maybe_load_drafter(card) is None  # pyright: ignore[reportPrivateUsage]


def test_maybe_load_drafter_swallows_load_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A drafter present on disk that fails to load must not break the target."""
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    card = _card_with_drafter(ModelId("mlx-community/broken-drafter"))

    def fixed_resolve(_model_id: ModelId) -> Path | None:
        return tmp_path

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", fixed_resolve)

    def boom_load(*_args: object, **_kwargs: object) -> tuple[Model, dict[str, object]]:
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(utils_mlx, "load_model", boom_load)

    assert utils_mlx._maybe_load_drafter(card) is None  # pyright: ignore[reportPrivateUsage]


def test_maybe_load_drafter_returns_loaded_model_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    card = _card_with_drafter(ModelId("mlx-community/fake-drafter"))

    def fixed_resolve(_model_id: ModelId) -> Path | None:
        return tmp_path

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", fixed_resolve)

    sentinel = object()

    def fake_load(
        *_args: object, **_kwargs: object
    ) -> tuple[object, dict[str, object]]:
        return sentinel, {}

    def noop_eval(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(utils_mlx, "load_model", fake_load)
    monkeypatch.setattr(utils_mlx.mx, "eval", noop_eval)

    result = utils_mlx._maybe_load_drafter(card)  # pyright: ignore[reportPrivateUsage]
    assert result is cast(Model, sentinel)
