"""Tests for MlxBuilder selecting the right Engine based on drafter presence.

These tests stub out the heavy MLX paths (model load, KVPrefixCache,
tokenizer probing) and just exercise the routing logic in ``MlxBuilder.build``:

- No drafter, batching enabled (default): ``BatchGenerator``.
- No drafter, ``EXO_NO_BATCH`` set: ``SequentialGenerator``.
- Drafter loaded, batching enabled: ``SequentialGenerator`` is forced because
  upstream ``BatchGenerator`` does not support speculative decoding.
- Drafter is threaded through into the SequentialGenerator's ``draft_model``
  field so ``mlx_generate`` can pass it to ``stream_generate``.
"""

from typing import cast
from unittest.mock import MagicMock

import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.mlx.builder import MlxBuilder
from exo.worker.engines.mlx.types import Model
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    SequentialGenerator,
)


def _build_mlx_builder(
    *,
    draft_model: Model | None,
    draft_model_id: ModelId | None = None,
) -> MlxBuilder:
    fake_tokenizer = MagicMock(spec=TokenizerWrapper)
    fake_tokenizer.has_tool_calling = False
    fake_tokenizer.tool_call_start = None
    fake_tokenizer.tool_call_end = None
    fake_tokenizer.tool_parser = None

    return MlxBuilder(
        model_id=ModelId("mlx-community/test-target"),
        event_sender=cast(MpSender[Event], MagicMock()),
        cancel_receiver=cast(MpReceiver[TaskId], MagicMock()),
        inference_model=cast(Model, MagicMock()),
        tokenizer=fake_tokenizer,
        group=None,
        vision_processor=None,
        draft_model=draft_model,
        draft_model_id=draft_model_id,
    )


def test_mlx_builder_uses_batch_generator_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    builder = _build_mlx_builder(draft_model=None)
    engine = builder.build()
    assert isinstance(engine, BatchGenerator)


def test_mlx_builder_uses_sequential_when_no_batch_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_NO_BATCH", "1")
    builder = _build_mlx_builder(draft_model=None)
    engine = builder.build()
    assert isinstance(engine, SequentialGenerator)
    assert engine.draft_model is None


def test_mlx_builder_forces_sequential_when_drafter_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a drafter model is present, BatchGenerator can't use it, so we must
    fall back to SequentialGenerator regardless of EXO_NO_BATCH."""
    monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    monkeypatch.delenv("EXO_NUM_DRAFT_TOKENS", raising=False)
    monkeypatch.delenv("EXO_DRAFTER_MIN_OUTPUT_TOKENS", raising=False)
    fake_drafter = cast(Model, MagicMock())
    drafter_id = ModelId("mlx-community/test-drafter")
    builder = _build_mlx_builder(draft_model=fake_drafter, draft_model_id=drafter_id)

    engine = builder.build()

    assert isinstance(engine, SequentialGenerator)
    assert engine.draft_model is fake_drafter
    assert engine.draft_model_id == drafter_id
    # Defaults should be applied so dashboards see the actual K in use.
    assert engine.num_draft_tokens is not None and engine.num_draft_tokens >= 2
    assert (
        engine.drafter_min_output_tokens is not None
        and engine.drafter_min_output_tokens > 0
    )


def test_mlx_builder_honours_env_overrides_for_drafter_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_NUM_DRAFT_TOKENS", "7")
    monkeypatch.setenv("EXO_DRAFTER_MIN_OUTPUT_TOKENS", "32")
    fake_drafter = cast(Model, MagicMock())
    builder = _build_mlx_builder(
        draft_model=fake_drafter,
        draft_model_id=ModelId("mlx-community/test-drafter"),
    )

    engine = builder.build()

    assert isinstance(engine, SequentialGenerator)
    assert engine.num_draft_tokens == 7
    assert engine.drafter_min_output_tokens == 32
