"""Resilience tests for :class:`SequentialGenerator`.

Regression coverage for PR #15: a per-task ``ValueError`` raised during
drafter construction (e.g. K above the transport's wire-protocol budget)
must not propagate out of ``step()`` and crash the runner subprocess.
The pre-fix behaviour was that ``_start_next`` re-raised after sending
the error chunk, which propagated through ``handle_generation_tasks``
and triggered ``RunnerFailed`` on the supervisor, leaving the peer rank
wedged in ``RunnerRunning`` while the respawned target sat in
``RunnerIdle`` forever.

These tests bypass the SequentialGenerator dataclass __init__ (which
needs a full MLX model + tokenizer stack) and patch only the failing
hot-spot, mirroring the pattern used by ``test_batch_generator_errors``.
"""

from __future__ import annotations

from collections import deque
from typing import Any, cast

import pytest

from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.tasks import TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.utils.channels import MpSender
from exo.worker.runner.llm_inference.batch_generator import (
    FinishedResponse,
    SequentialGenerator,
)


class _FakeEventSender:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)


def _make_text_task(text: str = "hello") -> TextGeneration:
    return TextGeneration(
        instance_id=InstanceId("instance"),
        command_id=CommandId(f"command-{text}"),
        task_params=TextGenerationTaskParams(
            model=ModelId("mlx-community/test-model"),
            input=[
                InputMessage(role="user", content=InputMessageContent(text)),
            ],
        ),
    )


def _bare_sequential_generator(
    sender: _FakeEventSender,
    queue: deque[TextGeneration],
) -> SequentialGenerator:
    """Construct a :class:`SequentialGenerator` without running its dataclass init.

    Only the attributes touched by ``step()`` / ``_start_next()`` /
    ``_send_error()`` are wired in, so the test stays MLX-free and focused
    on the resilience contract.
    """
    generator = object.__new__(SequentialGenerator)
    generator.model_id = ModelId("mlx-community/test-model")
    generator.device_rank = 0
    generator.tokenizer = cast(Any, object())
    generator.event_sender = cast(MpSender[Event], cast(object, sender))
    generator.group = None
    generator._maybe_queue = []  # pyright: ignore[reportPrivateUsage]
    generator._maybe_cancel = []  # pyright: ignore[reportPrivateUsage]
    generator._all_tasks = {  # pyright: ignore[reportPrivateUsage]
        task.task_id: task for task in queue
    }
    generator._queue = queue  # pyright: ignore[reportPrivateUsage]
    generator._cancelled_tasks = set()  # pyright: ignore[reportPrivateUsage]
    generator._active = None  # pyright: ignore[reportPrivateUsage]
    generator._pending_failed = []  # pyright: ignore[reportPrivateUsage]
    generator._recent_acceptance = deque()  # pyright: ignore[reportPrivateUsage]
    generator.adaptive_draft_tokens = False
    return generator


def test_start_next_failure_emits_finished_and_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drafter construction failure must surface as ``FinishedResponse``."""
    sender = _FakeEventSender()
    task = _make_text_task("first")
    generator = _bare_sequential_generator(sender, deque([task]))

    def boom(_self: SequentialGenerator, _task: TextGeneration) -> None:
        raise ValueError("num_draft_tokens (8) exceeds transport's max (5)")

    def no_agree(_self: SequentialGenerator) -> None:
        return None

    monkeypatch.setattr(
        SequentialGenerator,
        "_build_generator",
        boom,
    )
    monkeypatch.setattr(
        SequentialGenerator,
        "agree_on_tasks",
        no_agree,
    )

    results = list(generator.step())

    assert len(results) >= 1
    assert results[0][0] == task.task_id
    assert isinstance(results[0][1], FinishedResponse)
    assert (
        generator._active is None  # pyright: ignore[reportPrivateUsage]
    ), "no active task should be set after failed _start_next"
    assert len(sender.events) == 1
    assert isinstance(sender.events[0], ChunkGenerated)
    assert isinstance(sender.events[0].chunk, ErrorChunk)
    assert "num_draft_tokens" in sender.events[0].chunk.error_message


def test_runner_survives_sequential_failure_and_serves_next_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a per-task failure the runner must still serve the next task.

    This is the core regression: pre-fix, the first task's failure
    propagated out of ``step()`` and tore down the runner subprocess, so
    the second task never got a chance to run. We use two failing tasks
    so the test stays MLX-free; what matters is that ``step()`` survives
    both failures and surfaces them as ``FinishedResponse`` rather than
    propagating an exception out of the runner loop.
    """
    sender = _FakeEventSender()
    first = _make_text_task("first")
    second = _make_text_task("second")
    generator = _bare_sequential_generator(sender, deque([first, second]))

    call_log: list[str] = []

    def boom(_self: SequentialGenerator, task: TextGeneration) -> object:
        call_log.append(str(task.task_id))
        raise ValueError("num_draft_tokens (8) exceeds transport's max (5)")

    def no_agree(_self: SequentialGenerator) -> None:
        return None

    monkeypatch.setattr(
        SequentialGenerator,
        "_build_generator",
        boom,
    )
    monkeypatch.setattr(
        SequentialGenerator,
        "agree_on_tasks",
        no_agree,
    )

    tick1 = list(generator.step())
    assert any(
        result[0] == first.task_id and isinstance(result[1], FinishedResponse)
        for result in tick1
    ), "first failed task must surface as FinishedResponse on tick 1"

    tick2 = list(generator.step())
    assert any(
        result[0] == second.task_id and isinstance(result[1], FinishedResponse)
        for result in tick2
    ), "second task must be popped and finished on tick 2 (runner survived)"

    assert call_log == [str(first.task_id), str(second.task_id)], (
        "both tasks must reach _build_generator -- pre-fix the first "
        "failure propagated and the second task never got a chance"
    )
    assert len(sender.events) == 2, "both failures must emit ErrorChunks"


def test_step_exception_during_next_does_not_raise() -> None:
    """An exception during ``next(gen)`` mid-stream must surface as Finished, not crash."""
    sender = _FakeEventSender()
    task = _make_text_task()
    generator = _bare_sequential_generator(sender, deque())

    class _BoomError(Exception):
        pass

    def faulty_gen() -> object:
        raise _BoomError("runtime fault inside spec loop")
        yield  # pyright: ignore[reportUnreachable]

    queue: object = object()
    output_generator: object = iter([])
    generator._active = (  # pyright: ignore[reportPrivateUsage]
        task,
        cast(Any, faulty_gen()),
        cast(Any, queue),
        cast(Any, output_generator),
    )

    results = list(generator.step())

    assert any(
        result[0] == task.task_id and isinstance(result[1], FinishedResponse)
        for result in results
    )
    assert (
        generator._active is None  # pyright: ignore[reportPrivateUsage]
    )
    assert len(sender.events) == 1
    assert isinstance(sender.events[0], ChunkGenerated)
    assert isinstance(sender.events[0].chunk, ErrorChunk)
    assert "runtime fault" in sender.events[0].chunk.error_message
