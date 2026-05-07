"""Tests for :mod:`remote_drafter` -- wire protocol + transport behaviour.

These tests stay MLX-free where possible by exercising the pure
encoding helpers directly and by mocking ``mx.distributed.send/recv``
for the transport-level tests. End-to-end correctness against a real
``mx.distributed.Group`` is exercised by the twin-machine benchmark
runs (B10), not in unit tests.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import final

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.generator.remote_drafter import (
    ACK_FRAME_SIZE,
    ACK_OK,
    COMMAND_FRAME_SIZE,
    OP_FORWARD,
    OP_PREFILL,
    OP_SHUTDOWN,
    OP_TRIM_CACHE,
    RemoteTransport,
    _build_command_frame,  # type: ignore[reportPrivateUsage]
    _decode_command_frame,  # type: ignore[reportPrivateUsage]
)

# ---------------------------------------------------------------------------
# Wire protocol: command frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "inputs", "num_forwards", "trim_amount"),
    [
        (OP_FORWARD, [42], 4, 0),
        (OP_FORWARD, [10, 20], 5, 0),
        (OP_TRIM_CACHE, [], 0, 7),
        (OP_SHUTDOWN, [], 0, 0),
        (OP_PREFILL, [], 1024, 0),
        (OP_PREFILL, [], 0, 0),
    ],
)
def test_command_frame_round_trip(
    op: int,
    inputs: list[int],
    num_forwards: int,
    trim_amount: int,
) -> None:
    """Every command shape we send must round-trip through encode + decode."""
    frame = _build_command_frame(
        op=op,
        inputs=inputs,
        num_forwards=num_forwards,
        trim_amount=trim_amount,
    )
    assert frame.shape == (COMMAND_FRAME_SIZE,)
    assert frame.dtype == mx.uint32

    decoded_op, decoded_inputs, decoded_num_forwards, decoded_trim = (
        _decode_command_frame(frame)
    )
    assert decoded_op == op
    assert decoded_inputs == inputs
    assert decoded_num_forwards == num_forwards
    assert decoded_trim == trim_amount


def test_command_frame_rejects_long_inputs() -> None:
    with pytest.raises(ValueError, match=r"inputs length must be in \[0, 2\]"):
        _build_command_frame(
            op=OP_FORWARD, inputs=[1, 2, 3], num_forwards=4, trim_amount=0
        )


def test_decode_rejects_wrong_size() -> None:
    bogus = mx.array([0, 0, 0], dtype=mx.uint32)
    with pytest.raises(ValueError, match=r"expected 8"):
        _decode_command_frame(bogus)


# ---------------------------------------------------------------------------
# RemoteTransport with a mocked mx.distributed
# ---------------------------------------------------------------------------


@final
class _MockGroup:
    """Stand-in for :class:`mx.distributed.Group` for unit tests.

    The real group requires a backend (jaccl/ring) and at least two
    processes to instantiate. The wire protocol only needs ``size()``
    and ``rank()`` to be queryable, so we mock those plus an opaque
    sentinel that ``mx.distributed.send/recv`` can dispatch on.
    """

    def __init__(self, size_value: int, rank_value: int) -> None:
        self._size = size_value
        self._rank = rank_value

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


@dataclass
class _IpcRecord:
    """One observed mx.distributed.send/recv call."""

    op: str  # "send" | "recv"
    dst_or_src: int
    shape: tuple[int, ...]
    payload: list[int] = field(default_factory=list)


@dataclass
class _MockedDistributed:
    """Capture every send + serve a queue of recv responses.

    The transport's :meth:`forward` issues ``send(command_frame), recv(drafts_buffer)``;
    we feed the drafts buffer the test wants returned.
    """

    sent: list[_IpcRecord] = field(default_factory=list)
    recv_queue: deque[mx.array] = field(default_factory=deque)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_send(payload: mx.array, dst: int, *, group: object) -> None:
            del group
            flat = [int(x) for x in payload.tolist()]  # type: ignore[reportUnknownArgumentType]
            self.sent.append(
                _IpcRecord(
                    op="send",
                    dst_or_src=dst,
                    shape=tuple(payload.shape),
                    payload=flat,
                )
            )

        def fake_recv(
            shape: tuple[int, ...],
            dtype: object,
            src: int,
            *,
            group: object,
        ) -> mx.array:
            del dtype, src, group
            if not self.recv_queue:
                raise AssertionError(
                    f"recv() called with no queued response (shape={shape})"
                )
            return self.recv_queue.popleft()

        monkeypatch.setattr(mx.distributed, "send", fake_send)
        monkeypatch.setattr(mx.distributed, "recv", fake_recv)


def _enqueue_drafts(
    mocked: _MockedDistributed, drafts: list[int], buffer_size: int
) -> None:
    padded = drafts + [0] * (buffer_size - len(drafts))
    mocked.recv_queue.append(mx.array(padded, dtype=mx.uint32))


def _enqueue_ack(mocked: _MockedDistributed, status: int = ACK_OK) -> None:
    mocked.recv_queue.append(mx.array([status], dtype=mx.uint32))


def test_remote_transport_forward_serialises_command_and_returns_drafts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """forward() must send a command frame and return the recv'd drafts."""
    mocked = _MockedDistributed()
    mocked.install(monkeypatch)

    group = _MockGroup(size_value=2, rank_value=0)
    transport = RemoteTransport(
        num_draft_tokens=4,
        group=group,  # type: ignore[arg-type]
        drafter_rank=1,
        target_rank=0,
    )

    _enqueue_drafts(mocked, [10, 11, 12, 13], buffer_size=5)
    drafts = transport.forward([42], num_forwards=4).result()
    assert drafts == [10, 11, 12, 13]

    # Exactly one send (the command frame).
    sends = [r for r in mocked.sent if r.op == "send"]
    assert len(sends) == 1
    assert sends[0].shape == (COMMAND_FRAME_SIZE,)
    op, inputs, num_forwards, trim = _decode_command_frame(
        mx.array(sends[0].payload, dtype=mx.uint32)
    )
    assert op == OP_FORWARD
    assert inputs == [42]
    assert num_forwards == 4
    assert trim == 0

    _enqueue_ack(mocked)
    transport.shutdown()


def test_remote_transport_trim_cache_emits_ack_then_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocked = _MockedDistributed()
    mocked.install(monkeypatch)

    group = _MockGroup(size_value=2, rank_value=0)
    transport = RemoteTransport(
        num_draft_tokens=4,
        group=group,  # type: ignore[arg-type]
        drafter_rank=1,
        target_rank=0,
    )

    _enqueue_ack(mocked)
    transport.trim_cache(3)

    sends = [r for r in mocked.sent if r.op == "send"]
    assert len(sends) == 1
    op, _, _, trim = _decode_command_frame(mx.array(sends[0].payload, dtype=mx.uint32))
    assert op == OP_TRIM_CACHE
    assert trim == 3

    _enqueue_ack(mocked)
    transport.shutdown()


def test_remote_transport_trim_cache_zero_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocked = _MockedDistributed()
    mocked.install(monkeypatch)

    group = _MockGroup(size_value=2, rank_value=0)
    transport = RemoteTransport(
        num_draft_tokens=4,
        group=group,  # type: ignore[arg-type]
        drafter_rank=1,
        target_rank=0,
    )

    transport.trim_cache(0)
    assert mocked.sent == []

    _enqueue_ack(mocked)
    transport.shutdown()


def test_remote_transport_reset_and_prefill_sends_command_array_and_recv_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reset_and_prefill must send OP_PREFILL frame, prompt array, then await ack."""
    mocked = _MockedDistributed()
    mocked.install(monkeypatch)

    group = _MockGroup(size_value=2, rank_value=0)
    transport = RemoteTransport(
        num_draft_tokens=4,
        group=group,  # type: ignore[arg-type]
        drafter_rank=1,
        target_rank=0,
    )

    prompt = [101, 102, 103, 104, 105]
    _enqueue_ack(mocked)
    transport.reset_and_prefill(prompt)

    sends = [r for r in mocked.sent if r.op == "send"]
    assert len(sends) == 2
    # First send: command frame announcing num_prompt_tokens.
    op, inputs, num_forwards, trim = _decode_command_frame(
        mx.array(sends[0].payload, dtype=mx.uint32)
    )
    assert op == OP_PREFILL
    assert inputs == []
    assert num_forwards == len(prompt)
    assert trim == 0
    # Second send: prompt token array.
    assert sends[1].shape == (len(prompt),)
    assert sends[1].payload == prompt

    _enqueue_ack(mocked)
    transport.shutdown()


def test_remote_transport_reset_and_prefill_empty_prompt_skips_array_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocked = _MockedDistributed()
    mocked.install(monkeypatch)

    group = _MockGroup(size_value=2, rank_value=0)
    transport = RemoteTransport(
        num_draft_tokens=4,
        group=group,  # type: ignore[arg-type]
        drafter_rank=1,
        target_rank=0,
    )

    _enqueue_ack(mocked)
    transport.reset_and_prefill([])

    sends = [r for r in mocked.sent if r.op == "send"]
    # Only the command frame; no prompt array.
    assert len(sends) == 1
    op, _, num_forwards, _ = _decode_command_frame(
        mx.array(sends[0].payload, dtype=mx.uint32)
    )
    assert op == OP_PREFILL
    assert num_forwards == 0

    _enqueue_ack(mocked)
    transport.shutdown()


def test_remote_transport_shutdown_sends_op_and_drains_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocked = _MockedDistributed()
    mocked.install(monkeypatch)

    group = _MockGroup(size_value=2, rank_value=0)
    transport = RemoteTransport(
        num_draft_tokens=4,
        group=group,  # type: ignore[arg-type]
        drafter_rank=1,
        target_rank=0,
    )

    _enqueue_ack(mocked)
    transport.shutdown()

    sends = [r for r in mocked.sent if r.op == "send"]
    assert len(sends) == 1
    op, _, _, _ = _decode_command_frame(mx.array(sends[0].payload, dtype=mx.uint32))
    assert op == OP_SHUTDOWN

    # Idempotent: a second shutdown is a no-op.
    transport.shutdown()
    assert len([r for r in mocked.sent if r.op == "send"]) == 1


def test_remote_transport_rejects_use_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocked = _MockedDistributed()
    mocked.install(monkeypatch)

    group = _MockGroup(size_value=2, rank_value=0)
    transport = RemoteTransport(
        num_draft_tokens=4,
        group=group,  # type: ignore[arg-type]
        drafter_rank=1,
        target_rank=0,
    )

    _enqueue_ack(mocked)
    transport.shutdown()

    with pytest.raises(RuntimeError, match="after shutdown"):
        _ = transport.forward([1], num_forwards=2)
    with pytest.raises(RuntimeError, match="after shutdown"):
        transport.trim_cache(2)
    with pytest.raises(RuntimeError, match="after shutdown"):
        transport.reset_and_prefill([1, 2, 3])


def test_remote_transport_rank_validation() -> None:
    group = _MockGroup(size_value=2, rank_value=0)
    with pytest.raises(ValueError, match="differ"):
        RemoteTransport(
            num_draft_tokens=4,
            group=group,  # type: ignore[arg-type]
            drafter_rank=0,
            target_rank=0,
        )
    with pytest.raises(ValueError, match="out of bounds"):
        RemoteTransport(
            num_draft_tokens=4,
            group=group,  # type: ignore[arg-type]
            drafter_rank=2,
            target_rank=0,
        )


# ---------------------------------------------------------------------------
# drafter_serve_loop dispatch
# ---------------------------------------------------------------------------


def test_drafter_serve_loop_handles_shutdown_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare OP_SHUTDOWN frame must terminate the serve loop with an ACK."""
    from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

    sent: list[mx.array] = []

    def fake_send(payload: mx.array, dst: int, *, group: object) -> None:
        del dst, group
        sent.append(payload)

    shutdown_frame = _build_command_frame(
        op=OP_SHUTDOWN, inputs=[], num_forwards=0, trim_amount=0
    )
    recv_iter: Iterator[mx.array] = iter([shutdown_frame])

    def fake_recv(
        shape: tuple[int, ...],
        dtype: object,
        src: int,
        *,
        group: object,
    ) -> mx.array:
        del shape, dtype, src, group
        return next(recv_iter)

    monkeypatch.setattr(mx.distributed, "send", fake_send)
    monkeypatch.setattr(mx.distributed, "recv", fake_recv)

    group = _MockGroup(size_value=2, rank_value=1)
    drafter_serve_loop(
        draft_model=None,  # pyright: ignore[reportArgumentType]
        draft_cache=[],
        num_draft_tokens=4,
        group=group,  # pyright: ignore[reportArgumentType]
        target_rank=0,
    )

    assert len(sent) == 1
    assert sent[0].shape == (ACK_FRAME_SIZE,)
    assert int(sent[0][0].item()) == ACK_OK


def test_drafter_serve_loop_handles_trim_then_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OP_TRIM_CACHE replies with ACK; the loop continues until OP_SHUTDOWN."""
    from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

    sent: list[mx.array] = []

    def fake_send(payload: mx.array, dst: int, *, group: object) -> None:
        del dst, group
        sent.append(payload)

    trim_frame = _build_command_frame(
        op=OP_TRIM_CACHE, inputs=[], num_forwards=0, trim_amount=2
    )
    shutdown_frame = _build_command_frame(
        op=OP_SHUTDOWN, inputs=[], num_forwards=0, trim_amount=0
    )
    recv_iter: Iterator[mx.array] = iter([trim_frame, shutdown_frame])

    def fake_recv(
        shape: tuple[int, ...],
        dtype: object,
        src: int,
        *,
        group: object,
    ) -> mx.array:
        del shape, dtype, src, group
        return next(recv_iter)

    monkeypatch.setattr(mx.distributed, "send", fake_send)
    monkeypatch.setattr(mx.distributed, "recv", fake_recv)

    # Empty draft_cache: with trim_amount > 0 and an empty cache, the
    # underlying mlx_trim_prompt_cache call would no-op (or raise on
    # some mlx_lm versions). To keep this test independent of mlx_lm
    # internals we patch the trim helper to a no-op.
    def _noop_trim(cache: object, n: int) -> None:
        del cache, n

    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.remote_drafter.mlx_trim_prompt_cache",
        _noop_trim,
    )

    group = _MockGroup(size_value=2, rank_value=1)
    drafter_serve_loop(
        draft_model=None,  # pyright: ignore[reportArgumentType]
        draft_cache=[],
        num_draft_tokens=4,
        group=group,  # pyright: ignore[reportArgumentType]
        target_rank=0,
    )

    # Two acks (trim, shutdown).
    assert len(sent) == 2
    for ack in sent:
        assert ack.shape == (ACK_FRAME_SIZE,)
        assert int(ack[0].item()) == ACK_OK


def test_drafter_serve_loop_rejects_unknown_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown op code must raise so the target's IPC surfaces it."""
    from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

    bogus_frame = mx.array([99, 0, 0, 0, 0, 0, 0, 0], dtype=mx.uint32)
    recv_iter: Iterator[mx.array] = iter([bogus_frame])

    def fake_send(payload: mx.array, dst: int, *, group: object) -> None:
        del payload, dst, group

    def fake_recv(
        shape: tuple[int, ...],
        dtype: object,
        src: int,
        *,
        group: object,
    ) -> mx.array:
        del shape, dtype, src, group
        return next(recv_iter)

    monkeypatch.setattr(mx.distributed, "send", fake_send)
    monkeypatch.setattr(mx.distributed, "recv", fake_recv)

    group = _MockGroup(size_value=2, rank_value=1)
    with pytest.raises(RuntimeError, match="Unknown op code"):
        drafter_serve_loop(
            draft_model=None,  # pyright: ignore[reportArgumentType]
            draft_cache=[],
            num_draft_tokens=4,
            group=group,  # pyright: ignore[reportArgumentType]
            target_rank=0,
        )
