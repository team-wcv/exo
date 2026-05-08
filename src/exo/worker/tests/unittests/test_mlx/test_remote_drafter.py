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
    OP_END_SESSION,
    OP_FORWARD,
    OP_PREFILL,
    OP_SHUTDOWN,
    OP_TRIM_CACHE,
    SESSION_ID_NONE,
    RemoteTransport,
    _build_command_frame,  # type: ignore[reportPrivateUsage]
    _decode_command_frame,  # type: ignore[reportPrivateUsage]
)

# ---------------------------------------------------------------------------
# Wire protocol: command frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "inputs", "num_forwards", "trim_amount", "session_id"),
    [
        (OP_FORWARD, [42], 4, 0, 0),
        (OP_FORWARD, [10, 20], 5, 0, 7),
        (OP_TRIM_CACHE, [], 0, 7, 3),
        (OP_SHUTDOWN, [], 0, 0, SESSION_ID_NONE),
        (OP_PREFILL, [], 1024, 0, 1),
        (OP_PREFILL, [], 0, 0, 0),
        (OP_END_SESSION, [], 0, 0, 42),
        # Boundary: max uint32 - 1 (SESSION_ID_NONE is the sentinel).
        (OP_FORWARD, [1], 2, 0, 0xFFFFFFFE),
    ],
)
def test_command_frame_round_trip(
    op: int,
    inputs: list[int],
    num_forwards: int,
    trim_amount: int,
    session_id: int,
) -> None:
    """Every command shape we send must round-trip through encode + decode."""
    frame = _build_command_frame(
        op=op,
        inputs=inputs,
        num_forwards=num_forwards,
        trim_amount=trim_amount,
        session_id=session_id,
    )
    assert frame.shape == (COMMAND_FRAME_SIZE,)
    assert frame.dtype == mx.uint32

    decoded_op, decoded_inputs, decoded_num_forwards, decoded_trim, decoded_sid = (
        _decode_command_frame(frame)
    )
    assert decoded_op == op
    assert decoded_inputs == inputs
    assert decoded_num_forwards == num_forwards
    assert decoded_trim == trim_amount
    assert decoded_sid == session_id


def test_command_frame_rejects_long_inputs() -> None:
    with pytest.raises(ValueError, match=r"inputs length must be in \[0, 2\]"):
        _build_command_frame(
            op=OP_FORWARD,
            inputs=[1, 2, 3],
            num_forwards=4,
            trim_amount=0,
            session_id=0,
        )


def test_command_frame_rejects_session_id_out_of_uint32_range() -> None:
    with pytest.raises(ValueError, match=r"session_id must fit in uint32"):
        _build_command_frame(
            op=OP_FORWARD,
            inputs=[1],
            num_forwards=2,
            trim_amount=0,
            session_id=2**33,
        )


def test_decode_rejects_wrong_size() -> None:
    bogus = mx.array([0, 0, 0], dtype=mx.uint32)
    with pytest.raises(ValueError, match=r"expected 9"):
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


def _make_transport() -> tuple[RemoteTransport, _MockedDistributed]:
    """Helper: construct a 2-rank :class:`RemoteTransport` over mocked IPC."""
    mocked = _MockedDistributed()
    group = _MockGroup(size_value=2, rank_value=0)
    transport = RemoteTransport(
        num_draft_tokens=4,
        group=group,  # type: ignore[arg-type]
        drafter_rank=1,
        target_rank=0,
    )
    return transport, mocked


def test_open_session_allocates_unique_session_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each ``open_session`` call must yield a fresh session id."""
    transport, mocked = _make_transport()
    mocked.install(monkeypatch)

    session_a = transport.open_session()
    session_b = transport.open_session()
    session_c = transport.open_session()
    assert session_a.session_id != session_b.session_id
    assert session_b.session_id != session_c.session_id
    # ``num_draft_tokens`` is shared across sessions (transport-level
    # wire-protocol budget).
    assert session_a.num_draft_tokens == transport.num_draft_tokens

    _enqueue_ack(mocked)
    transport.shutdown()


def test_session_handle_forward_serialises_command_with_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``session.forward`` must encode the session id into the command frame."""
    transport, mocked = _make_transport()
    mocked.install(monkeypatch)

    session = transport.open_session()
    _enqueue_drafts(mocked, [10, 11, 12, 13], buffer_size=5)
    drafts = session.forward([42], num_forwards=4).result()
    assert drafts == [10, 11, 12, 13]

    sends = [r for r in mocked.sent if r.op == "send"]
    assert len(sends) == 1
    op, inputs, num_forwards, trim, session_id = _decode_command_frame(
        mx.array(sends[0].payload, dtype=mx.uint32)
    )
    assert op == OP_FORWARD
    assert inputs == [42]
    assert num_forwards == 4
    assert trim == 0
    assert session_id == session.session_id

    _enqueue_ack(mocked)  # for end_session
    session.shutdown()
    _enqueue_ack(mocked)  # for transport shutdown
    transport.shutdown()


def test_session_handle_trim_cache_emits_session_scoped_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, mocked = _make_transport()
    mocked.install(monkeypatch)

    session = transport.open_session()
    _enqueue_ack(mocked)
    session.trim_cache(3)

    sends = [r for r in mocked.sent if r.op == "send"]
    assert len(sends) == 1
    op, _, _, trim, session_id = _decode_command_frame(
        mx.array(sends[0].payload, dtype=mx.uint32)
    )
    assert op == OP_TRIM_CACHE
    assert trim == 3
    assert session_id == session.session_id

    _enqueue_ack(mocked)  # end_session
    session.shutdown()
    _enqueue_ack(mocked)  # transport shutdown
    transport.shutdown()


def test_session_handle_trim_cache_zero_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, mocked = _make_transport()
    mocked.install(monkeypatch)

    session = transport.open_session()
    session.trim_cache(0)
    assert mocked.sent == []

    _enqueue_ack(mocked)
    session.shutdown()
    _enqueue_ack(mocked)
    transport.shutdown()


def test_session_handle_reset_and_prefill_sends_command_array_and_recv_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reset_and_prefill must send OP_PREFILL frame, prompt array, then await ack."""
    transport, mocked = _make_transport()
    mocked.install(monkeypatch)

    session = transport.open_session()
    prompt = [101, 102, 103, 104, 105]
    _enqueue_ack(mocked)
    session.reset_and_prefill(prompt)

    sends = [r for r in mocked.sent if r.op == "send"]
    assert len(sends) == 2
    op, inputs, num_forwards, trim, session_id = _decode_command_frame(
        mx.array(sends[0].payload, dtype=mx.uint32)
    )
    assert op == OP_PREFILL
    assert inputs == []
    assert num_forwards == len(prompt)
    assert trim == 0
    assert session_id == session.session_id
    # Second send: prompt token array.
    assert sends[1].shape == (len(prompt),)
    assert sends[1].payload == prompt

    _enqueue_ack(mocked)
    session.shutdown()
    _enqueue_ack(mocked)
    transport.shutdown()


def test_session_handle_reset_and_prefill_empty_prompt_skips_array_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, mocked = _make_transport()
    mocked.install(monkeypatch)

    session = transport.open_session()
    _enqueue_ack(mocked)
    session.reset_and_prefill([])

    sends = [r for r in mocked.sent if r.op == "send"]
    assert len(sends) == 1
    op, _, num_forwards, _, _ = _decode_command_frame(
        mx.array(sends[0].payload, dtype=mx.uint32)
    )
    assert op == OP_PREFILL
    assert num_forwards == 0

    _enqueue_ack(mocked)
    session.shutdown()
    _enqueue_ack(mocked)
    transport.shutdown()


def test_session_handle_shutdown_sends_op_end_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``session.shutdown`` must send OP_END_SESSION (not OP_SHUTDOWN)."""
    transport, mocked = _make_transport()
    mocked.install(monkeypatch)

    session = transport.open_session()
    _enqueue_ack(mocked)
    session.shutdown()

    sends = [r for r in mocked.sent if r.op == "send"]
    assert len(sends) == 1
    op, _, _, _, session_id = _decode_command_frame(
        mx.array(sends[0].payload, dtype=mx.uint32)
    )
    assert op == OP_END_SESSION
    assert session_id == session.session_id

    # Idempotent: a second shutdown is a no-op.
    session.shutdown()
    assert len([r for r in mocked.sent if r.op == "send"]) == 1

    _enqueue_ack(mocked)
    transport.shutdown()


def test_session_handle_rejects_use_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, mocked = _make_transport()
    mocked.install(monkeypatch)

    session = transport.open_session()
    _enqueue_ack(mocked)
    session.shutdown()

    with pytest.raises(RuntimeError, match="after shutdown"):
        _ = session.forward([1], num_forwards=2)
    with pytest.raises(RuntimeError, match="after shutdown"):
        session.trim_cache(2)
    with pytest.raises(RuntimeError, match="after shutdown"):
        session.reset_and_prefill([1, 2, 3])

    _enqueue_ack(mocked)
    transport.shutdown()


def test_remote_transport_shutdown_sends_op_and_drains_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, mocked = _make_transport()
    mocked.install(monkeypatch)

    _enqueue_ack(mocked)
    transport.shutdown()

    sends = [r for r in mocked.sent if r.op == "send"]
    assert len(sends) == 1
    op, _, _, _, _ = _decode_command_frame(mx.array(sends[0].payload, dtype=mx.uint32))
    assert op == OP_SHUTDOWN

    # Idempotent: a second shutdown is a no-op.
    transport.shutdown()
    assert len([r for r in mocked.sent if r.op == "send"]) == 1


def test_remote_transport_rejects_use_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, mocked = _make_transport()
    mocked.install(monkeypatch)

    _enqueue_ack(mocked)
    transport.shutdown()

    with pytest.raises(RuntimeError, match="after shutdown"):
        _ = transport.open_session()


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


def _empty_cache_factory() -> object:
    """Drop-in factory for tests that don't actually run forwards."""
    return []


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
        op=OP_SHUTDOWN,
        inputs=[],
        num_forwards=0,
        trim_amount=0,
        session_id=SESSION_ID_NONE,
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
        make_draft_cache=_empty_cache_factory,  # pyright: ignore[reportArgumentType]
        num_draft_tokens=4,
        group=group,  # pyright: ignore[reportArgumentType]
        target_rank=0,
    )

    assert len(sent) == 1
    assert sent[0].shape == (ACK_FRAME_SIZE,)
    assert int(sent[0][0].item()) == ACK_OK


def test_drafter_serve_loop_handles_prefill_then_trim_then_end_then_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full session lifecycle: PREFILL allocates, TRIM operates, END_SESSION frees."""
    from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

    sent: list[mx.array] = []

    def fake_send(payload: mx.array, dst: int, *, group: object) -> None:
        del dst, group
        sent.append(payload)

    session_id = 7
    prefill_frame = _build_command_frame(
        op=OP_PREFILL,
        inputs=[],
        num_forwards=0,  # empty prompt -- skip prompt-array recv
        trim_amount=0,
        session_id=session_id,
    )
    trim_frame = _build_command_frame(
        op=OP_TRIM_CACHE,
        inputs=[],
        num_forwards=0,
        trim_amount=2,
        session_id=session_id,
    )
    end_frame = _build_command_frame(
        op=OP_END_SESSION,
        inputs=[],
        num_forwards=0,
        trim_amount=0,
        session_id=session_id,
    )
    shutdown_frame = _build_command_frame(
        op=OP_SHUTDOWN,
        inputs=[],
        num_forwards=0,
        trim_amount=0,
        session_id=SESSION_ID_NONE,
    )
    recv_iter: Iterator[mx.array] = iter(
        [prefill_frame, trim_frame, end_frame, shutdown_frame]
    )

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

    # Cache-trim helper is exercised under TRIM; mock it to avoid
    # mlx_lm version-dependent behaviour on empty caches.
    def _noop_trim(cache: object, n: int) -> None:
        del cache, n

    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.remote_drafter.mlx_trim_prompt_cache",
        _noop_trim,
    )

    cache_calls: list[None] = []

    def _cache_factory() -> object:
        cache_calls.append(None)
        return []

    group = _MockGroup(size_value=2, rank_value=1)
    drafter_serve_loop(
        draft_model=None,  # pyright: ignore[reportArgumentType]
        make_draft_cache=_cache_factory,  # pyright: ignore[reportArgumentType]
        num_draft_tokens=4,
        group=group,  # pyright: ignore[reportArgumentType]
        target_rank=0,
    )

    # Four acks: prefill, trim, end_session, shutdown.
    assert len(sent) == 4
    for ack in sent:
        assert ack.shape == (ACK_FRAME_SIZE,)
        assert int(ack[0].item()) == ACK_OK
    # Cache factory was called exactly once (on PREFILL).
    assert len(cache_calls) == 1


def test_drafter_serve_loop_rejects_forward_for_unknown_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OP_FORWARD against an unallocated session_id is a wire-protocol violation."""
    from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

    forward_frame = _build_command_frame(
        op=OP_FORWARD,
        inputs=[1],
        num_forwards=1,
        trim_amount=0,
        session_id=99,  # never PREFILL'd
    )
    recv_iter: Iterator[mx.array] = iter([forward_frame])

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
    with pytest.raises(RuntimeError, match="unknown session"):
        drafter_serve_loop(
            draft_model=None,  # pyright: ignore[reportArgumentType]
            make_draft_cache=_empty_cache_factory,  # pyright: ignore[reportArgumentType]
            num_draft_tokens=4,
            group=group,  # pyright: ignore[reportArgumentType]
            target_rank=0,
        )


def test_drafter_serve_loop_end_session_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OP_END_SESSION for a never-prefilled session_id must ack OK (not raise)."""
    from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

    sent: list[mx.array] = []

    def fake_send(payload: mx.array, dst: int, *, group: object) -> None:
        del dst, group
        sent.append(payload)

    end_frame = _build_command_frame(
        op=OP_END_SESSION,
        inputs=[],
        num_forwards=0,
        trim_amount=0,
        session_id=12345,  # never PREFILL'd
    )
    shutdown_frame = _build_command_frame(
        op=OP_SHUTDOWN,
        inputs=[],
        num_forwards=0,
        trim_amount=0,
        session_id=SESSION_ID_NONE,
    )
    recv_iter: Iterator[mx.array] = iter([end_frame, shutdown_frame])

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
        make_draft_cache=_empty_cache_factory,  # pyright: ignore[reportArgumentType]
        num_draft_tokens=4,
        group=group,  # pyright: ignore[reportArgumentType]
        target_rank=0,
    )

    # Two acks (end_session, shutdown), both OK.
    assert len(sent) == 2
    for ack in sent:
        assert int(ack[0].item()) == ACK_OK


def test_drafter_serve_loop_rejects_unknown_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown op code must raise so the target's IPC surfaces it."""
    from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

    bogus_frame = mx.array([99, 0, 0, 0, 0, 0, 0, 0, 0], dtype=mx.uint32)
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
            make_draft_cache=_empty_cache_factory,  # pyright: ignore[reportArgumentType]
            num_draft_tokens=4,
            group=group,  # pyright: ignore[reportArgumentType]
            target_rank=0,
        )


# ---------------------------------------------------------------------------
# Concurrent sessions on a single transport
# ---------------------------------------------------------------------------


def test_two_sessions_carry_distinct_session_ids_in_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent target requests must tag their wire ops with distinct session_ids.

    Verifies the target side: each ``_SessionHandle`` stamps its own
    session_id into every command it issues, so the drafter rank can
    multiplex its per-session caches without ambiguity. The wire stays
    serial via the transport's single ``ThreadPoolExecutor``; this
    test asserts the ordering and routing on the target side, not the
    drafter side (drafter-side multiplexing is covered above).
    """
    transport, mocked = _make_transport()
    mocked.install(monkeypatch)

    session_a = transport.open_session()
    session_b = transport.open_session()

    # Interleave forwards: A.forward, B.forward, A.forward.
    _enqueue_drafts(mocked, [10, 11, 12, 13], buffer_size=5)
    _enqueue_drafts(mocked, [20, 21, 22, 23], buffer_size=5)
    _enqueue_drafts(mocked, [30, 31, 32, 33], buffer_size=5)

    drafts_a1 = session_a.forward([1], num_forwards=4).result()
    drafts_b = session_b.forward([2], num_forwards=4).result()
    drafts_a2 = session_a.forward([3], num_forwards=4).result()

    assert drafts_a1 == [10, 11, 12, 13]
    assert drafts_b == [20, 21, 22, 23]
    assert drafts_a2 == [30, 31, 32, 33]

    # Each command frame carries the session id of the issuing handle.
    sends = [r for r in mocked.sent if r.op == "send"]
    assert len(sends) == 3
    sids = [
        _decode_command_frame(mx.array(s.payload, dtype=mx.uint32))[4] for s in sends
    ]
    assert sids[0] == session_a.session_id
    assert sids[1] == session_b.session_id
    assert sids[2] == session_a.session_id

    _enqueue_ack(mocked)  # end session_a
    session_a.shutdown()
    _enqueue_ack(mocked)  # end session_b
    session_b.shutdown()
    _enqueue_ack(mocked)  # transport shutdown
    transport.shutdown()
