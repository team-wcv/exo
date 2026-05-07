"""Drafter on a different MLX rank, IPC via ``mx.distributed.send/recv``.

:class:`RemoteTransport` (a concrete :class:`DrafterTransport`) and the
matching :func:`drafter_serve_loop` ride on the existing
``mx.distributed.Group`` initialised by exo's runner bootstrap, so the
underlying network transport is automatically:

  * **JACCL (RDMA over IB-verbs / Thunderbolt-bridge)** when the
    instance is :class:`MlxJacclInstance` (``MLX_IBV_DEVICES`` set,
    ``backend="jaccl"``), or
  * **ring (TCP)** when the instance is :class:`MlxRingInstance`
    (``backend="ring"``).

No new transport code in this module -- ``mx.distributed.send`` /
``recv`` carry the small fixed-size protocol messages, and MLX picks
the wire format per group backend. This is what ``rdma is already
built into exo do not reinvent the wheel`` means in practice: the
group is the transport.

Wire protocol (kept deliberately tiny so per-round IPC overhead is
microseconds, not milliseconds):

  * **Command frame** (target -> drafter), shape ``(8,)`` ``uint32``::

        [op, num_inputs, num_forwards, input_0, input_1, trim_amount, _, _]

    Fixed shape so :func:`mx.distributed.recv` can pre-allocate.
    Unused slots are zero-padded.

  * **Drafts frame** (drafter -> target), shape
    ``(K + 1,)`` ``uint32``: the requested forwards' outputs. Padded
    with zeros if the request asked for fewer than ``K + 1`` forwards
    (the caller knows its requested count and slices accordingly).

  * **Ack frame** (drafter -> target), shape ``(1,)`` ``uint32``:
    a single status byte (always ``0`` for "ok"). Sent after
    ``OP_TRIM_CACHE`` so the target rank has a synchronisation point
    against the drafter's cache state, and after ``OP_SHUTDOWN`` so
    the drain returns cleanly.

Op codes: :data:`OP_FORWARD` (1), :data:`OP_TRIM_CACHE` (2),
:data:`OP_SHUTDOWN` (3).

Topology assumption: the calling rank (target) and the drafter rank
are both members of ``group``. The asymmetric instance topology that
designates one rank as drafter-only ships in the same PR
(:class:`MlxJacclInstance` / :class:`MlxRingInstance` extension); the
target's pipeline-parallel collectives operate on a *subgroup*
constructed via ``group.split`` so they don't drag the drafter rank
in. The drafter rank's serve loop uses ``send/recv`` against the
parent ``group`` for cross-subgroup point-to-point, which is exactly
what those primitives are designed for.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Final, final

if TYPE_CHECKING:
    from exo.worker.engines.mlx.generator.drafter_transport import (
        DraftFuture,
        DrafterTransport,
    )
    from exo.worker.engines.mlx.types import KVCacheType, Model

import mlx.core as mx
from mlx_lm.models.cache import trim_prompt_cache as mlx_trim_prompt_cache


# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------

COMMAND_FRAME_SIZE: Final[int] = 8
"""Fixed size of a command frame (uint32 ints)."""

ACK_FRAME_SIZE: Final[int] = 1
"""Fixed size of an ack frame (uint32 ints). The single int is reserved
for a status code; ``0`` means ok. Future revisions may surface error
states here without changing the wire format."""

OP_FORWARD: Final[int] = 1
"""Drafter runs ``num_forwards`` forwards starting from
``inputs[:num_inputs]``. Replies with a Drafts frame."""

OP_TRIM_CACHE: Final[int] = 2
"""Drafter trims ``trim_amount`` positions from its KV cache. Replies
with an Ack frame so the target has a sync point."""

OP_SHUTDOWN: Final[int] = 3
"""Drafter exits its serve loop. Replies with an Ack frame, then the
serve loop returns."""

ACK_OK: Final[int] = 0


def _build_command_frame(
    *,
    op: int,
    inputs: list[int],
    num_forwards: int,
    trim_amount: int,
) -> mx.array:
    """Pack command parameters into a fixed-shape uint32 array.

    Layout: ``[op, num_inputs, num_forwards, input_0, input_1, trim_amount, 0, 0]``.

    ``inputs`` must have length 0, 1, or 2 (the spec loop only ever
    passes length-1 or length-2 inputs to ``forward``; ``OP_TRIM_CACHE``
    and ``OP_SHUTDOWN`` pass length 0). Out-of-band lengths are a
    programming error and raise.
    """
    if len(inputs) > 2:
        raise ValueError(f"inputs length must be in [0, 2], got {len(inputs)}")
    frame = [
        op,
        len(inputs),
        num_forwards,
        inputs[0] if len(inputs) >= 1 else 0,
        inputs[1] if len(inputs) >= 2 else 0,
        trim_amount,
        0,
        0,
    ]
    return mx.array(frame, dtype=mx.uint32)


def _decode_command_frame(frame: mx.array) -> tuple[int, list[int], int, int]:
    """Inverse of :func:`_build_command_frame`.

    Returns ``(op, inputs, num_forwards, trim_amount)``.
    """
    flat = [int(x) for x in frame.tolist()]  # type: ignore[reportUnknownArgumentType]
    if len(flat) != COMMAND_FRAME_SIZE:
        raise ValueError(
            f"Command frame has {len(flat)} ints, expected {COMMAND_FRAME_SIZE}"
        )
    op = flat[0]
    num_inputs = flat[1]
    num_forwards = flat[2]
    trim_amount = flat[5]
    inputs = flat[3 : 3 + num_inputs]
    return op, inputs, num_forwards, trim_amount


# ---------------------------------------------------------------------------
# RemoteTransport (target side)
# ---------------------------------------------------------------------------


@final
class RemoteTransport:
    """Drafter lives on a different MLX rank.

    Each :meth:`forward` send/recv pair is done synchronously on a
    background thread (via :class:`ThreadPoolExecutor`) so the calling
    rank can return a non-blocking :class:`Future` and dispatch the
    target verify forward in parallel.

    Why a thread, given MLX is single-GIL? ``mx.distributed.send/recv``
    block on the network until the peer responds; running them on a
    background thread lets the target's main thread issue MLX
    target-verify dispatches in parallel. The drafter's actual
    compute happens on the *drafter rank's* GPU, not on a thread of
    the calling rank, so there's no GIL contention to worry about.
    """

    def __init__(
        self,
        *,
        num_draft_tokens: int,
        group: mx.distributed.Group,
        drafter_rank: int,
        target_rank: int,
    ) -> None:
        if num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {num_draft_tokens}")
        if drafter_rank == target_rank:
            raise ValueError(
                f"drafter_rank ({drafter_rank}) must differ from target_rank "
                f"({target_rank})"
            )
        if not 0 <= drafter_rank < group.size():
            raise ValueError(
                f"drafter_rank ({drafter_rank}) out of bounds for group size {group.size()}"
            )
        if not 0 <= target_rank < group.size():
            raise ValueError(
                f"target_rank ({target_rank}) out of bounds for group size {group.size()}"
            )
        self._num_draft_tokens = num_draft_tokens
        self._group = group
        self._drafter_rank = drafter_rank
        self._target_rank = target_rank
        # Single-worker pool: requests are strictly serialised because
        # the wire protocol is a per-round state machine (one in-flight
        # forward at a time).
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="exo-drafter-ipc"
        )
        self._is_shutdown = False

    @property
    def num_draft_tokens(self) -> int:
        return self._num_draft_tokens

    def forward(self, inputs: list[int], num_forwards: int) -> "DraftFuture":
        if self._is_shutdown:
            raise RuntimeError(
                "RemoteTransport.forward called after shutdown; the drafter "
                "rank's serve loop has exited and won't respond"
            )
        upper = self._num_draft_tokens + 1
        if not 1 <= num_forwards <= upper:
            raise ValueError(
                f"num_forwards must be in [1, {upper}], got {num_forwards}"
            )
        if not 1 <= len(inputs) <= 2:
            raise ValueError(f"inputs must have length 1 or 2, got {len(inputs)}")
        # Submit to the background thread; the caller gets a Future back
        # that resolves once send/recv completes.
        return self._executor.submit(self._forward_blocking, inputs, num_forwards)

    def trim_cache(self, n_positions: int) -> None:
        if self._is_shutdown:
            raise RuntimeError("RemoteTransport.trim_cache called after shutdown")
        if n_positions < 0:
            raise ValueError(f"n_positions must be >= 0, got {n_positions}")
        if n_positions == 0:
            return
        # Trim is synchronous from the caller's POV (no overlap with
        # target verify: trim happens *after* verify resolves the
        # accept count). We still ack so the target has a sync point.
        self._executor.submit(self._trim_blocking, n_positions).result()

    def shutdown(self) -> None:
        if self._is_shutdown:
            return
        self._is_shutdown = True
        # Send shutdown to the drafter and wait for the ack so the
        # drafter has a chance to drain its own state cleanly.
        try:
            self._executor.submit(self._shutdown_blocking).result(timeout=10.0)
        finally:
            self._executor.shutdown(wait=True)

    # -- internals --------------------------------------------------------

    def _forward_blocking(self, inputs: list[int], num_forwards: int) -> list[int]:
        """Send a forward command and recv the drafts. Runs on the IPC thread."""
        frame = _build_command_frame(
            op=OP_FORWARD,
            inputs=inputs,
            num_forwards=num_forwards,
            trim_amount=0,
        )
        mx.distributed.send(frame, self._drafter_rank, group=self._group)
        # Drafts buffer is fixed-size at K + 1 (the upper bound of any
        # forward request); we slice to ``num_forwards`` here.
        drafts_buffer = mx.distributed.recv(
            shape=(self._num_draft_tokens + 1,),
            dtype=mx.uint32,
            src=self._drafter_rank,
            group=self._group,
        )
        mx.eval(drafts_buffer)
        flat = [int(x) for x in drafts_buffer.tolist()]  # type: ignore[reportUnknownArgumentType]
        return flat[:num_forwards]

    def _trim_blocking(self, n_positions: int) -> None:
        """Send a trim command and wait for the ack."""
        frame = _build_command_frame(
            op=OP_TRIM_CACHE,
            inputs=[],
            num_forwards=0,
            trim_amount=n_positions,
        )
        mx.distributed.send(frame, self._drafter_rank, group=self._group)
        ack = mx.distributed.recv(
            shape=(ACK_FRAME_SIZE,),
            dtype=mx.uint32,
            src=self._drafter_rank,
            group=self._group,
        )
        mx.eval(ack)
        if int(ack[0].item()) != ACK_OK:
            raise RuntimeError(
                f"Drafter rank reported error code {int(ack[0].item())} for trim_cache({n_positions})"
            )

    def _shutdown_blocking(self) -> None:
        """Send shutdown command and wait for the ack."""
        frame = _build_command_frame(
            op=OP_SHUTDOWN,
            inputs=[],
            num_forwards=0,
            trim_amount=0,
        )
        mx.distributed.send(frame, self._drafter_rank, group=self._group)
        ack = mx.distributed.recv(
            shape=(ACK_FRAME_SIZE,),
            dtype=mx.uint32,
            src=self._drafter_rank,
            group=self._group,
        )
        mx.eval(ack)


def make_remote_transport(
    *,
    draft_model: "Model | None" = None,
    draft_cache: "KVCacheType | None" = None,
    num_draft_tokens: int,
    group: mx.distributed.Group | None = None,
    drafter_rank: int | None = None,
    target_rank: int | None = None,
) -> "DrafterTransport":
    """Construct a :class:`RemoteTransport` for the calling target rank.

    On the target rank this returns a transport that sends commands to
    ``drafter_rank``. The drafter rank should run
    :func:`drafter_serve_loop` instead (it loads the drafter model and
    cache, processes commands, sends drafts back).

    Args:
        draft_model: Ignored (the model lives on the drafter rank).
            Included in the signature for parity with the in-process
            factory so callers don't branch on transport kind.
        draft_cache: Ignored (lives on the drafter rank).
        num_draft_tokens: ``K`` -- max drafts per round.
        group: The MLX distributed group both ranks belong to.
        drafter_rank: Rank index of the drafter inside ``group``.
        target_rank: Rank index of the calling target inside ``group``.

    Raises:
        ValueError: required kwargs missing, or rank indices invalid.
    """
    del draft_model, draft_cache  # not relevant on target rank
    if group is None:
        raise ValueError(
            "make_remote_transport requires `group`; the asymmetric "
            "instance topology ensures the runner bootstrap initialises "
            "an mx.distributed group before calling this factory"
        )
    if drafter_rank is None or target_rank is None:
        raise ValueError(
            "make_remote_transport requires `drafter_rank` and `target_rank`; "
            "these are produced by the asymmetric instance topology and "
            "passed through `make_drafter`"
        )
    return RemoteTransport(
        num_draft_tokens=num_draft_tokens,
        group=group,
        drafter_rank=drafter_rank,
        target_rank=target_rank,
    )


# ---------------------------------------------------------------------------
# drafter_serve_loop (drafter side)
# ---------------------------------------------------------------------------


def drafter_serve_loop(
    *,
    draft_model: "Model",
    draft_cache: "KVCacheType",
    num_draft_tokens: int,
    group: mx.distributed.Group,
    target_rank: int,
) -> None:
    """Run the drafter rank's command-loop until ``OP_SHUTDOWN``.

    Receives :data:`COMMAND_FRAME_SIZE`-element command frames from
    ``target_rank``, dispatches on the op code, executes the
    drafter-side work, and replies with the appropriate frame.

    See module docstring for the wire protocol.
    """
    drafts_buffer_size = num_draft_tokens + 1

    while True:
        frame = mx.distributed.recv(
            shape=(COMMAND_FRAME_SIZE,),
            dtype=mx.uint32,
            src=target_rank,
            group=group,
        )
        mx.eval(frame)
        op, inputs, num_forwards, trim_amount = _decode_command_frame(frame)

        if op == OP_SHUTDOWN:
            ack = mx.array([ACK_OK], dtype=mx.uint32)
            mx.distributed.send(ack, target_rank, group=group)
            return

        if op == OP_TRIM_CACHE:
            if trim_amount > 0:
                from typing import cast as _cast

                mlx_trim_prompt_cache(_cast(list[object], draft_cache), trim_amount)  # type: ignore[reportArgumentType]
            ack = mx.array([ACK_OK], dtype=mx.uint32)
            mx.distributed.send(ack, target_rank, group=group)
            continue

        if op == OP_FORWARD:
            outputs = _run_drafter_forwards_remote(
                draft_model=draft_model,
                draft_cache=draft_cache,
                inputs=inputs,
                num_forwards=num_forwards,
            )
            # Pad to fixed-shape buffer so the target's recv pre-allocation matches.
            padded = list(outputs) + [0] * (drafts_buffer_size - len(outputs))
            response = mx.array(padded, dtype=mx.uint32)
            mx.distributed.send(response, target_rank, group=group)
            continue

        # Unknown op code: this is a wire-protocol violation, not a
        # recoverable error. Send an ack with a non-zero status so the
        # caller's trim_cache/shutdown blocks observe it; for
        # OP_FORWARD the caller is waiting for a drafts frame so we
        # can't use the ack channel -- raise instead so the serve
        # loop dies and the caller's RemoteTransport surfaces the
        # network error.
        raise RuntimeError(f"Unknown op code from target rank: {op}")


def _run_drafter_forwards_remote(
    *,
    draft_model: "Model",
    draft_cache: "KVCacheType",
    inputs: list[int],
    num_forwards: int,
) -> list[int]:
    """Same forward semantics as ``InProcessTransport._run_drafter_forwards``.

    Kept as a free function to avoid importing the in-process transport
    on the drafter rank (which only loads the drafter model, not any
    target-side code).
    """
    if num_forwards < 1:
        raise ValueError(f"num_forwards must be >= 1, got {num_forwards}")
    if not 1 <= len(inputs) <= 2:
        raise ValueError(f"inputs must have length 1 or 2, got {len(inputs)}")
    ys: list[mx.array] = []
    y = mx.array(inputs, dtype=mx.uint32)
    for _ in range(num_forwards):
        logits = draft_model(y[None], cache=draft_cache)
        sampled = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
        mx.async_eval(sampled)
        ys.append(sampled)
        y = sampled
    mx.eval(ys + [c.state for c in draft_cache])  # type: ignore[reportArgumentType]
    return [int(t.item()) for t in ys]


__all__ = [
    "ACK_FRAME_SIZE",
    "ACK_OK",
    "COMMAND_FRAME_SIZE",
    "OP_FORWARD",
    "OP_SHUTDOWN",
    "OP_TRIM_CACHE",
    "RemoteTransport",
    "drafter_serve_loop",
    "make_remote_transport",
]


# Suppress the unused-import warnings for the future-only Future type:
# ThreadPoolExecutor.submit returns ``Future`` which is structurally
# compatible with :data:`DraftFuture`, but we annotate the return type
# inside the class body and the import is otherwise unused.
_ = Future
