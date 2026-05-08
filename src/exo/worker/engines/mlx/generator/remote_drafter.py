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

Wire protocol v2 (session-aware -- multiple in-flight target requests
share the wire by tagging every command with a ``session_id`` so the
drafter rank can keep separate KV caches per session):

  * **Command frame** (target -> drafter), shape ``(9,)`` ``uint32``::

        [op, num_inputs, num_forwards, input_0, input_1, trim_amount, session_id, _, _]

    Fixed shape so :func:`mx.distributed.recv` can pre-allocate.
    ``session_id`` selects which per-session draft cache the drafter
    rank routes the op to. ``OP_SHUTDOWN`` ignores ``session_id``
    (it tears down the entire serve loop). All other ops require a
    valid ``session_id`` -- :data:`OP_PREFILL` allocates the session,
    :data:`OP_END_SESSION` frees it, the rest reference an existing
    session. Unused slots are zero-padded.

  * **Drafts frame** (drafter -> target), shape
    ``(K + 1,)`` ``uint32``: the requested forwards' outputs. Padded
    with zeros if the request asked for fewer than ``K + 1`` forwards
    (the caller knows its requested count and slices accordingly).

  * **Ack frame** (drafter -> target), shape ``(1,)`` ``uint32``:
    a single status byte (always ``0`` for "ok"). Sent after
    ``OP_TRIM_CACHE``, ``OP_PREFILL``, ``OP_END_SESSION``, and
    ``OP_SHUTDOWN`` so the target rank has a synchronisation point
    against the drafter's cache state.

Op codes: :data:`OP_FORWARD` (1), :data:`OP_TRIM_CACHE` (2),
:data:`OP_SHUTDOWN` (3), :data:`OP_PREFILL` (4),
:data:`OP_END_SESSION` (5).

Concurrency model: ``RemoteTransport`` exposes :meth:`open_session`
which allocates a fresh ``session_id`` and returns a session-scoped
:class:`DrafterTransport` view. Each in-flight target request gets
its own session handle; the underlying wire protocol stays serial
(single ``ThreadPoolExecutor``) because ``mx.distributed.send/recv``
on a given group is not safe to interleave from multiple threads,
but the drafter rank multiplexes operations across sessions by
keying each op's KV-cache lookup on ``session_id``. The cap on
concurrent target requests is therefore set by the *target* runner
(``EXO_MAX_CONCURRENT_REQUESTS``), not by the drafter wire.

Topology assumption: the calling rank (target) and the drafter rank
are both members of ``group``. The asymmetric instance topology that
designates one rank as drafter-only ships in the same PR
(:class:`MlxJacclInstance` / :class:`MlxRingInstance` extension); the
target's pipeline-parallel collectives operate on a *subgroup*
constructed via ``group.split`` so they don't drag the drafter rank
in. The drafter rank's serve loop uses ``send/recv`` against the
parent ``group`` for cross-subgroup point-to-point, which is exactly
what those primitives are designed for.

Twin-machine testing recipe (V1 N=1 asymmetric):
================================================

Two Macs reachable over Tailscale (e.g. ``wc-smbp`` + ``wc-smbpt``):

    # On both machines, ensure the model card lists drafter_eligible_nodes:
    # ModelCard(..., drafter_eligible_nodes=[wc_smbpt_node_id], drafters=[<drafter_id>])

    # Worker on wc-smbp (target rank): defaults
    EXO_DRAFT_MODE=pipelined uv run exo

    # Worker on wc-smbpt (drafter rank): same env, the asymmetric
    # placement layer assigns the drafter role automatically based
    # on drafter_eligible_nodes. Pre-download the drafter weights
    # there so DrafterRunner.LoadModel doesn't fault.
    EXO_DRAFT_MODE=pipelined uv run exo

For TCP/IP testing, use ``MlxRing`` (single-shard target) + the drafter
node listed in eligibility. For RDMA testing, use ``MlxJaccl``
(typically multi-node target) + Thunderbolt-bridge between the twins;
the same code path runs over both backends -- the only difference is
which ``mx.distributed`` backend negotiates the wire format.

Single target rank (N=1) is the only supported V1 topology because
the spec loop currently runs in lockstep on a single rank. Multi-target
asymmetric (N>1, e.g. four target shards + one drafter on a fifth
node) is gated behind a ``NotImplementedError`` in
:func:`exo.worker.engines.mlx.generator.drafter.make_drafter` until the
pipelined spec loop gains target-subgroup draft broadcast support.
Placement still allows N>1 to keep telemetry honest; the runner side
fails loudly so misconfiguration doesn't silently fall back to an
arbitrary mode.
"""

from __future__ import annotations

import itertools
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable, Final, final

if TYPE_CHECKING:
    from exo.worker.engines.mlx.generator.drafter_transport import DraftFuture
    from exo.worker.engines.mlx.types import KVCacheType, Model

import mlx.core as mx
from mlx_lm.models.cache import trim_prompt_cache as mlx_trim_prompt_cache

# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------

COMMAND_FRAME_SIZE: Final[int] = 9
"""Fixed size of a command frame (uint32 ints).

Bumped from 8 to 9 in the session-aware revision to carry a
``session_id`` slot so the drafter rank can multiplex per-session KV
caches without interleaving wire ops on the same socket.
"""

ACK_FRAME_SIZE: Final[int] = 1
"""Fixed size of an ack frame (uint32 ints). The single int is reserved
for a status code; ``0`` means ok. Future revisions may surface error
states here without changing the wire format."""

OP_FORWARD: Final[int] = 1
"""Drafter runs ``num_forwards`` forwards starting from
``inputs[:num_inputs]`` against ``sessions[session_id]``'s KV cache.
Replies with a Drafts frame."""

OP_TRIM_CACHE: Final[int] = 2
"""Drafter trims ``trim_amount`` positions from
``sessions[session_id]``'s KV cache. Replies with an Ack frame so the
target has a sync point."""

OP_SHUTDOWN: Final[int] = 3
"""Drafter exits its serve loop. Replies with an Ack frame, then the
serve loop returns. ``session_id`` is ignored -- this op tears down
the entire wire, not a single session. Per-session cleanup uses
:data:`OP_END_SESSION` instead."""

OP_PREFILL: Final[int] = 4
"""Per-request setup: target announces a prompt of ``num_inputs`` (used
as ``num_prompt_tokens``) tokens for ``session_id``. The drafter
allocates a fresh KV cache for the session (or resets the existing
one to offset 0), recvs the prompt token array (shape
``(num_prompt_tokens,)``), runs prefill forwards through the drafter
model, then replies with an Ack frame. Issued once at the start of
every request so the spec loop's first ``OP_FORWARD`` seeds against
an aligned drafter cache."""

OP_END_SESSION: Final[int] = 5
"""Per-request teardown: drafter drops ``sessions[session_id]`` to free
the KV cache memory and replies with an Ack frame so the target has a
sync point. Idempotent: ending a non-existent session is also a
successful ack (sessions can drop themselves on the drafter side via
target shutdown without the target getting a chance to send this op).
"""

ACK_OK: Final[int] = 0

# ``session_id`` slot in the command frame. Sentinel for shutdown
# (which has no session). 0 is also the first session_id allocated by
# the target -- shutdown ops use the sentinel to make the wire trace
# easier to read; the drafter's serve loop ignores ``session_id`` for
# ``OP_SHUTDOWN`` regardless.
SESSION_ID_NONE: Final[int] = 0xFFFFFFFF


def _build_command_frame(
    *,
    op: int,
    inputs: list[int],
    num_forwards: int,
    trim_amount: int,
    session_id: int,
) -> mx.array:
    """Pack command parameters into a fixed-shape uint32 array.

    Layout: ``[op, num_inputs, num_forwards, input_0, input_1, trim_amount, session_id, 0, 0]``.

    ``inputs`` must have length 0, 1, or 2 (the spec loop only ever
    passes length-1 or length-2 inputs to ``forward``; ``OP_TRIM_CACHE``,
    ``OP_END_SESSION``, and ``OP_SHUTDOWN`` pass length 0). Out-of-band
    lengths are a programming error and raise.

    ``session_id`` MUST fit in uint32. The target allocates session ids
    monotonically per :class:`RemoteTransport` instance from a counter,
    which gives ~4G sessions per runner lifetime -- plenty for any
    realistic deployment. Wraparound is not handled (the runner would
    have to serve > 4 billion concurrent requests; if that ever
    happens, switch the counter to a free-list of recycled ids).
    """
    if len(inputs) > 2:
        raise ValueError(f"inputs length must be in [0, 2], got {len(inputs)}")
    if not 0 <= session_id <= 0xFFFFFFFF:
        raise ValueError(f"session_id must fit in uint32, got {session_id}")
    frame = [
        op,
        len(inputs),
        num_forwards,
        inputs[0] if len(inputs) >= 1 else 0,
        inputs[1] if len(inputs) >= 2 else 0,
        trim_amount,
        session_id,
        0,
        0,
    ]
    return mx.array(frame, dtype=mx.uint32)


def _decode_command_frame(frame: mx.array) -> tuple[int, list[int], int, int, int]:
    """Inverse of :func:`_build_command_frame`.

    Returns ``(op, inputs, num_forwards, trim_amount, session_id)``.
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
    session_id = flat[6]
    inputs = flat[3 : 3 + num_inputs]
    return op, inputs, num_forwards, trim_amount, session_id


# ---------------------------------------------------------------------------
# RemoteTransport (target side)
# ---------------------------------------------------------------------------


@final
class RemoteTransport:
    """Wire-protocol owner for the asymmetric drafter rank.

    Holds the long-lived ``mx.distributed`` group + IPC thread; vends
    per-request :class:`_SessionHandle` instances via :meth:`open_session`.
    Each handle implements :class:`DrafterTransport` so the spec loop
    code is unchanged -- it just receives a session-scoped transport
    rather than the shared one.

    Each wire op (forward / trim / prefill / end-session) is dispatched
    on a single-worker :class:`ThreadPoolExecutor`. Wire ops therefore
    serialise even when multiple in-flight target requests are calling
    methods concurrently from different :class:`_SessionHandle`
    instances, which is exactly what we need: ``mx.distributed.send/recv``
    on a single group is not safe to interleave from multiple threads,
    but the drafter rank multiplexes operations across sessions by
    keying its KV-cache lookup on ``session_id``.

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
        # Single-worker pool: every wire op (across all sessions) goes
        # through it serially, which keeps ``mx.distributed.send/recv``
        # safe even when multiple :class:`_SessionHandle` instances are
        # in flight on different target tasks.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="exo-drafter-ipc"
        )
        self._is_shutdown = False
        # Monotonic session id allocator. ``itertools.count`` gives us a
        # thread-safe unsigned counter; we wrap it in a lock-free
        # ``next()`` call inside :meth:`open_session` (Python's GIL
        # makes the increment atomic for CPython, but the lock makes
        # the contract explicit and survives a free-threaded build).
        self._session_id_counter = itertools.count()
        self._session_lock = threading.Lock()

    @property
    def num_draft_tokens(self) -> int:
        return self._num_draft_tokens

    def open_session(self) -> "_SessionHandle":
        """Allocate a fresh session and return a :class:`DrafterTransport` view.

        Each call yields a unique ``session_id``; the handle's
        :meth:`_SessionHandle.shutdown` sends ``OP_END_SESSION`` so the
        drafter rank can free the per-session KV cache. Forgetting to
        call :meth:`_SessionHandle.shutdown` leaks a KV cache on the
        drafter rank for that session id; ``RemoteTransport.shutdown``
        cleans up at process exit either way.
        """
        if self._is_shutdown:
            raise RuntimeError(
                "RemoteTransport.open_session called after shutdown; the "
                "drafter rank's serve loop has exited and won't respond"
            )
        with self._session_lock:
            session_id = next(self._session_id_counter)
        if session_id == SESSION_ID_NONE:
            # 4G sessions exhausted; bump again so we never collide
            # with the shutdown sentinel. In practice unreachable.
            with self._session_lock:
                session_id = next(self._session_id_counter)
        return _SessionHandle(owner=self, session_id=session_id)

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

    # -- session-scoped wire ops (called by _SessionHandle) -------------

    def _submit_forward(
        self, session_id: int, inputs: list[int], num_forwards: int
    ) -> "DraftFuture":
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
        return self._executor.submit(
            self._forward_blocking, session_id, inputs, num_forwards
        )

    def _submit_trim(self, session_id: int, n_positions: int) -> None:
        if self._is_shutdown:
            raise RuntimeError("RemoteTransport.trim_cache called after shutdown")
        if n_positions < 0:
            raise ValueError(f"n_positions must be >= 0, got {n_positions}")
        if n_positions == 0:
            return
        self._executor.submit(self._trim_blocking, session_id, n_positions).result()

    def _submit_prefill(self, session_id: int, prompt_tokens: list[int]) -> None:
        if self._is_shutdown:
            raise RuntimeError(
                "RemoteTransport.reset_and_prefill called after shutdown"
            )
        self._executor.submit(
            self._reset_and_prefill_blocking, session_id, prompt_tokens
        ).result()

    def _submit_end_session(self, session_id: int) -> None:
        # Best-effort: if the wire is already shut down (process is
        # tearing down), the session-side OP_END_SESSION would fail
        # but the drafter rank is also exiting, so the cache is freed
        # by process death anyway.
        if self._is_shutdown:
            return
        self._executor.submit(self._end_session_blocking, session_id).result()

    # -- internals --------------------------------------------------------

    def _forward_blocking(
        self, session_id: int, inputs: list[int], num_forwards: int
    ) -> list[int]:
        """Send a forward command and recv the drafts. Runs on the IPC thread."""
        frame = _build_command_frame(
            op=OP_FORWARD,
            inputs=inputs,
            num_forwards=num_forwards,
            trim_amount=0,
            session_id=session_id,
        )
        sent_frame = mx.distributed.send(frame, self._drafter_rank, group=self._group)
        # Force the send to actually leave the local stream. ``mx.distributed.send``
        # returns an ``mx.array`` whose evaluation is what flushes the
        # underlying RDMA / ring transmit; without this kick MLX leaves
        # the send queued indefinitely while the target sits inside
        # ``mx.eval(drafts_buffer)`` polling for a response that the
        # peer can't produce -- a wire-level deadlock that masquerades
        # as "drafter never replies". Mirrors the pattern in
        # ``auto_parallel.flush_prefill_sends``.
        mx.async_eval(sent_frame)
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

    def _trim_blocking(self, session_id: int, n_positions: int) -> None:
        """Send a trim command and wait for the ack."""
        frame = _build_command_frame(
            op=OP_TRIM_CACHE,
            inputs=[],
            num_forwards=0,
            trim_amount=n_positions,
            session_id=session_id,
        )
        sent_frame = mx.distributed.send(frame, self._drafter_rank, group=self._group)
        mx.async_eval(sent_frame)
        ack = mx.distributed.recv(
            shape=(ACK_FRAME_SIZE,),
            dtype=mx.uint32,
            src=self._drafter_rank,
            group=self._group,
        )
        mx.eval(ack)
        if int(ack[0].item()) != ACK_OK:
            raise RuntimeError(
                f"Drafter rank reported error code {int(ack[0].item())} "
                f"for trim_cache(session={session_id}, n={n_positions})"
            )

    def _shutdown_blocking(self) -> None:
        """Send shutdown command and wait for the ack."""
        frame = _build_command_frame(
            op=OP_SHUTDOWN,
            inputs=[],
            num_forwards=0,
            trim_amount=0,
            session_id=SESSION_ID_NONE,
        )
        sent_frame = mx.distributed.send(frame, self._drafter_rank, group=self._group)
        mx.async_eval(sent_frame)
        ack = mx.distributed.recv(
            shape=(ACK_FRAME_SIZE,),
            dtype=mx.uint32,
            src=self._drafter_rank,
            group=self._group,
        )
        mx.eval(ack)

    def _reset_and_prefill_blocking(
        self, session_id: int, prompt_tokens: list[int]
    ) -> None:
        """Send the prefill command + token array and wait for the ack.

        The command frame announces ``num_prompt_tokens`` (encoded in
        the ``num_forwards`` slot) and the ``session_id`` to allocate
        / reset on the drafter rank. For an empty prefill we only send
        the command frame; the drafter's cache reset is implicit in
        every ``OP_PREFILL`` whether or not tokens follow.
        """
        num_prompt_tokens = len(prompt_tokens)
        frame = _build_command_frame(
            op=OP_PREFILL,
            inputs=[],
            num_forwards=num_prompt_tokens,
            trim_amount=0,
            session_id=session_id,
        )
        sent_frame = mx.distributed.send(frame, self._drafter_rank, group=self._group)
        mx.async_eval(sent_frame)
        if num_prompt_tokens > 0:
            tokens_array = mx.array(prompt_tokens, dtype=mx.uint32)
            sent_tokens = mx.distributed.send(
                tokens_array, self._drafter_rank, group=self._group
            )
            mx.async_eval(sent_tokens)
        ack = mx.distributed.recv(
            shape=(ACK_FRAME_SIZE,),
            dtype=mx.uint32,
            src=self._drafter_rank,
            group=self._group,
        )
        mx.eval(ack)
        if int(ack[0].item()) != ACK_OK:
            raise RuntimeError(
                f"Drafter rank reported error code {int(ack[0].item())} "
                f"for reset_and_prefill(session={session_id}, "
                f"{num_prompt_tokens} tokens)"
            )

    def _end_session_blocking(self, session_id: int) -> None:
        """Send OP_END_SESSION and wait for the ack."""
        frame = _build_command_frame(
            op=OP_END_SESSION,
            inputs=[],
            num_forwards=0,
            trim_amount=0,
            session_id=session_id,
        )
        sent_frame = mx.distributed.send(frame, self._drafter_rank, group=self._group)
        mx.async_eval(sent_frame)
        ack = mx.distributed.recv(
            shape=(ACK_FRAME_SIZE,),
            dtype=mx.uint32,
            src=self._drafter_rank,
            group=self._group,
        )
        mx.eval(ack)
        if int(ack[0].item()) != ACK_OK:
            raise RuntimeError(
                f"Drafter rank reported error code {int(ack[0].item())} "
                f"for end_session({session_id})"
            )


@final
class _SessionHandle:
    """Per-request :class:`DrafterTransport` view of a :class:`RemoteTransport`.

    Each in-flight target task gets its own handle via
    :meth:`RemoteTransport.open_session`. The handle's wire ops carry
    the handle's ``session_id`` so the drafter rank can route them to
    the right per-session KV cache.

    Lifecycle:

    * :meth:`reset_and_prefill` allocates the session on the drafter
      rank and seeds its KV cache with the prompt prefix.
    * :meth:`forward` / :meth:`trim_cache` advance / rollback the
      session's KV cache.
    * :meth:`shutdown` ends the session (sends ``OP_END_SESSION`` so
      the drafter rank frees the KV cache). Idempotent; safe to call
      from a generator's ``finally`` block.

    All methods raise :class:`RuntimeError` after :meth:`shutdown` so
    use-after-end mistakes surface immediately rather than corrupting
    a freshly allocated session that happens to reuse the id.
    """

    def __init__(self, *, owner: "RemoteTransport", session_id: int) -> None:
        self._owner = owner
        self._session_id = session_id
        self._closed = False

    @property
    def num_draft_tokens(self) -> int:
        return self._owner.num_draft_tokens

    @property
    def session_id(self) -> int:
        return self._session_id

    def forward(self, inputs: list[int], num_forwards: int) -> "DraftFuture":
        if self._closed:
            raise RuntimeError(
                f"_SessionHandle({self._session_id}).forward called after shutdown"
            )
        return self._owner._submit_forward(self._session_id, inputs, num_forwards)  # pyright: ignore[reportPrivateUsage]

    def trim_cache(self, n_positions: int) -> None:
        if self._closed:
            raise RuntimeError(
                f"_SessionHandle({self._session_id}).trim_cache called after shutdown"
            )
        self._owner._submit_trim(self._session_id, n_positions)  # pyright: ignore[reportPrivateUsage]

    def reset_and_prefill(self, prompt_tokens: list[int]) -> None:
        if self._closed:
            raise RuntimeError(
                f"_SessionHandle({self._session_id}).reset_and_prefill called after shutdown"
            )
        self._owner._submit_prefill(self._session_id, prompt_tokens)  # pyright: ignore[reportPrivateUsage]

    def shutdown(self) -> None:
        """End the session on the drafter rank. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._owner._submit_end_session(self._session_id)  # pyright: ignore[reportPrivateUsage]


def make_remote_transport(
    *,
    draft_model: "Model | None" = None,
    draft_cache: "KVCacheType | None" = None,
    num_draft_tokens: int,
    group: mx.distributed.Group | None = None,
    drafter_rank: int | None = None,
    target_rank: int | None = None,
) -> "RemoteTransport":
    """Construct a :class:`RemoteTransport` for the calling target rank.

    Returns the wire-protocol owner; per-task callers should call
    :meth:`RemoteTransport.open_session` to obtain a session-scoped
    :class:`DrafterTransport` view that the spec loop consumes. The
    factory does not implement ``DrafterTransport`` directly because
    its lifecycle is bound to the runner (long-lived) while the spec
    loop's transport is bound to a single request (short-lived).

    On the target rank this returns a transport that sends commands to
    ``drafter_rank``. The drafter rank should run
    :func:`drafter_serve_loop` instead (it loads the drafter model and
    cache factory, processes commands, sends drafts back).

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
    make_draft_cache: Callable[[], "KVCacheType"],
    num_draft_tokens: int,
    group: mx.distributed.Group,
    target_rank: int,
) -> None:
    """Run the drafter rank's command-loop until ``OP_SHUTDOWN``.

    Receives :data:`COMMAND_FRAME_SIZE`-element command frames from
    ``target_rank``, dispatches on the op code, executes the
    drafter-side work, and replies with the appropriate frame.

    Maintains a per-session KV cache (``sessions[session_id]``)
    allocated lazily on the first ``OP_PREFILL`` for each session and
    freed by ``OP_END_SESSION`` (or implicitly by ``OP_SHUTDOWN``).
    Multiple sessions may be live concurrently; the wire stays serial
    but the drafter rank multiplexes by ``session_id``.

    See module docstring for the wire protocol.
    """
    drafts_buffer_size = num_draft_tokens + 1
    sessions: dict[int, "KVCacheType"] = {}

    while True:
        frame = mx.distributed.recv(
            shape=(COMMAND_FRAME_SIZE,),
            dtype=mx.uint32,
            src=target_rank,
            group=group,
        )
        mx.eval(frame)
        op, inputs, num_forwards, trim_amount, session_id = _decode_command_frame(frame)

        if op == OP_SHUTDOWN:
            # Drop every session's cache before the serve loop returns
            # so the drafter rank's process exits with no dangling
            # KV-cache references holding GPU memory.
            sessions.clear()
            ack = mx.array([ACK_OK], dtype=mx.uint32)
            sent_ack = mx.distributed.send(ack, target_rank, group=group)
            # Force the reply to flush before we return -- the target
            # is waiting on this ack inside ``_shutdown_blocking`` and
            # we will tear the serve loop down right after.
            mx.eval(sent_ack)
            return

        if op == OP_END_SESSION:
            # Idempotent: ending a non-existent session is also a
            # successful ack. Forgetful targets (e.g. a runner that
            # crashed without calling shutdown on its session) are
            # cleaned up by the next ``OP_SHUTDOWN`` either way.
            sessions.pop(session_id, None)
            ack = mx.array([ACK_OK], dtype=mx.uint32)
            sent_ack = mx.distributed.send(ack, target_rank, group=group)
            mx.async_eval(sent_ack)
            continue

        if op == OP_TRIM_CACHE:
            session_cache = sessions.get(session_id)
            if session_cache is None:
                raise RuntimeError(
                    f"OP_TRIM_CACHE for unknown session {session_id}; "
                    f"OP_PREFILL must allocate the session first"
                )
            if trim_amount > 0:
                # ``mlx_trim_prompt_cache`` is typed against ``List[Cache]``
                # but exo's ``KVCacheType`` is structurally a list of
                # mlx_lm caches; the runtime types match exactly. We
                # erase to ``Any`` here to bypass list invariance.
                from typing import Any
                from typing import cast as _cast

                mlx_trim_prompt_cache(_cast(Any, session_cache), trim_amount)  # type: ignore[reportArgumentType]
            ack = mx.array([ACK_OK], dtype=mx.uint32)
            sent_ack = mx.distributed.send(ack, target_rank, group=group)
            mx.async_eval(sent_ack)
            continue

        if op == OP_FORWARD:
            session_cache = sessions.get(session_id)
            if session_cache is None:
                raise RuntimeError(
                    f"OP_FORWARD for unknown session {session_id}; "
                    f"OP_PREFILL must allocate the session first"
                )
            outputs = _run_drafter_forwards_remote(
                draft_model=draft_model,
                draft_cache=session_cache,
                inputs=inputs,
                num_forwards=num_forwards,
            )
            # Pad to fixed-shape buffer so the target's recv pre-allocation matches.
            padded = list(outputs) + [0] * (drafts_buffer_size - len(outputs))
            response = mx.array(padded, dtype=mx.uint32)
            sent_response = mx.distributed.send(response, target_rank, group=group)
            mx.async_eval(sent_response)
            continue

        if op == OP_PREFILL:
            # ``num_forwards`` is overloaded here as the prompt token
            # count (see _build_command_frame call site in
            # _reset_and_prefill_blocking).
            num_prompt_tokens = num_forwards
            # Allocate (or replace) the session's KV cache. Replacement
            # semantics let a target re-use a session_id after
            # OP_END_SESSION + OP_PREFILL without leaking the old cache.
            session_cache = make_draft_cache()
            sessions[session_id] = session_cache
            _reset_and_prefill_remote(
                draft_model=draft_model,
                draft_cache=session_cache,
                num_prompt_tokens=num_prompt_tokens,
                target_rank=target_rank,
                group=group,
            )
            ack = mx.array([ACK_OK], dtype=mx.uint32)
            sent_ack = mx.distributed.send(ack, target_rank, group=group)
            mx.async_eval(sent_ack)
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


_DRAFTER_PREFILL_STEP_SIZE: Final[int] = 4096
"""Chunk size for drafter-side prefill forwards.

Mirrors :func:`exo.worker.engines.mlx.generator.generate._spec_drafter_prefill`'s
``step`` default. Drafter weights are small (typically <2 GB) so the
4096-token chunks comfortably fit in the drafter rank's command queue
without OOM, even at long prompts."""


def _reset_and_prefill_remote(
    *,
    draft_model: "Model",
    draft_cache: "KVCacheType",
    num_prompt_tokens: int,
    target_rank: int,
    group: mx.distributed.Group,
) -> None:
    """Reset drafter cache and prefill against an incoming prompt.

    Pulled out as a free function (matches
    :func:`_run_drafter_forwards_remote`) so the drafter rank doesn't
    depend on any target-side code. The target rank already sent the
    ``OP_PREFILL`` command frame; this function handles the cache
    reset, recvs the prompt array (if any), and runs the prefill
    forwards. The serve loop sends the ack after this returns.
    """
    # Trim cache to offset 0 so the new prompt starts cleanly. KVCache's
    # offset is the only state we need to reset; SSM caches and other
    # exotic types are not in scope for the drafter (drafter models are
    # standard transformers by convention). If the offset is 0 the trim
    # is a no-op.
    current_offset = 0
    if draft_cache:
        # Every cache entry shares the same offset for transformer
        # drafters; use entry 0 as the source of truth.
        cache_zero = draft_cache[0]
        offset_attr = getattr(cache_zero, "offset", None)
        if isinstance(offset_attr, int):
            current_offset = offset_attr
    if current_offset > 0:
        from typing import cast as _cast

        mlx_trim_prompt_cache(_cast(list[object], draft_cache), current_offset)  # type: ignore[reportArgumentType]

    if num_prompt_tokens == 0:
        return

    # Pull the prompt array from the target rank.
    tokens = mx.distributed.recv(
        shape=(num_prompt_tokens,),
        dtype=mx.uint32,
        src=target_rank,
        group=group,
    )
    mx.eval(tokens)

    # Mirror :func:`_spec_drafter_prefill`: feed tokens through the
    # drafter model in chunks, advancing its KV cache.
    step = _DRAFTER_PREFILL_STEP_SIZE
    cursor = 0
    while cursor < num_prompt_tokens:
        chunk_end = min(cursor + step, num_prompt_tokens)
        chunk = tokens[cursor:chunk_end]
        draft_model(chunk[None], cache=draft_cache)
        mx.eval([c.state for c in draft_cache])  # type: ignore[reportArgumentType]
        cursor = chunk_end


__all__ = [
    "ACK_FRAME_SIZE",
    "ACK_OK",
    "COMMAND_FRAME_SIZE",
    "OP_END_SESSION",
    "OP_FORWARD",
    "OP_PREFILL",
    "OP_SHUTDOWN",
    "OP_TRIM_CACHE",
    "SESSION_ID_NONE",
    "RemoteTransport",
    "drafter_serve_loop",
    "make_remote_transport",
]


# Suppress the unused-import warnings for the future-only Future type:
# ThreadPoolExecutor.submit returns ``Future`` which is structurally
# compatible with :data:`DraftFuture`, but we annotate the return type
# inside the class body and the import is otherwise unused.
_ = Future
