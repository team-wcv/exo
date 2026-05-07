"""Drafter on a different MLX rank, IPC via ``mx.distributed.send/recv``.

This file defines the surface of :class:`RemoteTransport` (a concrete
:class:`DrafterTransport` implementation) and the matching drafter
serve loop that runs on the drafter rank. Both ride on the existing
``mx.distributed.Group`` initialised by exo's runner bootstrap, so the
underlying network transport is automatically:

  * **JACCL (RDMA over IB-verbs / Thunderbolt-bridge)** when the
    instance is :class:`MlxJacclInstance` (``MLX_IBV_DEVICES`` set,
    ``backend="jaccl"``), or
  * **ring (TCP)** when the instance is :class:`MlxRingInstance`
    (``backend="ring"``).

No new transport code in this module -- ``mx.distributed.send`` /
``recv`` carry the small fixed-size protocol messages, and MLX picks
the wire format per group backend.

Wire protocol (kept deliberately tiny so per-round IPC overhead is
microseconds, not milliseconds):

  * **Command frame** (target -> drafter), shape ``(8,)`` ``uint32``:
    ``[op, num_inputs, num_drafts, input_0, input_1, trim_amount, _, _]``.
    Fixed shape so :func:`mx.distributed.recv` can pre-allocate.
  * **Drafts frame** (drafter -> target), shape
    ``(self._num_draft_tokens,)`` ``uint32``: the K drafts. Padded with
    zeros if the round requested fewer than K drafts (the caller knows
    its requested count and slices accordingly).
  * **Ack frame** (drafter -> target), shape ``(1,)`` ``uint32``: a
    single status byte. Used after ``trim_cache`` and ``shutdown`` so
    the target rank can sync.

Op codes:

  * ``OP_PROPOSE`` (1): drafter runs ``num_drafts`` forwards starting
    from ``inputs[:num_inputs]``, advances its KV cache, and replies
    with a Drafts frame.
  * ``OP_TRIM_CACHE`` (2): drafter trims ``trim_amount`` positions from
    its KV cache and replies with an Ack frame.
  * ``OP_SHUTDOWN`` (3): drafter exits its serve loop, replies with an
    Ack frame, then closes.

The asymmetric rank topology (drafter rank vs. target ranks) is
constructed by :class:`MlxJacclInstance` / :class:`MlxRingInstance`
extension landing in Layer B. Until that lands, calling
:func:`make_remote_transport` raises ``NotImplementedError`` with a
pointer to the planned topology change, so the Protocol surface stays
honest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import mlx.core as mx

    from exo.worker.engines.mlx.generator.drafter_transport import DrafterTransport
    from exo.worker.engines.mlx.types import KVCacheType, Model


# Wire-protocol constants. Kept module-level (not behind any feature
# flag) so unit tests can exercise the framing in isolation without
# pulling in mx.distributed.
COMMAND_FRAME_SIZE: Final[int] = 8
"""Fixed size of a command frame (uint32 ints)."""

OP_PROPOSE: Final[int] = 1
OP_TRIM_CACHE: Final[int] = 2
OP_SHUTDOWN: Final[int] = 3


def make_remote_transport(
    *,
    draft_model: "Model | None" = None,
    draft_cache: "KVCacheType | None" = None,
    num_draft_tokens: int,
    group: "mx.distributed.Group | None" = None,
    drafter_rank: int | None = None,
    target_rank: int | None = None,
) -> "DrafterTransport":
    """Build a :class:`RemoteTransport` for one rank.

    On the *target* rank: ``draft_model`` and ``draft_cache`` are None
    (they live on the drafter rank). The transport sends propose /
    trim / shutdown commands to ``drafter_rank`` and receives drafts
    back.

    On the *drafter* rank: this constructor is **not** the right call
    site -- the drafter rank should run :func:`drafter_serve_loop`
    instead, which loads the drafter model + cache and serves command
    frames from ``target_rank``.

    Args:
        draft_model: Drafter model on the drafter rank, ``None`` on the
            target rank.
        draft_cache: Drafter KV cache on the drafter rank, ``None`` on
            the target rank.
        num_draft_tokens: K -- the maximum drafts per round.
        group: The MLX distributed group both ranks belong to. Must be
            non-None for the remote transport.
        drafter_rank: Rank index of the drafter inside ``group``.
        target_rank: Rank index of the calling target inside ``group``.

    Returns:
        A :class:`DrafterTransport`.

    Raises:
        NotImplementedError: until Layer B lands the asymmetric
            instance topology, ``MlxJacclInstance`` / ``MlxRingInstance``
            don't carry a ``drafter_rank`` field, so this constructor
            has no callers in production. The signature is final so
            Layer B is purely additive.
    """
    raise NotImplementedError(
        "RemoteTransport requires the asymmetric instance topology "
        "(drafter_rank field on MlxJacclInstance / MlxRingInstance), "
        "shipping in Layer B of this PR alongside the placement and "
        "runner changes. The signature is final; only the body and "
        "the matching drafter_serve_loop are pending."
    )


def drafter_serve_loop(
    *,
    draft_model: "Model",
    draft_cache: "KVCacheType",
    num_draft_tokens: int,
    group: "mx.distributed.Group",
    target_rank: int,
) -> None:
    """Run the drafter rank's command-loop.

    Receives :data:`COMMAND_FRAME_SIZE`-element command frames from
    ``target_rank`` over ``group``, dispatches on the op code,
    executes the drafter forward / cache trim / shutdown, and replies
    with a Drafts or Ack frame.

    Exits cleanly on receiving :data:`OP_SHUTDOWN`.

    See module docstring for the wire protocol.
    """
    raise NotImplementedError(
        "drafter_serve_loop ships in Layer B alongside the asymmetric "
        "topology and placement changes."
    )


__all__ = [
    "COMMAND_FRAME_SIZE",
    "OP_PROPOSE",
    "OP_SHUTDOWN",
    "OP_TRIM_CACHE",
    "drafter_serve_loop",
    "make_remote_transport",
]
