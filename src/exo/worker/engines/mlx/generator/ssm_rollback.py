"""Captured-forward + per-position SSM rollback dispatch for spec decoding.

The pipelined drafter's verify loop processes ``k_this + 1`` tokens per
round in a single batched forward, then trims the rejected drafts from
the target cache via :func:`mlx_lm.models.cache.trim_prompt_cache`.
That trim primitive is a *no-op* for non-trimmable caches: any
``ArraysCache`` entry in the list short-circuits
:func:`mlx_lm.models.cache.can_trim_prompt_cache` to ``False`` and the
function returns ``0`` without touching state. For Qwen 3.5 hybrid
targets (GatedDeltaNet linear-attn layers backed by ``ArraysCache``,
full-attn layers backed by ``KVCache``), that silent no-op leaves the
SSM state polluted by the rejected drafts, the next round's verify
runs against a corrupted recurrence, and the model emits the all-``!``
output observed on the 122B-A10B / 397B-A17B asymmetric runs.

This module is the architecture-specific dispatch that lets the
pipelined verify loop recover correctness *without* falling back to
target-only generation:

  * :func:`supports_ssm_rollback` advertises whether a given target
    has a captured-forward + rollback contract on this build.

  * :func:`captured_verify` runs the verify forward and returns a
    :class:`SSMCaptureHandle` carrying the architecture's opaque
    intermediate state (Qwen 3.5: per-SSM-layer ``GdnState`` 11-tuples).

  * :func:`rollback_after_verify` consumes the handle to roll the
    cache back to ``num_accepted + 1`` committed tokens -- KV via the
    architecture's per-layer ``trim``, SSM via per-layer
    ``gated_delta_update`` replay against the captured intermediates.

The Qwen 3.5 dispatch reuses
:mod:`exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks` -- the same
captured-forward and rollback that the DFlash coupled drafter uses,
now repurposed for the asymmetric/pipelined path where no coupled
drafter is present. The hook sentinel
:func:`~exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks.has_dflash_hooks`
is intentionally NOT consulted here: the sentinel is a coupled-drafter
dispatch marker (its presence means "this target is wired for
``CoupledModelDrafter``"); the captured-forward and rollback functions
themselves are pure of any coupled-drafter requirement and only need
the target to be a Qwen 3.5 text model.

Adding a new SSM-hybrid architecture: implement the captured-forward
+ rollback in
:mod:`exo.worker.engines.mlx.vendor.<arch>_hooks`, extend the
:func:`supports_ssm_rollback` switch and the two dispatch functions
below to recognise it, and add an equivalence regression to
:mod:`exo.worker.tests.unittests.test_mlx.test_ssm_rollback`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, final

import mlx.core as mx

from exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks import (
    GdnState,
    qwen3_5_dflash_forward,
    qwen3_5_rollback_speculative_cache,
    resolve_qwen3_5_text_model,
)


@final
@dataclass(frozen=True)
class SSMCaptureHandle:
    """Opaque handle returned by :func:`captured_verify`.

    Carries the per-SSM-layer intermediates the architecture's
    rollback consumes. Frozen so the pipelined verify loop can stash
    one instance per round without worrying about in-place mutation,
    and so handles from different rounds cannot be accidentally
    merged.

    ``gdn_states`` is the only field today (Qwen 3.5 captures
    11-tuples of ``q, k, v, a, b, A_log, dt_bias, state, mask,
    conv_input, conv_kernel_size`` per gated-delta layer). When a
    second architecture's rollback wants a different intermediate
    shape, add a fresh field rather than reshaping
    ``gdn_states`` -- the rollback dispatch reads the right field per
    architecture and unused fields stay empty.
    """

    gdn_states: list[GdnState]


def supports_ssm_rollback(model: object) -> bool:
    """True iff ``model`` has a captured-forward + SSM rollback contract.

    Used by the generator's draft-mode demotion gate: when
    ``has_non_kv_caches(caches)`` is True we normally demote ``model``
    / ``pipelined`` drafting to ``none`` (the generic trim primitive
    is a no-op for SSM caches and would silently poison subsequent
    rounds). When this function returns True, the pipelined loop has
    an architecture-specific rollback path it can use instead, so the
    demotion is skipped.

    Returning True is the *capability* signal -- whether the path is
    actually exercised depends on the pipelined loop being routed
    with ``ssm_aware=True``.
    """
    return resolve_qwen3_5_text_model(model) is not None


def is_target_ssm_trim_unsafe(
    *,
    pipelined_ssm_aware: bool,
    target_hit: int,
    drafter_hit: int,
) -> bool:
    """True iff the pre-spec cache-alignment trim of the target cache is unsafe.

    ``mlx_trim_prompt_cache`` is a NO-OP for non-trimmable cache entries
    (``ArraysCache``, e.g. Qwen 3.5 GatedDeltaNet SSM state). When the
    pipelined-SSM-aware path bypasses the demotion gate AND the target
    prefix-cache hit overshoots the drafter's, the caller would otherwise
    rewrite ``prompt_tokens`` / ``prefix_hit_length`` as if the trim
    succeeded while the underlying SSM state remained at the original
    offset -- silently corrupting verify/propose. The verify-loop
    captured-forward + replay-rollback path
    (:func:`captured_verify` / :func:`rollback_after_verify`) only rewinds
    SSM state by N tokens *during speculation* using GDN states from the
    same forward pass; it cannot retroactively rewind an arbitrary warm
    SSM cache to match a colder drafter cache.

    Returning True signals the caller should demote this single request to
    ``draft_mode='none'`` so target-only generation runs on the warm
    target cache without an unsafe trim. The next request with a warm
    drafter prefix cache will speculate normally.

    Args:
        pipelined_ssm_aware: ``True`` iff the request is on the
            pipelined SSM-aware path (i.e. the demotion gate at
            ``generate.py`` was bypassed because the model has the
            captured-forward + replay-rollback contract).
        target_hit: Tokens already in the target prefix cache.
        drafter_hit: Tokens already in the drafter prefix cache.
    """
    return pipelined_ssm_aware and target_hit > drafter_hit


def captured_verify(
    model: object,
    inputs: mx.array,
    cache: list[Any],
) -> tuple[mx.array, SSMCaptureHandle]:
    """Run the verify forward with SSM intermediate capture.

    Drop-in replacement for the standard
    ``logits = model(inputs, cache=cache)`` call used by the pipelined
    verify loop. The returned :class:`SSMCaptureHandle` is opaque to
    the caller -- pass it back to :func:`rollback_after_verify` on
    partial acceptance and discard otherwise.

    Raises:
        TypeError: ``model`` is not an SSM-rollback-capable target.
            Callers should gate this dispatch on
            :func:`supports_ssm_rollback` (or fall back to the
            standard forward) -- reaching this branch unguarded
            indicates the pipelined loop was routed with
            ``ssm_aware=True`` against an unsupported target.
    """
    if resolve_qwen3_5_text_model(model) is None:
        raise TypeError(
            "captured_verify called against a non-SSM-rollback-capable target; "
            f"got {type(model).__name__!r}. The pipelined loop must gate this "
            "dispatch on supports_ssm_rollback() and fall back to the standard "
            "forward when False."
        )
    output = qwen3_5_dflash_forward(
        model,
        inputs,
        cache=cache,
        capture_gdn_states=True,
    )
    return output.logits, SSMCaptureHandle(gdn_states=output.gdn_states)


def rollback_after_verify(
    model: object,
    cache: list[Any],
    captured: SSMCaptureHandle,
    num_accepted: int,
    block_size: int,
) -> None:
    """Roll the cache back to ``num_accepted + 1`` committed positions.

    Counterpart to :func:`captured_verify`: trims KV caches by
    ``block_size - (num_accepted + 1)`` and rewinds SSM state via
    per-layer ``gated_delta_update`` replay against the captured
    intermediates.

    Idempotent when ``num_accepted == block_size - 1`` (full accept):
    the architecture hooks compute ``trim == 0`` and skip both the KV
    in-place trim and the SSM replay. Callers can either gate the
    invocation on ``num_accepted < block_size - 1`` or let it
    short-circuit cheaply.

    Raises:
        TypeError: same gating contract as :func:`captured_verify` --
            reaching this branch against an unsupported target means
            the pipelined loop's routing drifted from the capability
            signal.
    """
    if resolve_qwen3_5_text_model(model) is None:
        raise TypeError(
            "rollback_after_verify called against a non-SSM-rollback-capable "
            f"target; got {type(model).__name__!r}. Same gating contract as "
            "captured_verify -- supports_ssm_rollback() must hold for the "
            "calling site."
        )
    qwen3_5_rollback_speculative_cache(
        model,
        caches=cache,
        gdn_states=captured.gdn_states,
        accepted=num_accepted,
        block_size=block_size,
    )


__all__ = [
    "SSMCaptureHandle",
    "captured_verify",
    "rollback_after_verify",
    "supports_ssm_rollback",
]
