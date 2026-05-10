"""Single-node coupled (mtp/dflash) speculative-decoding dispatch.

mlx-vlm ships a ready-to-use round loop -- :func:`mlx_vlm.generate._mtp_rounds`
-- but it expects the target language model to expose two methods that
exo's pinned ``mlx-lm`` fork doesn't natively provide:

- ``rollback_speculative_cache``
- ``__call__(..., return_hidden=True, return_shared_kv=True)`` returning a
  ``LanguageModelOutput``-shaped object.

We satisfy that contract WITHOUT mutating mlx-lm's classes (which
would persist for every other instance the runner ever loads) by
wrapping the loaded target in :class:`Gemma4MTPTargetAdapter`.
The adapter forwards forward-passes and rollbacks through the
package-level functions in
:mod:`exo.worker.engines.mlx.vendor.gemma4_mtp_hooks`, which were
vendored from mlx-vlm's gemma4 language model.

This module provides:

- :class:`Gemma4MTPTargetAdapter` -- the wrapper that satisfies
  ``_mtp_rounds`` -- and bind-time access to the underlying
  ``embed_tokens`` slot the drafter walks during ``bind``.
- :func:`run_coupled_round_loop` -- a thin generator that drives the
  mlx-vlm round loop given a prefilled target cache and the captured
  prefill intermediates. The caller owns prefill + emission +
  cancellation; this function is a pure round-loop driver, kept narrow
  so it can be swapped for a vendored loop later without disturbing
  ``mlx_generate``'s control flow.

DFlash forward-compat: the adapter is Gemma 4-specific because that's
the only target type the vendored hooks support today. Adding DFlash
will introduce a sibling adapter (``Qwen3DFlashTargetAdapter``) that
mirrors this surface against a different vendor module.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any, cast, final

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gemma4_text import Model as Gemma4Model

from exo.worker.engines.mlx.vendor.gemma4_mtp_hooks import (
    Gemma4MTPForwardOutput,
    gemma4_mtp_forward,
    gemma4_rollback_speculative_cache,
    has_mtp_hooks,
)


# mlx-vlm's ``_mtp_rounds`` is a private module-level helper without a
# typed stub; resolve it lazily through ``importlib`` so the
# type-check narrowing happens at the import boundary instead of every
# call site. The eager-import path would force every coupled-drafter
# call site to ride a multi-line ``pyright: ignore`` block.
def _resolve_mtp_rounds_fn() -> (
    Callable[..., Generator[tuple[int, None], None, None]]
):
    import importlib

    module: Any = importlib.import_module("mlx_vlm.generate")
    return cast(
        "Callable[..., Generator[tuple[int, None], None, None]]",
        module._mtp_rounds,
    )


@final
class Gemma4MTPTargetAdapter:
    """Adapter that exposes the ``_mtp_rounds`` target contract.

    mlx-vlm's ``_mtp_rounds`` does three things with the target it
    receives:

    1. ``lm = model.language_model if hasattr(model, "language_model") else model``
       and then walks ``lm.embed_tokens``, ``lm.embed_scale``, etc.
       via the drafter's ``bind`` step.
    2. ``lm.rollback_speculative_cache(cache, gdn_states, accepted, bs)``.
    3. ``lm(verify_input, cache=..., return_hidden=True, return_shared_kv=True)``
       returning an object with ``.logits``, ``.hidden_states``, and
       ``.shared_kv_states``.

    For (1), the underlying ``mlx_lm.models.gemma4_text.Model``
    already satisfies the structure: ``Model.model.embed_tokens`` is
    populated and the drafter's ``bind`` walks ``model.embed_tokens``
    OR ``model.model.embed_tokens``. The adapter exposes ``model``
    as a passthrough so the drafter binds to the SAME embed_tokens
    instance it would have bound to without the adapter -- no weight
    duplication, no bind-time divergence.

    For (2) and (3), the adapter wires ``rollback_speculative_cache``
    and ``__call__`` through to the vendored hook functions.

    The adapter is a plain class (NOT an ``nn.Module``) -- it holds no
    parameters of its own and the ``__call__`` return type
    (:class:`Gemma4MTPForwardOutput`) is incompatible with
    ``Module.__call__``'s ``mx.array`` return. The wrapped target
    keeps its own parameters and continues to be eligible for
    ``mx.eval`` / cache-resizing as before.
    """

    def __init__(self, target_model: Gemma4Model) -> None:
        if not has_mtp_hooks(target_model):
            # The hook attach is gated by ``utils_mlx.load_mlx_items`` --
            # if we got here without the attach call, the loader's
            # post-load wiring drifted from this dispatch.
            raise RuntimeError(
                "Gemma4MTPTargetAdapter requires a target with attached "
                "MTP hooks; call attach_mtp_hooks(target) at load time. "
                "This is a runtime guard against loader/dispatch drift."
            )
        self._target: Gemma4Model = target_model

    @property
    def target(self) -> Gemma4Model:
        """The underlying mlx-lm gemma4 model (escape hatch for tests)."""
        return self._target

    @property
    def model(self) -> nn.Module:
        """``Model.model`` passthrough used by the drafter's ``bind``."""
        return self._target.model

    def __call__(
        self,
        inputs: mx.array,
        *,
        cache: list[Any] | None = None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
    ) -> Gemma4MTPForwardOutput:
        """Forward pass returning the MTP-flavoured capture tuple.

        ``_mtp_rounds`` always passes ``return_hidden=True`` and
        ``return_shared_kv=True``, so the hot path is the captured-
        forward case. We accept the off variants for API parity.

        Note that the return type is :class:`Gemma4MTPForwardOutput`,
        NOT raw logits. Calling sites that want raw logits (e.g. the
        prefill path before entering the round loop) should call this
        the same way and read ``.logits`` -- the structural shape lets
        ``_mtp_rounds`` read ``.hidden_states[-1]`` and
        ``.shared_kv_states`` directly without an unwrap step.
        """
        return gemma4_mtp_forward(
            self._target,
            inputs,
            cache=cache,
            return_hidden=return_hidden,
            return_shared_kv=return_shared_kv,
        )

    def rollback_speculative_cache(
        self,
        caches: list[Any],
        gdn_states: object,
        accepted: int | mx.array,
        block_size: int,
    ) -> int:
        """Trim target KV caches after partial-acceptance.

        Delegated to :func:`gemma4_rollback_speculative_cache`; see
        that function's docstring for ``gdn_states`` semantics
        (accepted-and-ignored for Gemma 4, used by DFlash on Qwen3).
        """
        return gemma4_rollback_speculative_cache(
            self._target,
            caches=caches,
            gdn_states=gdn_states,
            accepted=accepted,
            block_size=block_size,
        )


def run_coupled_round_loop(
    *,
    target: Gemma4Model,
    drafter: nn.Module,
    prompt_cache: list[Any],
    prefill_output: Gemma4MTPForwardOutput,
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: int | None,
    token_dtype: mx.Dtype = mx.int32,
) -> Generator[int, None, None]:
    """Drive mlx-vlm's MTP round loop and yield decoded token ids.

    The caller (``mlx_generate``'s coupled-drafter dispatch) is
    responsible for:

    - encoding the prompt + prefilling ``prompt_cache`` via
      :func:`gemma4_mtp_forward` to obtain ``prefill_output``;
    - sampling the first bonus token from ``prefill_output.logits[:, -1:, :]``
      and emitting it as the first decode token (this function does
      NOT yield the first bonus -- it picks up from round 1);
    - threading the yielded tokens through the existing emission path
      (cancellation checks, stop-token detection, ``GenerationResponse``
      construction, usage accounting).

    Why split the loop driver from the surrounding I/O contract: the
    round loop's correctness (rollback, accept-walk, set_shared_kv
    sequencing) is independent of how exo emits tokens. Keeping the
    driver narrow means tests can mock target + drafter and exercise
    the loop without instantiating the full ``GenerationResponse``
    pipeline.

    Implementation note: we delegate the actual round logic to
    mlx-vlm's :func:`mlx_vlm.generate._mtp_rounds` rather than
    re-implement it. mlx-vlm owns the canonical accept-walk +
    rollback semantics, and re-implementing them would create a
    silent-divergence risk every time mlx-vlm tightens the loop.
    """
    if not prefill_output.hidden_states:
        # Should be unreachable: callers MUST request hidden capture
        # in the prefill forward (otherwise the drafter has nothing
        # to consume on round 1). Surface as a clear error rather
        # than letting ``_mtp_rounds`` index into an empty list.
        raise RuntimeError(
            "run_coupled_round_loop requires the prefill_output to "
            "carry a captured hidden state. Call gemma4_mtp_forward "
            "with return_hidden=True before entering the round loop."
        )

    adapter = Gemma4MTPTargetAdapter(target)
    last_hidden = prefill_output.hidden_states[-1]
    shared_kv = prefill_output.shared_kv_states
    mtp_rounds = _resolve_mtp_rounds_fn()

    for token, _unused in mtp_rounds(
        adapter,
        drafter,
        prompt_cache,
        last_hidden,
        shared_kv,
        first_bonus=first_bonus,
        max_tokens=max_tokens,
        sampler=sampler,
        draft_block_size=draft_block_size,
        token_dtype=token_dtype,
    ):
        yield token


__all__ = [
    "Gemma4MTPTargetAdapter",
    "run_coupled_round_loop",
]
