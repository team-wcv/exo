"""DFlash target-side hooks for Qwen 3.5, vendored from mlx-vlm (skeleton).

The mlx-vlm DFlash drafter (``mlx_vlm.speculative.drafters.qwen3_dflash``)
needs the same two methods on the target language model that
:mod:`exo.worker.engines.mlx.vendor.gemma4_mtp_hooks` exposes for Gemma 4:

1. ``forward_with_capture(inputs, cache, return_hidden, return_shared_kv)``
   -- a forward pass that returns logits **plus** a captured hidden tensor
   and the per-layer-type shared-KV / SSM-state snapshots that DFlash's
   round loop walks. Qwen 3.5 has a hybrid attention+SSM (gated-delta)
   architecture, so the capture surface includes the SSM state in
   addition to the attention KVs.

2. ``rollback_speculative_cache(caches, gdn_states, accepted, block_size)``
   -- per-layer KV trim **and** per-row SSM-state rewind (via
   ``gated_delta_update``) used after partial-acceptance rounds. Unlike
   the Gemma 4 hook (which ignores ``gdn_states``), the Qwen 3.5 hook
   has to actually rewind the gated-delta state because Qwen 3.5 layers
   alternate attention / SSM and the SSM cache is path-dependent.

Status
------
Skeleton only. The actual forward + rollback implementations live in
mlx-vlm's ``mlx_vlm/models/qwen3_5/language.py`` (rollback at line
~450; capture happens via the ``capture_layer_ids`` kwarg on the
``LanguageModel.__call__`` body) but cannot be directly imported
because exo runs Qwen 3.5 through ``mlx_lm.models.qwen3`` /
``mlx_lm.models.qwen3_5`` (different class lineage and attribute
spellings). Translating those bodies onto mlx-lm's surface mirrors
the work in :mod:`gemma4_mtp_hooks` and is left as a follow-up.

What this module DOES provide today:

- :func:`resolve_qwen3_5_text_model` mirrors
  :func:`resolve_gemma4_text_model` so the loader can locate the
  inner LM regardless of multimodal-wrapper shape.
- :func:`attach_dflash_hooks` is the kind-routed entry point called
  by :func:`utils_mlx._dispatch_attach_coupled_hooks`. Today it
  raises :class:`DFlashHooksNotImplementedError` so the loader's
  ``except (TypeError, DFlashHooksNotImplementedError)`` handler
  discards the drafter and falls back to standard drafting. Once the
  vendor work below lands, this becomes a one-line ``setattr``
  identical to the Gemma 4 sentinel write.
- :func:`has_dflash_hooks` mirrors :func:`gemma4_mtp_hooks.has_mtp_hooks`.

What this module DOES NOT provide yet:

- ``qwen3_5_dflash_forward`` (the captured forward) -- the function
  is declared with the right signature and raises
  :class:`DFlashHooksNotImplementedError`. Vendor source: mlx-vlm
  ``mlx_vlm/models/qwen3_5/language.py`` ``LanguageModel.__call__``
  body, plus the ``capture_layer_ids`` plumbing.
- ``qwen3_5_rollback_speculative_cache`` -- same pattern. Vendor
  source: mlx-vlm ``mlx_vlm/models/qwen3_5/language.py``
  ``LanguageModel.rollback_speculative_cache`` (line ~450).

When those land, also flip ``"dflash"`` into
:data:`exo.worker.engines.mlx.generator.coupled_drafter.DISPATCHABLE_COUPLED_DRAFTER_KINDS`
and add a sibling ``Qwen3DFlashTargetAdapter`` to
:mod:`coupled_drafter` that delegates to the new functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, final

import mlx.core as mx

# We intentionally import qwen3 from mlx-lm's text-only path (NOT mlx-vlm)
# so the type guard in :func:`resolve_qwen3_5_text_model` matches what
# exo's loader actually returns. mlx-lm's ``qwen3`` module is the dense
# Qwen3 family; if a ``qwen3_5`` (hybrid attention+SSM) module appears
# in mlx-lm in the future, the guard expands to a ``Union`` here without
# changing call sites.
try:
    from mlx_lm.models.qwen3 import Model as Qwen3Model
except ImportError:  # pragma: no cover - mlx-lm always ships qwen3 today
    Qwen3Model = None


# Attribute name that marks a target instance as "DFlash hooks attached".
# Symmetric with :data:`gemma4_mtp_hooks._MTP_HOOKS_ATTACHED_ATTR`. Kept on
# its own constant (rather than a single shared "coupled hooks attached"
# flag) so a target wired for one kind cannot be silently mistaken for a
# target wired for the other.
_DFLASH_HOOKS_ATTACHED_ATTR: Final[str] = "_exo_dflash_hooks_attached"


class DFlashHooksNotImplementedError(RuntimeError):
    """Raised when the DFlash hook surface is not yet vendored.

    Caught by :func:`utils_mlx._try_load_coupled_drafter`'s caller in the
    same arm as :class:`TypeError` so the runner degrades to standard
    drafting instead of crashing. Distinct exception type so logs are
    unambiguous (``TypeError`` -> wrong target architecture; this class
    -> right architecture, hooks not yet vendored).
    """


@final
@dataclass(frozen=True, kw_only=True)
class Qwen3DFlashForwardOutput:
    """Captured output of a DFlash-flavoured Qwen 3.5 forward pass.

    Mirrors :class:`gemma4_mtp_hooks.Gemma4MTPForwardOutput` but adds the
    ``ssm_states`` slot DFlash needs. The fields populated depend on the
    capture flags passed to the forward; an unrequested capture leaves the
    corresponding container empty rather than ``None`` so call sites can
    iterate without a per-field ``is None`` guard.

    - ``logits``: ``[B, T, vocab]`` post-LM-head logits.
    - ``hidden_states``: list of ``[B, T, hidden]`` pre-norm hiddens
      (one per ``capture_layer_ids`` entry, or the last layer alone in
      the default case).
    - ``shared_kv_states``: ``{layer_type: (K, V)}`` snapshot of each
      layer-type's shared attention-KV slot at the END of the forward.
    - ``ssm_states``: ``{layer_index: state_tensor}`` snapshot of every
      gated-delta layer's recurrent state. Empty for pure-attention
      forwards (the round loop only requests SSM capture when running
      against a hybrid target).
    """

    logits: mx.array
    hidden_states: list[mx.array]
    shared_kv_states: dict[str, tuple[mx.array, mx.array]]
    ssm_states: dict[int, mx.array]


def resolve_qwen3_5_text_model(target_model: object) -> object | None:
    """Return the inner Qwen 3.5 text model or ``None``.

    Symmetric with :func:`gemma4_mtp_hooks.resolve_gemma4_text_model`.
    Today the multimodal wrapper case is hypothetical (mlx-lm ships
    Qwen 3.5 as a single text class) but the indirection lets the
    loader code stay shape-agnostic when a vision-paired Qwen 3.5
    target lands.
    """
    if Qwen3Model is None:
        return None
    if isinstance(target_model, Qwen3Model):
        return target_model
    inner: object = getattr(target_model, "language_model", None)
    if isinstance(inner, Qwen3Model):
        return inner
    return None


def attach_dflash_hooks(target_model: object) -> None:
    """Mark a Qwen 3.5 target as DFlash-hooks-attached.

    Today this raises :class:`DFlashHooksNotImplementedError`; the
    loader's caller catches the exception and falls back to the
    standard drafter path so a dflash-only configuration degrades
    cleanly rather than crashing the runner.

    When the vendor work in this module's docstring lands, the body
    becomes::

        inner = resolve_qwen3_5_text_model(target_model)
        if inner is None:
            raise TypeError(
                "attach_dflash_hooks expected a Qwen 3.5 target; "
                f"got {type(target_model).__name__!r}."
            )
        setattr(target_model, _DFLASH_HOOKS_ATTACHED_ATTR, True)
        if inner is not target_model:
            setattr(inner, _DFLASH_HOOKS_ATTACHED_ATTR, True)

    -- identical in shape to :func:`gemma4_mtp_hooks.attach_mtp_hooks`.
    """
    raise DFlashHooksNotImplementedError(
        "DFlash target-side hooks for Qwen 3.5 are not yet vendored. The "
        "drafter loaded successfully (mlx-vlm understands its weights) but "
        "exo's generator dispatch needs ``forward_with_capture`` and "
        "``rollback_speculative_cache`` translated onto mlx-lm's qwen3_5 "
        "surface before it can drive the DFlash round loop. Until that lands, "
        "the loader degrades to the standard drafter path. See "
        "exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks module docstring "
        "for the vendor source pointers."
    )


def has_dflash_hooks(target_model: object) -> bool:
    """True iff :func:`attach_dflash_hooks` has run on this target.

    Today always returns ``False`` since :func:`attach_dflash_hooks`
    raises before setting the sentinel. Kept separate from
    :func:`gemma4_mtp_hooks.has_mtp_hooks` so the dispatch path can
    answer "this target is wired for dflash" without conflating it
    with "this target is wired for mtp" -- the two coupled-drafter
    kinds are mutually exclusive on a given runner.
    """
    if bool(getattr(target_model, _DFLASH_HOOKS_ATTACHED_ATTR, False)):
        return True
    inner = resolve_qwen3_5_text_model(target_model)
    if inner is None or inner is target_model:
        return False
    return bool(getattr(inner, _DFLASH_HOOKS_ATTACHED_ATTR, False))


def qwen3_5_dflash_forward(
    target: object,
    inputs: mx.array,
    *,
    cache: list[Any] | None = None,
    return_hidden: bool = False,
    return_shared_kv: bool = False,
    return_ssm_states: bool = False,
) -> Qwen3DFlashForwardOutput:
    """Forward pass capturing the intermediates DFlash needs.

    Vendor target: mlx-vlm ``mlx_vlm/models/qwen3_5/language.py``
    ``LanguageModel.__call__`` (with the ``capture_layer_ids`` /
    ``return_hidden`` / ``return_shared_kv`` kwargs).

    Today raises :class:`DFlashHooksNotImplementedError`. Future
    implementation will mirror
    :func:`gemma4_mtp_hooks.gemma4_mtp_forward`.
    """
    del target, inputs, cache, return_hidden, return_shared_kv, return_ssm_states
    raise DFlashHooksNotImplementedError(
        "qwen3_5_dflash_forward is not yet implemented. Vendor source: "
        "mlx_vlm/models/qwen3_5/language.py LanguageModel.__call__ + "
        "capture_layer_ids plumbing."
    )


def qwen3_5_rollback_speculative_cache(
    target: object,
    *,
    caches: list[Any],
    gdn_states: object,
    accepted: int | mx.array,
    block_size: int,
) -> int:
    """Per-layer KV + SSM trim after a partial-acceptance round.

    Vendor target: mlx-vlm ``mlx_vlm/models/qwen3_5/language.py``
    ``LanguageModel.rollback_speculative_cache`` (line ~450). The
    SSM-rewind path uses ``mlx_vlm/...gated_delta_update`` and is
    where the bulk of the vendoring effort lives -- the attention
    rollback is straightforward per-row truncation.

    Today raises :class:`DFlashHooksNotImplementedError`.
    """
    del target, caches, gdn_states, accepted, block_size
    raise DFlashHooksNotImplementedError(
        "qwen3_5_rollback_speculative_cache is not yet implemented. Vendor "
        "source: mlx_vlm/models/qwen3_5/language.py "
        "LanguageModel.rollback_speculative_cache."
    )


__all__ = [
    "DFlashHooksNotImplementedError",
    "Qwen3DFlashForwardOutput",
    "attach_dflash_hooks",
    "has_dflash_hooks",
    "qwen3_5_dflash_forward",
    "qwen3_5_rollback_speculative_cache",
    "resolve_qwen3_5_text_model",
]
