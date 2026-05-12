"""Tests for :mod:`exo.worker.engines.mlx.generator.ssm_rollback`.

The SSM rollback dispatch is the pipelined-drafter equivalent of the
DFlash coupled-drafter contract: it lets the pipelined verify loop on
Qwen 3.5 SSM-hybrid targets (35B-A3B / 122B-A10B / 397B-A17B) keep
running speculative decoding correctly without falling back to
target-only generation. The dispatch is gated by
:func:`supports_ssm_rollback`; the captured-forward branch swaps the
standard ``model(inputs, cache=cache)`` for
:func:`captured_verify`, and the trim primitive is replaced with
:func:`rollback_after_verify`.

These tests build a small Qwen 3.5 in-memory and assert that:

  1. :func:`supports_ssm_rollback` answers True for a real Qwen 3.5
     instance and False for unrelated callables.

  2. :func:`captured_verify` produces logits that are bit-identical to
     a standard ``Qwen3_5.TextModel.__call__`` over the same input,
     within the Metal-kernel noise floor.

  3. :func:`captured_verify` + :func:`rollback_after_verify` rolls the
     prompt cache back to a state semantically equivalent to a fresh
     forward over only the accepted prefix -- the argmax of the
     subsequent forward must match.

The synthetic Qwen 3.5 helper is the same shape as in
:mod:`test_qwen3_5_dflash_hooks`; both modules share the
``full_attention_interval=2`` layout that interleaves SSM and KV
layers so the rollback covers both cache families.
"""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
from mlx_lm.models.qwen3_5 import (
    TextModel as Qwen3_5LanguageModel,
)
from mlx_lm.models.qwen3_5 import (
    TextModelArgs,
)

from exo.worker.engines.mlx.generator.ssm_rollback import (
    SSMCaptureHandle,
    captured_verify,
    rollback_after_verify,
    supports_ssm_rollback,
)


def _build_tiny_qwen3_5(*, num_layers: int = 4) -> Qwen3_5LanguageModel:
    """Construct a small Qwen 3.5 language model in-memory.

    Mirrors ``_build_tiny_qwen3_5`` in
    :mod:`test_qwen3_5_dflash_hooks`: 4-layer hybrid (two SSM + two
    attention) is the minimum configuration that exercises both cache
    families AND the per-layer SSM replay path that the DFlash rollback
    fix added.
    """
    args = TextModelArgs(
        model_type="qwen3_5_text",
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=num_layers,
        intermediate_size=256,
        vocab_size=128,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        head_dim=32,
        full_attention_interval=2,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_value_head_dim=64,
        num_experts=0,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        attention_bias=False,
        num_experts_per_tok=0,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=0,
        moe_intermediate_size=0,
        norm_topk_prob=True,
        partial_rotary_factor=0.25,
        rope_scaling=None,
        rope_parameters={},
    )
    model = Qwen3_5LanguageModel(args)
    model.eval()
    return model


def _fresh_cache(model: Qwen3_5LanguageModel) -> list[Any]:
    return cast("list[Any]", model.make_cache())


def test_supports_ssm_rollback_true_for_qwen3_5() -> None:
    """The dispatch advertises Qwen 3.5 as SSM-rollback-capable.

    This is the capability signal the generator's draft-mode gate
    consults to decide whether to demote pipelined drafting against
    an SSM-hybrid target. False here would re-introduce the silent
    no-op trim and the all-``!`` output regression.
    """
    model = _build_tiny_qwen3_5()
    assert supports_ssm_rollback(model)


def test_supports_ssm_rollback_false_for_plain_object() -> None:
    """A non-Qwen-3.5 object is correctly reported as unsupported.

    Defends against future architecture additions inadvertently
    advertising support without wiring captured-forward + rollback.
    """

    class _NotAModel:
        """Sentinel: shape-compatible with ``object``, no Qwen 3.5 ancestor."""

    assert not supports_ssm_rollback(_NotAModel())


def test_captured_verify_matches_standard_forward_logits() -> None:
    """Captured forward emits the same logits as ``TextModel.__call__``.

    The pipelined verify loop pre-fix called ``model(...)`` directly;
    post-fix it routes SSM-hybrid targets through :func:`captured_verify`
    instead. For determinism to hold across the prefill/verify boundary
    -- the prompt cache is built by the standard prefill path and then
    consumed by the verify path -- the captured forward must produce
    bit-identical logits (modulo Metal noise) given the same input and
    cache state.
    """
    mx.random.seed(0)
    model = _build_tiny_qwen3_5(num_layers=4)
    tokens = mx.array([[1, 2, 3, 4]])

    cache_standard = _fresh_cache(model)
    standard_logits = model(tokens, cache=cache_standard)

    cache_captured = _fresh_cache(model)
    captured_logits, _ = captured_verify(model, tokens, cache_captured)

    diff = float(mx.abs(standard_logits - captured_logits).max().item())
    assert diff < 1e-3, (
        f"captured_verify diverged from standard forward by {diff:.3e}; "
        "Metal noise floor is ~1e-4 so this exceeds the tolerance and "
        "indicates the captured forward is no longer a faithful vendor "
        "of TextModel.__call__"
    )

    argmax_standard = mx.argmax(standard_logits, axis=-1)
    argmax_captured = mx.argmax(captured_logits, axis=-1)
    assert bool(mx.array_equal(argmax_standard, argmax_captured).item()), (
        "Captured forward picks a different greedy token than the "
        "standard forward -- the pipelined verify path would diverge "
        "from the prefill path and accept/reject decisions would not "
        "agree across rounds"
    )


def test_captured_verify_returns_gdn_states_for_each_ssm_layer() -> None:
    """The capture handle holds one ``GdnState`` per gated-delta layer.

    :func:`rollback_after_verify` iterates the gdn_states list and
    matches each entry against the corresponding non-trimmable cache
    entry. Drift between the two lists would silently roll the wrong
    layer's state back and produce garbage on partial accept; this
    test guards that invariant at the smallest possible model size.
    """
    model = _build_tiny_qwen3_5(num_layers=4)
    cache = _fresh_cache(model)
    tokens = mx.array([[1, 2, 3, 4]])

    _, handle = captured_verify(model, tokens, cache)

    # 4 layers, full_attention_interval=2 -> [linear, attn, linear, attn]
    # -> 2 SSM layers. The handle must carry one GdnState per SSM layer.
    expected_ssm_layers = 2
    assert len(handle.gdn_states) == expected_ssm_layers


def test_pipelined_verify_then_rollback_argmax_equivalence() -> None:
    """End-to-end equivalence: verify+rollback roundtrips to prefix state.

    Simulates the pipelined drafter's per-round contract:

      1. forward(``[seed, *drafts]``) -- the verify forward over K+1
         tokens with capture enabled.
      2. rollback to ``num_accepted + 1`` committed positions.
      3. forward(``next_tokens``) -- the next round's seed forward.

    Reference path:

      1. forward(``[seed, *drafts][:num_accepted+1]``) -- a fresh
         forward over only the committed prefix.
      2. forward(``next_tokens``) -- same next-round forward.

    For greedy decoding (the only case the pipelined drafter's
    accept-by-equality contract is correct for), the two paths must
    pick the same argmax token. The exact-equality assertion below is
    what the pipelined loop's accept/reject loop relies on; argmax
    drift here would surface as the all-``!`` regression.
    """
    mx.random.seed(0)
    model = _build_tiny_qwen3_5(num_layers=4)
    block_tokens = mx.array([[1, 2, 3, 4]])
    num_accepted = 2

    # Reference cache: forward over the committed prefix only.
    cache_ref = _fresh_cache(model)
    _ = model(block_tokens[:, : num_accepted + 1], cache=cache_ref)

    # Test cache: forward over the full block, then rollback.
    cache_test = _fresh_cache(model)
    _, handle = captured_verify(model, block_tokens, cache_test)
    rollback_after_verify(
        model,
        cache_test,
        handle,
        num_accepted=num_accepted,
        block_size=block_tokens.shape[1],
    )

    next_tokens = mx.array([[14, 15, 16]])
    out_ref = model(next_tokens, cache=cache_ref)
    out_test = model(next_tokens, cache=cache_test)

    argmax_ref = mx.argmax(out_ref, axis=-1)
    argmax_test = mx.argmax(out_test, axis=-1)
    assert bool(mx.array_equal(argmax_ref, argmax_test).item()), (
        f"Subsequent forward argmax diverges after pipelined rollback: "
        f"ref={argmax_ref.tolist()}, test={argmax_test.tolist()} -- "
        "this is the regression that produced all-``!`` output on Qwen 3.5 "
        "SSM-hybrid targets before the per-layer A_log/dt_bias replay fix "
        "in qwen3_5_rollback_speculative_cache"
    )


def test_handle_is_immutable() -> None:
    """The capture handle is frozen so loop-local stashing is safe.

    The pipelined loop stashes one handle per round; if the dataclass
    were mutable a subsequent round's in-place update could
    retroactively corrupt the previous round's rollback (e.g. if the
    rollback is deferred or the handle is captured by a future).
    """
    handle = SSMCaptureHandle(gdn_states=[])
    try:
        cast(Any, handle).gdn_states = ["mutated"]
    except (AttributeError, TypeError):
        return
    raise AssertionError(
        "SSMCaptureHandle is not frozen: mutation succeeded where it "
        "should have raised. Per-round stashing in the pipelined loop "
        "is no longer safe."
    )
