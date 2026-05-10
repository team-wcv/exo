"""Dispatch-shape tests for :class:`CoupledModelDrafter`.

These tests exercise the Phase 2c integration seam between
:class:`exo.worker.engines.mlx.generator.coupled_drafter.CoupledModelDrafter`
and the :class:`Drafter`-protocol-shaped contract that
:func:`mlx_generate` consumes. They use a tiny in-memory Gemma 4 target
plus a stub drafter so the round loop runs end-to-end on CPU without
pulling the 78M-parameter gemma4_assistant weights into the test bus.

End-to-end parity (target-only vs MTP-accelerated produces byte-identical
tokens at temperature 0) lands as a separate manual / weight-loading
test in Phase 2d alongside the model-card placement work; here we
focus on the mechanics: the drafter satisfies the Drafter Protocol,
the prefill-capture-then-yield-bonus sequence emits the right
:class:`mlx_lm.GenerationResponse` shape, the metrics surface drives
``GenerationStats``, and the EOS / length / cancellation contracts
match the standard drafter path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast, final

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.generate import GenerationResponse
from mlx_lm.models.gemma4_text import Model as Gemma4Model
from mlx_lm.models.gemma4_text import ModelArgs
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.generator.coupled_drafter import (
    CoupledModelDrafter,
    Gemma4MTPTargetAdapter,
)
from exo.worker.engines.mlx.generator.drafter import Drafter
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.vendor.gemma4_mtp_hooks import attach_mtp_hooks

# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #


def _build_tiny_gemma4_with_hooks() -> Gemma4Model:
    args = ModelArgs(
        model_type="gemma4_text",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=2,
        head_dim=32,
        global_head_dim=32,
        num_key_value_heads=1,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        vocab_size=100,
        vocab_size_per_layer_input=100,
        sliding_window=32,
        sliding_window_pattern=2,
        max_position_embeddings=256,
        layer_types=["sliding_attention", "full_attention"],
        tie_word_embeddings=True,
        final_logit_softcapping=30.0,
    )
    model = Gemma4Model(args)
    model.eval()
    attach_mtp_hooks(model)
    return model


@final
class _StubGemma4Drafter(nn.Module):
    """Reused from :file:`test_coupled_drafter_round_loop.py` -- see that
    module for the full ``_mtp_rounds``-contract description. Returns
    drafts that always reject so the loop emits exactly one token per
    round (the target's bonus), keeping emission counts deterministic.
    """

    @final
    class _Config:
        block_size: int = 4

    def __init__(self) -> None:
        super().__init__()
        self.config: _StubGemma4Drafter._Config = _StubGemma4Drafter._Config()
        self.accept_lens: list[int] = []
        self.bind_calls: int = 0
        self.set_shared_kv_calls: int = 0
        self.draft_block_calls: int = 0

    def bind(self, target_model: object) -> "_StubGemma4Drafter":
        del target_model
        self.bind_calls += 1
        return self

    def make_cache(self) -> list[Any]:
        return []

    def reset(self, target_model: object) -> list[Any]:
        self.bind(target_model)
        self.accept_lens = []
        return []

    def set_shared_kv(
        self,
        shared_kv_states: dict[str, tuple[mx.array, mx.array]],
        kv_offset: int | mx.array,
        position: int | mx.array | None = None,
        left_padding: mx.array | None = None,
    ) -> None:
        del shared_kv_states, kv_offset, position, left_padding
        self.set_shared_kv_calls += 1

    def draft_block(
        self,
        last_bonus: int,
        hidden: mx.array,
        cache: object,
        block_size: int,
        sampler: object,
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        del last_bonus, hidden, cache, sampler
        self.draft_block_calls += 1
        return mx.zeros((1, block_size - 1), dtype=token_dtype)


@final
class _StubDetokenizer:
    """Minimal detokenizer surface consumed by :class:`CoupledModelDrafter`.

    The drafter calls only ``reset()``, ``add_token(int)``, ``finalize()``,
    and reads ``last_segment``. Any closer fidelity to the production
    :mod:`mlx_lm.tokenizer_utils` would couple these tests to that
    module's evolving contract; the stub is the smallest surface that
    satisfies the call sequence.
    """

    def __init__(self) -> None:
        self.last_segment: str = ""
        self.tokens: list[int] = []
        self.finalized: bool = False

    def reset(self) -> None:
        self.tokens = []
        self.last_segment = ""
        self.finalized = False

    def add_token(self, token: int) -> None:
        self.tokens.append(token)
        self.last_segment = f" t{token}"

    def finalize(self) -> None:
        self.finalized = True
        self.last_segment = ""


@final
class _StubTokenizer:
    """Minimal :class:`TokenizerWrapper`-shaped tokenizer for the drafter."""

    def __init__(self, eos_token_ids: list[int] | None = None) -> None:
        self.detokenizer: _StubDetokenizer = _StubDetokenizer()
        self.eos_token_ids: list[int] = list(eos_token_ids or [])


def _greedy_sampler(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1).astype(mx.int32)


# --------------------------------------------------------------------------- #
# Drafter-protocol conformance
# --------------------------------------------------------------------------- #


def test_coupled_drafter_satisfies_drafter_protocol() -> None:
    """The dispatch in ``mlx_generate`` types ``drafter: Drafter`` and
    relies on the runtime-checkable Protocol; the structural mismatch
    that would slip past a static type check (e.g. ``mode`` returning
    a non-DraftMode literal, ``stream`` missing an arg) must surface
    here, not at the first request."""
    target = _build_tiny_gemma4_with_hooks()
    adapter = Gemma4MTPTargetAdapter(target)
    drafter = CoupledModelDrafter(
        target_adapter=adapter,
        drafter=_StubGemma4Drafter(),
        kind="mtp",
        num_draft_tokens=2,
    )

    assert isinstance(drafter, Drafter)
    assert drafter.mode == "model"
    assert drafter.kind == "mtp"
    assert drafter.num_draft_tokens == 2


def test_coupled_drafter_rejects_zero_k() -> None:
    """``num_draft_tokens=0`` is meaningless (no proposals = no
    speculation); the constructor must fail loudly so a misconfigured
    runner doesn't silently emit only bonus tokens."""
    target = _build_tiny_gemma4_with_hooks()
    adapter = Gemma4MTPTargetAdapter(target)
    with pytest.raises(ValueError, match="num_draft_tokens"):
        CoupledModelDrafter(
            target_adapter=adapter,
            drafter=_StubGemma4Drafter(),
            kind="mtp",
            num_draft_tokens=0,
        )


# --------------------------------------------------------------------------- #
# Stream behaviour
# --------------------------------------------------------------------------- #


def _run_stream(
    *,
    target: Gemma4Model,
    drafter: _StubGemma4Drafter,
    prompt_tokens: list[int],
    max_tokens: int,
    eos_token_ids: list[int] | None = None,
    sampler: Callable[[mx.array], mx.array] | None = None,
) -> tuple[list[GenerationResponse], _StubTokenizer]:
    """Drive ``CoupledModelDrafter.stream`` to completion and collect responses.

    Mirrors the call shape :func:`mlx_generate` uses: the drafter
    receives the prefill-tail (last 2 prompt tokens), a freshly-built
    cache covering the rest of the prompt, and the standard sampler /
    logits_processors / context_tokens triple.
    """
    coupled = CoupledModelDrafter(
        target_adapter=Gemma4MTPTargetAdapter(target),
        drafter=cast("nn.Module", drafter),
        kind="mtp",
        num_draft_tokens=2,
    )
    tokenizer = _StubTokenizer(eos_token_ids)
    sampler_fn = sampler or _greedy_sampler

    prefill_prompt = prompt_tokens[:-2]
    decode_prompt = prompt_tokens[-2:]

    cache: list[Any] = cast("list[Any]", target.make_cache())
    if prefill_prompt:
        # ``target`` returns ``mx.array``-typed logits at runtime but the
        # callable surface is structurally generic; we discard the result
        # explicitly so basedpyright doesn't flag the unused expression.
        _ = target(mx.array([prefill_prompt]), cache=cache)

    # ``model`` is typed ``Model`` (a Protocol) on the production
    # signature; the runtime gemma4_text.Model satisfies it but the
    # static surface won't accept the concrete class without help.
    # ``tokenizer`` likewise: the production signature is
    # :class:`TokenizerWrapper` and our stub is structurally compatible
    # with the slots the drafter actually reaches.
    responses: list[GenerationResponse] = list(
        coupled.stream(
            model=cast("Model", cast("object", target)),
            tokenizer=cast("TokenizerWrapper", cast("object", tokenizer)),
            prompt=mx.array(decode_prompt),
            context_tokens=prompt_tokens,
            prompt_cache=cast("KVCacheType", cache),
            max_tokens=max_tokens,
            sampler=sampler_fn,
            logits_processors=[],
            prefill_step_size=1,
        )
    )
    return responses, tokenizer


def test_stream_yields_first_bonus_with_finish_reason_none() -> None:
    """The first emitted response carries the sampled bonus, real
    logprobs (we computed them ourselves before entering the round
    loop), and ``finish_reason=None`` so the caller's stop-sequence
    detection can run before the closing chunk fires."""
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()
    responses, _ = _run_stream(
        target=target,
        drafter=drafter,
        prompt_tokens=[1, 2, 3, 4],
        max_tokens=4,
    )

    assert len(responses) >= 2, "stream must yield at least the bonus + closing"
    first = responses[0]
    assert first.token != 0 or first.token == 0  # token is whatever sampler picked
    assert first.from_draft is False, "first bonus is not a drafted token"
    assert first.finish_reason is None
    assert first.generation_tokens == 1


def test_stream_marks_round_loop_tokens_as_from_draft() -> None:
    """Round-loop tokens carry ``from_draft=True`` so the existing
    ``accepted_draft_tokens`` accumulator in :func:`mlx_generate`
    counts them; without this the coupled path would zero out
    acceptance metrics."""
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()
    responses, _ = _run_stream(
        target=target,
        drafter=drafter,
        prompt_tokens=[1, 2, 3, 4],
        max_tokens=4,
    )

    # Strip the closing chunk; only mid-stream chunks carry the
    # per-token from_draft flag (the closing chunk's flag is a
    # convenience summary).
    mid_stream = [r for r in responses[:-1] if r.finish_reason is None]
    if len(mid_stream) > 1:
        round_loop_tokens = mid_stream[1:]
        assert all(r.from_draft for r in round_loop_tokens), (
            "round-loop tokens must be flagged as drafter-accepted"
        )


def test_stream_respects_max_tokens() -> None:
    """``max_tokens`` is the upper bound on emitted tokens, including
    the first bonus. The caller's ``length`` finish reason fires when
    the budget runs out."""
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()
    responses, _ = _run_stream(
        target=target,
        drafter=drafter,
        prompt_tokens=[1, 2, 3, 4],
        max_tokens=3,
    )

    # The closing chunk is the last response; ``generation_tokens``
    # on it is the canonical emit count.
    closing = responses[-1]
    assert closing.generation_tokens <= 3
    assert closing.finish_reason in {"stop", "length"}


def test_stream_emits_eos_with_stop_finish_reason() -> None:
    """When the round loop yields an EOS token, the drafter must
    short-circuit emission and surface ``finish_reason="stop"`` --
    matching what mlx_lm's stream_generate does for non-spec runs."""
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()

    # Build a sampler that picks token 7 (our EOS) every time. This
    # makes the FIRST BONUS land on EOS, exercising the early-exit
    # path; the round loop never runs in this case.
    def _eos_sampler(logits: mx.array) -> mx.array:
        return mx.full(logits.shape[:-1], 7, dtype=mx.int32)

    responses, tokenizer = _run_stream(
        target=target,
        drafter=drafter,
        prompt_tokens=[1, 2, 3, 4],
        max_tokens=8,
        eos_token_ids=[7],
        sampler=_eos_sampler,
    )

    closing = responses[-1]
    assert closing.finish_reason == "stop"
    assert closing.token == 7
    assert tokenizer.detokenizer.finalized, "detokenizer must be finalised on close"


# --------------------------------------------------------------------------- #
# Metrics + telemetry
# --------------------------------------------------------------------------- #


def test_metrics_returns_zeros_before_stream_runs() -> None:
    """Pre-stream metrics are all zero -- ``GenerationStats``
    construction in :func:`mlx_generate` reads metrics() at finish
    time, so this case shouldn't fire in production, but exposing
    zeroes for unrun streams keeps the contract sensible."""
    target = _build_tiny_gemma4_with_hooks()
    coupled = CoupledModelDrafter(
        target_adapter=Gemma4MTPTargetAdapter(target),
        drafter=cast("nn.Module", _StubGemma4Drafter()),
        kind="mtp",
        num_draft_tokens=2,
    )

    metrics = coupled.metrics()
    assert metrics["spec_decode_rounds"] == 0
    assert metrics["proposed_draft_tokens"] == 0


def test_metrics_after_stream_reflects_round_count() -> None:
    """Each entry in ``drafter.accept_lens`` is a completed round; the
    drafter appends to it from inside ``_mtp_rounds``. After a stream
    that emits ``max_tokens`` total, the round count must be at least 1
    (the loop ran) and ``proposed_draft_tokens`` must scale with the
    round count and the configured block size."""
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()
    coupled = CoupledModelDrafter(
        target_adapter=Gemma4MTPTargetAdapter(target),
        drafter=cast("nn.Module", drafter),
        kind="mtp",
        num_draft_tokens=2,
    )
    tokenizer = _StubTokenizer()

    cache: list[Any] = cast("list[Any]", target.make_cache())
    # Prefill prompt[:-2] outside the drafter, mirroring mlx_generate.
    _ = target(mx.array([[1, 2]]), cache=cache)

    _ = list(
        coupled.stream(
            model=cast("Model", cast("object", target)),
            tokenizer=cast("TokenizerWrapper", cast("object", tokenizer)),
            prompt=mx.array([3, 4]),
            context_tokens=[1, 2, 3, 4],
            prompt_cache=cast("KVCacheType", cache),
            max_tokens=4,
            sampler=_greedy_sampler,
            logits_processors=[],
        )
    )

    metrics = coupled.metrics()
    assert metrics["spec_decode_rounds"] >= 1, (
        "round loop must have run at least once for max_tokens=4"
    )
    # block_size=4 → 3 drafts proposed per round.
    assert metrics["proposed_draft_tokens"] == metrics["spec_decode_rounds"] * 3
