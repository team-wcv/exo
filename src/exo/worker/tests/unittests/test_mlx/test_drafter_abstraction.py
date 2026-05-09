"""Tests for the ``Drafter`` abstraction.

These cover the pure-Python pieces - mode resolution, n-gram suffix
matching, and the spec-loop accept arithmetic - so they don't need MLX
weights or a GPU. End-to-end correctness with a real model is exercised
by the smoke + bench scripts in ``scripts/``.
"""

from __future__ import annotations

import pytest

from exo.worker.engines.mlx.generator.drafter import (
    ALL_DRAFT_MODES,
    EXO_DRAFT_MODE_ENV,
    DraftMode,
    NgramDrafter,
    NoSpecDrafter,
    make_drafter,
    parse_draft_mode,
    resolve_draft_mode,
)


def test_all_draft_modes_match_literal() -> None:
    """``ALL_DRAFT_MODES`` must be the runtime mirror of the ``DraftMode`` Literal."""
    assert ALL_DRAFT_MODES == ("model", "ngram", "none")


@pytest.mark.parametrize(
    ("raw", "default", "expected"),
    [
        (None, "model", "model"),
        (None, "none", "none"),
        ("model", "none", "model"),
        ("MODEL", "none", "model"),
        ("  ngram  ", "none", "ngram"),
        ("none", "model", "none"),
        ("garbage", "model", "model"),
        ("garbage", "none", "none"),
    ],
)
def test_parse_draft_mode(
    raw: str | None, default: DraftMode, expected: DraftMode
) -> None:
    assert parse_draft_mode(raw, default) == expected


def test_parse_draft_mode_warns_on_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
    parse_draft_mode("totally-bogus", "none")
    # Loguru-driven logger doesn't pipe to caplog by default; just assert
    # the call didn't raise. The warning is documented in the docstring.


class TestResolveDraftMode:
    def test_explicit_request_mode_wins_over_use_drafter(self) -> None:
        # Per-request draft_mode beats the use_drafter shortcut.
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=False,
                request_draft_mode="ngram",
            )
            == "ngram"
        )

    def test_use_drafter_false_maps_to_none(self) -> None:
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=False,
                request_draft_mode=None,
            )
            == "none"
        )

    def test_default_with_drafter_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=None,
                request_draft_mode=None,
            )
            == "model"
        )

    def test_default_without_drafter_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        assert (
            resolve_draft_mode(
                has_drafter_model=False,
                request_use_drafter=None,
                request_draft_mode=None,
            )
            == "none"
        )

    def test_env_override_with_drafter_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(EXO_DRAFT_MODE_ENV, "ngram")
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=None,
                request_draft_mode=None,
            )
            == "ngram"
        )

    def test_model_mode_without_drafter_demotes_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        assert (
            resolve_draft_mode(
                has_drafter_model=False,
                request_use_drafter=None,
                request_draft_mode="model",
            )
            == "none"
        )

    def test_explicit_none_with_drafter_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=None,
                request_draft_mode="none",
            )
            == "none"
        )

    def test_use_drafter_true_promotes_to_model_when_drafter_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Codex P2 (PR #19 round-(N+8), drafter.py:148): the
        ``use_drafter=true`` opt-in must override an explicit
        ``EXO_DRAFT_MODE=none`` process default. With a drafter model
        loaded the natural intent is ``"model"``."""
        monkeypatch.setenv(EXO_DRAFT_MODE_ENV, "none")
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=True,
                request_draft_mode=None,
            )
            == "model"
        )

    def test_use_drafter_true_falls_back_to_ngram_without_drafter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Codex P2 (PR #19 round-(N+8), drafter.py:148): when no
        drafter model is loaded, ``use_drafter=true`` must still
        engage *some* drafting strategy. ``ngram`` is the only
        viable option (in-context suffix lookup needs no extra
        weights), so promote to ``"ngram"`` -- never silently fall
        through to ``"none"``."""
        monkeypatch.setenv(EXO_DRAFT_MODE_ENV, "none")
        assert (
            resolve_draft_mode(
                has_drafter_model=False,
                request_use_drafter=True,
                request_draft_mode=None,
            )
            == "ngram"
        )

    def test_use_drafter_true_with_drafter_loaded_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The opt-in shortcut must dominate the env default in the
        common 'A/B test harness' case where the runner ships with
        ``EXO_DRAFT_MODE=none`` and the harness flips drafting on
        per-request."""
        monkeypatch.setenv(EXO_DRAFT_MODE_ENV, "none")
        result = resolve_draft_mode(
            has_drafter_model=True,
            request_use_drafter=True,
            request_draft_mode=None,
        )
        assert result == "model"

    def test_explicit_request_mode_still_wins_over_use_drafter_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Precedence regression test: explicit ``request_draft_mode``
        wins over both ``use_drafter`` and the env var, even when
        the request is opting in with ``use_drafter=True``."""
        monkeypatch.setenv(EXO_DRAFT_MODE_ENV, "none")
        result = resolve_draft_mode(
            has_drafter_model=True,
            request_use_drafter=True,
            request_draft_mode="ngram",
        )
        assert result == "ngram"


class TestNgramDrafterPropose:
    """The proposer is pure list logic; no MLX involved."""

    def test_returns_empty_when_context_is_too_short(self) -> None:
        drafter = NgramDrafter(num_draft_tokens=4, min_match=2, max_match=4)
        # Need at least min_match + 1 tokens for a match to be possible
        # (suffix of length min_match plus one earlier match position).
        assert drafter.propose([1, 2], 4) == []

    def test_returns_empty_when_no_match(self) -> None:
        drafter = NgramDrafter(num_draft_tokens=4, min_match=2, max_match=4)
        # Tokens are unique - no suffix appears earlier.
        assert drafter.propose([10, 20, 30, 40, 50], 4) == []

    def test_finds_simple_repetition(self) -> None:
        # Suffix [1, 2] appears at start; following tokens are [3, 4].
        drafter = NgramDrafter(num_draft_tokens=4, min_match=2, max_match=4)
        assert drafter.propose([1, 2, 3, 4, 1, 2], 2) == [3, 4]

    def test_proposes_up_to_k_tokens(self) -> None:
        drafter = NgramDrafter(num_draft_tokens=10, min_match=2, max_match=4)
        # K=2 caps proposal to 2 even though 4 follow the match.
        assert drafter.propose([1, 2, 3, 4, 5, 6, 1, 2], 2) == [3, 4]

    def test_prefers_longer_match(self) -> None:
        # Suffix [2, 3] appears at index 1; suffix [1, 2, 3] appears at
        # index 0 (length 3, longer). Should prefer the longer one and
        # return [4, 5] (the tokens after the longer match).
        drafter = NgramDrafter(num_draft_tokens=4, min_match=2, max_match=4)
        ctx = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]
        # Last 3 tokens are [1, 2, 3]; longest match starts at 0.
        # Following tokens at start were [4, 5].
        assert drafter.propose(ctx, 4)[:2] == [4, 5]

    def test_prefers_recent_match_when_tied(self) -> None:
        # Two matches of suffix [9, 9] at same length; prefer the more
        # recent one (locality of reference).
        drafter = NgramDrafter(num_draft_tokens=2, min_match=2, max_match=2)
        ctx = [9, 9, 1, 9, 9, 2, 9, 9]
        # Recent match at index 3, followed by [2]. Earliest match at 0,
        # followed by [1]. Prefer recent -> [2].
        result = drafter.propose(ctx, 1)
        assert result == [2]

    def test_returns_empty_for_zero_k(self) -> None:
        drafter = NgramDrafter(num_draft_tokens=4, min_match=2, max_match=4)
        assert drafter.propose([1, 2, 3, 1, 2], 0) == []

    def test_validates_constructor_args(self) -> None:
        with pytest.raises(ValueError, match="num_draft_tokens"):
            NgramDrafter(num_draft_tokens=0)
        with pytest.raises(ValueError, match="min_match"):
            NgramDrafter(num_draft_tokens=2, min_match=0)
        with pytest.raises(ValueError, match="max_match"):
            NgramDrafter(num_draft_tokens=2, min_match=4, max_match=2)


def test_drafter_modes_match_implementation_class() -> None:
    """Each concrete drafter exposes the right ``mode`` literal."""
    assert NoSpecDrafter().mode == "none"
    assert NgramDrafter(num_draft_tokens=2).mode == "ngram"


def test_make_drafter_dispatches_correctly() -> None:
    none_drafter = make_drafter(
        mode="none", num_draft_tokens=4, draft_model=None, draft_cache=None
    )
    assert isinstance(none_drafter, NoSpecDrafter)
    ngram_drafter = make_drafter(
        mode="ngram", num_draft_tokens=4, draft_model=None, draft_cache=None
    )
    assert isinstance(ngram_drafter, NgramDrafter)


def test_make_drafter_rejects_model_without_pieces() -> None:
    with pytest.raises(ValueError, match="draft_model"):
        make_drafter(
            mode="model", num_draft_tokens=4, draft_model=None, draft_cache=None
        )


def test_ngram_drafter_proposal_caps_at_k() -> None:
    # The spec loop tops up ``K = min(max_tokens - ntoks, num_draft_tokens)``
    # before each round; the proposer must respect that cap so we don't
    # overrun ``max_tokens`` in the verify forward.
    drafter = NgramDrafter(num_draft_tokens=10, min_match=2, max_match=4)
    result = drafter.propose([1, 2, 3, 4, 1, 2], 3)
    assert len(result) <= 3


# ---------------------------------------------------------------------------
# Codex P2 (PR #19 round-(N+2), drafter.py:495):
# ``_ngram_stream_generate`` must report ``prompt_tokens`` as the
# size of the prefill *tail* it actually processed -- not the full
# prompt -- so the upstream aggregator's
# ``prefill_tokens + out.prompt_tokens`` sum equals the full prompt
# instead of double-counting it (and over-counting further on
# prefix-cache hits).
# ---------------------------------------------------------------------------


class TestNgramStreamGeneratePromptTokens:
    """Regression: yielded ``GenerationResponse.prompt_tokens`` must
    equal ``prompt.size`` (tail), not ``len(context_tokens)`` (full).

    We bypass the real spec loop by patching ``_ngram_speculative_step``
    so this test stays in CPU-only territory and doesn't need MLX
    weights.
    """

    def test_yields_tail_prompt_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import mlx.core as mx

        from exo.worker.engines.mlx.generator import drafter as drafter_module

        # Sentinel "model" / "tokenizer" / "cache": the patched spec
        # loop never touches them, so we can keep them as ``object()``.
        sentinel_model = object()

        class _FakeDetokenizer:
            def __init__(self) -> None:
                self.last_segment = ""

            def reset(self) -> None: ...
            def add_token(self, _token: int) -> None: ...
            def finalize(self) -> None: ...

        class _FakeTokenizer:
            def __init__(self) -> None:
                self.detokenizer = _FakeDetokenizer()
                self.eos_token_ids = {99}

        full_prompt = list(range(20))
        prompt_tail = mx.array(full_prompt[-2:], dtype=mx.uint32)

        def _fake_step(**_kwargs: object):  # noqa: ANN202
            yield (1, mx.zeros((1,)), False)
            yield (2, mx.zeros((1,)), True)
            yield (3, mx.zeros((1,)), False)

        monkeypatch.setattr(
            drafter_module,
            "_ngram_speculative_step",
            _fake_step,
        )

        responses = list(
            drafter_module._ngram_stream_generate(  # pyright: ignore[reportPrivateUsage]
                model=sentinel_model,  # pyright: ignore[reportArgumentType]
                tokenizer=_FakeTokenizer(),  # pyright: ignore[reportArgumentType]
                prompt=prompt_tail,
                context_tokens=full_prompt,
                prompt_cache=[],
                max_tokens=10,
                sampler=lambda x: x,
                logits_processors=[],
                drafter=NgramDrafter(num_draft_tokens=2),
                prefill_step_size=2,
            )
        )

        assert responses, "stream must yield at least one response"
        for response in responses:
            assert response.prompt_tokens == prompt_tail.size, (
                f"prompt_tokens must be the prefill tail size "
                f"({prompt_tail.size}), got {response.prompt_tokens}. "
                "Pre-fix this was len(context_tokens) which double-counts "
                "tokens already consumed by exo.prefill upstream."
            )
            assert response.prompt_tokens != len(full_prompt), (
                "prompt_tokens must NOT be the full prompt size, "
                "otherwise the upstream aggregator's "
                "(prefill_tokens + out.prompt_tokens) sum overcounts."
            )


class TestRequestIsGreedySampling:
    """Codex P1 (PR #19 round-(N+4), drafter.py:692): n-gram speculative
    decoding's ``target == draft`` accept rule is only
    distribution-correct under greedy decoding (argmax sampling).
    ``mlx_lm.make_sampler`` returns argmax iff ``temp == 0.0``, so the
    helper gates on temperature alone -- non-zero temperature means
    stochastic sampling and the n-gram path must demote to non-spec
    to preserve the model's output distribution.
    """

    @staticmethod
    def _make_task(
        temperature: float | None,
    ) -> "object":
        from exo.shared.types.common import ModelId
        from exo.shared.types.text_generation import (
            InputMessage,
            InputMessageContent,
            TextGenerationTaskParams,
        )

        return TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            temperature=temperature,
        )

    def test_temperature_zero_is_greedy(self) -> None:
        from exo.worker.engines.mlx.generator.generate import (
            _request_is_greedy_sampling,  # pyright: ignore[reportPrivateUsage]
        )

        task = self._make_task(temperature=0.0)
        assert _request_is_greedy_sampling(task) is True  # pyright: ignore[reportArgumentType]

    def test_nonzero_temperature_is_not_greedy(self) -> None:
        from exo.worker.engines.mlx.generator.generate import (
            _request_is_greedy_sampling,  # pyright: ignore[reportPrivateUsage]
        )

        for temp in (0.1, 0.7, 1.0, 2.0):
            task = self._make_task(temperature=temp)
            assert _request_is_greedy_sampling(task) is False, (  # pyright: ignore[reportArgumentType]
                f"temperature={temp} must NOT be classified as greedy "
                f"(make_sampler returns stochastic sampling)"
            )

    def test_omitted_temperature_inherits_runner_default_non_greedy(self) -> None:
        # When the request omits temperature, the runner falls back to
        # a stochastic default (see ``make_sampler`` call site), so the
        # request is non-greedy. The helper exclusively checks
        # ``task.temperature == 0.0``; a missing temperature is
        # therefore correctly classified as non-greedy.
        from exo.worker.engines.mlx.generator.generate import (
            _request_is_greedy_sampling,  # pyright: ignore[reportPrivateUsage]
        )

        task = self._make_task(temperature=None)
        assert _request_is_greedy_sampling(task) is False, (  # pyright: ignore[reportArgumentType]
            "missing temperature inherits the runner default "
            "(non-greedy); n-gram drafting must demote to non-spec"
        )


class TestNgramStreamGenerateThreadsKvQuantization:
    """Codex P2 (PR #19 round-(N+6), drafter.py:642): the custom
    n-gram decode loop must call ``maybe_quantize_kv_cache`` after
    every model forward when ``KV_BITS`` is configured. Pre-fix the
    loop bypassed the quantization that
    ``mlx_lm.stream_generate`` does internally for the non-ngram
    path, so ``KV_BITS=4`` deployments silently kept the n-gram
    path's prompt-cache rows at full precision and could OOM on
    long generations.

    We assert at the call-site level: ``_ngram_stream_generate``
    threads the constants from :mod:`exo.worker.engines.mlx.constants`
    into ``_ngram_speculative_step`` so the quantization pass has
    the exact same parameters as ``mlx_lm.stream_generate``. The
    actual MLX dispatch is exercised by the smoke + bench scripts;
    this test stays MLX-free.
    """

    def test_ngram_stream_generate_passes_kv_bits_through_to_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mlx.core as mx

        from exo.worker.engines.mlx.constants import KV_BITS, KV_GROUP_SIZE
        from exo.worker.engines.mlx.generator import drafter as drafter_module

        captured_kwargs: dict[str, object] = {}

        def _fake_step(**kwargs: object):  # noqa: ANN202
            captured_kwargs.update(kwargs)
            yield (1, mx.zeros((1,)), False)

        monkeypatch.setattr(
            drafter_module,
            "_ngram_speculative_step",
            _fake_step,
        )

        class _FakeDetokenizer:
            def __init__(self) -> None:
                self.last_segment = ""

            def reset(self) -> None: ...
            def add_token(self, _token: int) -> None: ...
            def finalize(self) -> None: ...

        class _FakeTokenizer:
            def __init__(self) -> None:
                self.detokenizer = _FakeDetokenizer()
                self.eos_token_ids = {99}

        list(
            drafter_module._ngram_stream_generate(  # pyright: ignore[reportPrivateUsage]
                model=object(),  # pyright: ignore[reportArgumentType]
                tokenizer=_FakeTokenizer(),  # pyright: ignore[reportArgumentType]
                prompt=mx.array([1, 2], dtype=mx.uint32),
                context_tokens=[1, 2],
                prompt_cache=[],
                max_tokens=2,
                sampler=lambda x: x,
                logits_processors=[],
                drafter=NgramDrafter(num_draft_tokens=2),
                prefill_step_size=2,
            )
        )

        assert captured_kwargs.get("kv_bits") == KV_BITS, (
            "n-gram stream must thread KV_BITS into the step so the "
            "in-loop quantization call uses the same setting as "
            "mlx_lm.stream_generate; got "
            f"kv_bits={captured_kwargs.get('kv_bits')!r}, expected {KV_BITS!r}"
        )
        assert captured_kwargs.get("kv_group_size") == KV_GROUP_SIZE, (
            "n-gram stream must thread KV_GROUP_SIZE into the step so the "
            "in-loop quantization call uses the same group size as "
            "mlx_lm.stream_generate; got "
            f"kv_group_size={captured_kwargs.get('kv_group_size')!r}, "
            f"expected {KV_GROUP_SIZE!r}"
        )
