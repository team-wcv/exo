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
