"""Tests for :mod:`pipelined_drafter` and :mod:`drafter_transport`.

The cross-round speculation accounting is the only complex piece, so
these tests focus on:

  * The :class:`DrafterTransport` Protocol contract (any implementation
    that satisfies the Protocol must accept the call sequence the spec
    loop emits).
  * The spec-loop's cache-trim arithmetic for partial accept, full
    accept, speculation hit, and speculation miss -- exercised through
    a deterministic fake transport that records every call so we can
    assert on the trim/forward sequence without spinning up MLX
    weights.
  * Transport-kind parsing (``EXO_DRAFTER_TRANSPORT`` env var).

End-to-end correctness with real MLX weights is exercised by the smoke
+ bench scripts; this file stays MLX-free so it runs in seconds on CI.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Final

import pytest

from exo.worker.engines.mlx.generator.drafter_transport import (
    ALL_TRANSPORT_KINDS,
    EXO_DRAFTER_TRANSPORT_ENV,
    DrafterTransport,
    DraftFuture,
    parse_transport_kind,
    transport_factory_for,
)

# ---------------------------------------------------------------------------
# Test fixtures: deterministic fake transport
# ---------------------------------------------------------------------------


@dataclass
class _Call:
    """One method call against the fake transport, in arrival order."""

    kind: str  # "forward" or "trim"
    inputs: tuple[int, ...] = ()
    num_forwards: int = 0
    n_positions: int = 0


@dataclass
class _ForwardScript:
    """Pre-recorded outputs for the next ``forward`` call."""

    outputs: list[int]


@dataclass
class FakeTransport:
    """A :class:`DrafterTransport` that records calls and returns scripted drafts.

    Used to exercise the spec loop's bookkeeping without running MLX.
    Every ``forward`` consumes one entry from ``script``; if the script
    is exhausted, the test has hit a code path it didn't predict and
    the transport raises (failing the test loudly).
    """

    num_draft_tokens_value: int
    script: list[_ForwardScript] = field(default_factory=list)
    calls: list[_Call] = field(default_factory=list)
    cache_offset: int = 0

    @property
    def num_draft_tokens(self) -> int:
        return self.num_draft_tokens_value

    def forward(self, inputs: list[int], num_forwards: int) -> DraftFuture:
        if not 1 <= num_forwards <= self.num_draft_tokens_value + 1:
            raise ValueError(f"num_forwards out of bounds: {num_forwards}")
        if not 1 <= len(inputs) <= 2:
            raise ValueError(f"inputs length out of bounds: {len(inputs)}")
        if not self.script:
            raise AssertionError(
                "FakeTransport.forward called without script entry; "
                "test missed a code path"
            )
        entry = self.script.pop(0)
        if len(entry.outputs) != num_forwards:
            raise AssertionError(
                f"Script entry has {len(entry.outputs)} outputs; "
                f"forward asked for {num_forwards}"
            )
        self.calls.append(
            _Call(kind="forward", inputs=tuple(inputs), num_forwards=num_forwards)
        )
        # Cache extends by ``len(inputs) + num_forwards - 1`` per spec.
        self.cache_offset += len(inputs) + num_forwards - 1
        future: DraftFuture = Future()
        future.set_result(list(entry.outputs))
        return future

    def trim_cache(self, n_positions: int) -> None:
        if n_positions < 0:
            raise ValueError(f"n_positions must be >= 0, got {n_positions}")
        if n_positions > self.cache_offset:
            raise AssertionError(
                f"Trim {n_positions} would exceed cache offset {self.cache_offset}; "
                "spec loop is over-trimming"
            )
        self.calls.append(_Call(kind="trim", n_positions=n_positions))
        self.cache_offset -= n_positions

    def shutdown(self) -> None:
        return


def test_fake_transport_satisfies_protocol() -> None:
    """The fake transport must structurally satisfy :class:`DrafterTransport`."""
    transport: DrafterTransport = FakeTransport(num_draft_tokens_value=4)
    assert isinstance(transport, DrafterTransport)


# ---------------------------------------------------------------------------
# Transport-kind parsing
# ---------------------------------------------------------------------------


_KIND_DEFAULT: Final[str] = "inprocess"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, _KIND_DEFAULT),
        ("inprocess", "inprocess"),
        ("INPROCESS", "inprocess"),
        ("  inprocess  ", "inprocess"),
        ("remote", "remote"),
        ("Remote", "remote"),
    ],
)
def test_parse_transport_kind_recognised(raw: str | None, expected: str) -> None:
    assert parse_transport_kind(raw, _KIND_DEFAULT) == expected


def test_parse_transport_kind_falls_back_for_unknown() -> None:
    # Unknown kinds warn and fall back to the default rather than
    # raising; that mirrors how ``parse_draft_mode`` handles unknown
    # ``EXO_DRAFT_MODE`` values.
    assert parse_transport_kind("totally-bogus", _KIND_DEFAULT) == _KIND_DEFAULT


def test_all_transport_kinds_match_factory_dispatch() -> None:
    """Every kind in :data:`ALL_TRANSPORT_KINDS` must have a factory.

    The factory may raise ``NotImplementedError`` (Layer B's remote
    transport does), but :func:`transport_factory_for` itself must
    always return a callable -- the dispatch table is part of the
    public contract.
    """
    for kind in ALL_TRANSPORT_KINDS:
        factory = transport_factory_for(kind)
        assert callable(factory)


def test_transport_factory_for_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown drafter transport kind"):
        transport_factory_for("totally-bogus")


# ---------------------------------------------------------------------------
# Spec loop arithmetic via the fake transport
# ---------------------------------------------------------------------------


# These tests exercise the cache-trim arithmetic *as the spec loop
# emits it*, without running the MLX target. We construct call traces
# the loop would produce for a known accept pattern and assert the
# trim/forward sequence matches the formula derived in the
# pipelined_drafter module docstring.
#
# Strategy: don't actually run the spec loop (which needs an MLX
# target). Instead, simulate the spec loop's transport calls
# imperatively for each scenario and assert the cache offset / call
# sequence matches what the docstring promises.


class TestSpecLoopArithmetic:
    """Trace the transport-call sequence for canonical accept patterns."""

    def test_partial_accept_no_speculation(self) -> None:
        """Partial accept (n=2 of K=4): trim K-n-1 = 1, propose [target_correction]."""
        k = 4
        n = 2
        transport = FakeTransport(
            num_draft_tokens_value=k,
            script=[
                # Round 0: 4 drafts.
                _ForwardScript(outputs=[10, 11, 12, 13]),
                # Round 1: 4 drafts after partial-accept setup.
                _ForwardScript(outputs=[20, 21, 22, 23]),
            ],
        )

        # Round 0 propose.
        drafts = transport.forward([1], k).result()
        assert drafts == [10, 11, 12, 13]
        assert transport.cache_offset == k  # 4 positions

        # Spec loop: partial accept after target verify (n=2, drafts[2] mismatched).
        # Transport bookkeeping for next round:
        #   * trim k - n - 1 = 1 position
        #   * propose [target_correction] (length 1), k outputs
        transport.trim_cache(k - n - 1)
        assert transport.cache_offset == k - 1  # 3 positions

        # Next round propose with length-1 input.
        next_drafts = transport.forward([99], k).result()
        assert next_drafts == [20, 21, 22, 23]
        # Cache extends by k (length-1 input + k-1 length-1 forwards = k).
        assert transport.cache_offset == k - 1 + k  # 7 positions

        # Verify call trace.
        assert [c.kind for c in transport.calls] == [
            "forward",
            "trim",
            "forward",
        ]
        assert transport.calls[1].n_positions == 1

    def test_full_accept_no_speculation(self) -> None:
        """Full accept (n=k): no trim; next round propose has length-2 input."""
        k = 4
        transport = FakeTransport(
            num_draft_tokens_value=k,
            script=[
                _ForwardScript(outputs=[10, 11, 12, 13]),
                _ForwardScript(outputs=[20, 21, 22, 23]),
            ],
        )

        transport.forward([1], k).result()
        assert transport.cache_offset == k

        # Full accept: no trim. Next round propose with [drafts[-1], bonus].
        next_drafts = transport.forward([13, 99], k).result()
        assert next_drafts == [20, 21, 22, 23]
        # Cache extends by k + 1 (length-2 input + k-1 length-1 forwards).
        assert transport.cache_offset == k + (k + 1)

        assert [c.kind for c in transport.calls] == ["forward", "forward"]
        assert transport.calls[1].inputs == (13, 99)
        assert transport.calls[1].num_forwards == k

    def test_speculation_hit(self) -> None:
        """Full accept + speculation hit: round t+1 drafts come for free."""
        k = 4
        transport = FakeTransport(
            num_draft_tokens_value=k,
            script=[
                # Round 0 propose: [10, 11, 12, 13].
                _ForwardScript(outputs=[10, 11, 12, 13]),
                # Speculative round (input=[13], k+1 outputs):
                # outputs[0] = drafter's bonus prediction; outputs[1..k] = round
                # 1's drafts.
                _ForwardScript(outputs=[99, 30, 31, 32, 33]),
            ],
        )

        # Round 0 propose.
        round0_drafts = transport.forward([1], k).result()
        assert round0_drafts == [10, 11, 12, 13]

        # Speculative call.
        spec_outputs = transport.forward([13], k + 1).result()
        assert spec_outputs == [99, 30, 31, 32, 33]
        # After speculation: cache extended by k (round 0) + (k + 1)
        # (speculation) = 2k+1 positions.
        assert transport.cache_offset == k + (k + 1)

        # Speculation hit: target's bonus_t == 99 == spec_outputs[0].
        # Round 1's drafts = spec_outputs[1:k+1].
        round1_drafts = spec_outputs[1 : k + 1]
        assert round1_drafts == [30, 31, 32, 33]

        # No additional transport calls (drafter cache state already
        # correct for round 1).
        assert [c.kind for c in transport.calls] == ["forward", "forward"]

    def test_speculation_miss_full_accept(self) -> None:
        """Full accept but bonus mismatched: rollback k+1, length-2 propose."""
        k = 4
        transport = FakeTransport(
            num_draft_tokens_value=k,
            script=[
                _ForwardScript(outputs=[10, 11, 12, 13]),
                _ForwardScript(outputs=[88, 80, 81, 82, 83]),  # speculative
                _ForwardScript(outputs=[40, 41, 42, 43]),  # round 1 standard
            ],
        )

        transport.forward([1], k).result()
        spec_outputs = transport.forward([13], k + 1).result()
        # bonus_t = 99 (target), spec_outputs[0] = 88 -> miss.

        # Rollback the k+1 speculative positions.
        transport.trim_cache(k + 1)
        assert transport.cache_offset == k  # back to round-0 state

        # Standard length-2-seed propose for round 1: [drafts[-1], bonus_t].
        round1_drafts = transport.forward([13, 99], k).result()
        assert round1_drafts == [40, 41, 42, 43]

        del spec_outputs
        kinds = [c.kind for c in transport.calls]
        assert kinds == ["forward", "forward", "trim", "forward"]
        assert transport.calls[2].n_positions == k + 1
        assert transport.calls[3].inputs == (13, 99)

    def test_speculation_miss_partial_accept(self) -> None:
        """Partial accept with speculation in flight: rollback k+1 + partial trim."""
        k = 4
        n = 2
        transport = FakeTransport(
            num_draft_tokens_value=k,
            script=[
                _ForwardScript(outputs=[10, 11, 12, 13]),
                _ForwardScript(outputs=[88, 80, 81, 82, 83]),  # speculative
                _ForwardScript(outputs=[50, 51, 52, 53]),  # round 1
            ],
        )

        transport.forward([1], k).result()
        transport.forward([13], k + 1).result()
        # cache offset: k + (k + 1) = 2k + 1 = 9

        # Partial accept at round 0: speculation is invalid AND partial
        # trim is needed. The combined trim is (k + 1) + (k - n - 1).
        combined_trim = (k + 1) + (k - n - 1)
        transport.trim_cache(combined_trim)
        # cache offset: 2k + 1 - combined_trim = n + 1 = 3
        assert transport.cache_offset == n + 1

        # Round 1 standard propose with length-1 input.
        round1_drafts = transport.forward([99], k).result()
        assert round1_drafts == [50, 51, 52, 53]

        kinds = [c.kind for c in transport.calls]
        assert kinds == ["forward", "forward", "trim", "forward"]
        assert transport.calls[2].n_positions == combined_trim


# ---------------------------------------------------------------------------
# PipelinedModelDrafter wiring
# ---------------------------------------------------------------------------


def test_pipelined_drafter_mode_is_pipelined() -> None:
    # Imported lazily so this file stays importable without the drafter
    # module's MLX-bound siblings; the import itself is what we're
    # exercising (catches accidental syntax errors in pipelined_drafter
    # that the type checker might miss for runtime-only paths).
    from exo.worker.engines.mlx.generator.pipelined_drafter import (
        PipelinedModelDrafter,
    )

    transport = FakeTransport(num_draft_tokens_value=4)
    drafter = PipelinedModelDrafter(transport=transport, num_draft_tokens=4)
    assert drafter.mode == "pipelined"
    assert drafter.num_draft_tokens == 4


def test_pipelined_drafter_validates_num_draft_tokens() -> None:
    from exo.worker.engines.mlx.generator.pipelined_drafter import (
        PipelinedModelDrafter,
    )

    transport = FakeTransport(num_draft_tokens_value=4)
    with pytest.raises(ValueError, match="num_draft_tokens"):
        PipelinedModelDrafter(transport=transport, num_draft_tokens=0)
    with pytest.raises(ValueError, match="exceeds transport's max"):
        PipelinedModelDrafter(transport=transport, num_draft_tokens=10)


def test_pipelined_drafter_shutdown_delegates() -> None:
    """Shutdown should propagate to the transport so remote serve loops drain cleanly."""
    from exo.worker.engines.mlx.generator.pipelined_drafter import (
        PipelinedModelDrafter,
    )

    shutdown_calls: list[None] = []

    class _ShutdownRecorder(FakeTransport):
        def shutdown(self) -> None:
            shutdown_calls.append(None)

    transport = _ShutdownRecorder(num_draft_tokens_value=4)
    drafter = PipelinedModelDrafter(transport=transport, num_draft_tokens=4)
    drafter.shutdown()
    assert len(shutdown_calls) == 1


# ---------------------------------------------------------------------------
# Transport-kind environment plumbing
# ---------------------------------------------------------------------------


def test_make_drafter_pipelined_resolves_transport_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``make_drafter("pipelined", ...)`` should consult ``EXO_DRAFTER_TRANSPORT``.

    Without a real MLX drafter model we can't construct an in-process
    transport here; we just assert the env var is read and that the
    dispatch reaches :func:`transport_factory_for` (which raises a
    ``ValueError`` for unknown kinds).
    """
    from exo.worker.engines.mlx.generator.drafter import make_drafter

    monkeypatch.setenv(EXO_DRAFTER_TRANSPORT_ENV, "totally-bogus")
    # Bogus kind warns and falls back to ``inprocess``; ``make_drafter``
    # then fails because we passed no model/cache. The error message
    # should reference both ``pipelined`` and the missing pieces.
    with pytest.raises(ValueError, match="pipelined.*inprocess"):
        make_drafter(
            mode="pipelined",
            num_draft_tokens=4,
            draft_model=None,
            draft_cache=None,
        )
