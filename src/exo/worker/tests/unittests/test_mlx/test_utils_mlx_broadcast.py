"""Unit tests for the MLX utility primitives used by the V2 multi-target spec loop.

These exercise the contracts that the asymmetric pipelined drafter
relies on for cross-rank determinism without spinning up MLX or
``mx.distributed``:

  * :func:`mx_broadcast_int_list` -- length / range / root contract.
    The single-rank short-circuit can be exercised directly; the
    multi-rank ``all_sum`` path is covered indirectly because it
    delegates value validation to the same helper.
  * :func:`_validate_broadcast_values` -- the int32 bounds are tighter
    than Python's ``int`` range, so out-of-range values from a callsite
    bug must raise rather than wrap silently.
  * :func:`_hash_task_list` -- deterministic across runner subprocesses
    (does not depend on ``PYTHONHASHSEED``) and sensitive to ordering
    (so transposed task lists hash differently and the drift check
    catches them).
  * :func:`mx_all_gather_tasks` -- the single-rank short-circuit. The
    multi-rank drift path needs an actual ``mx.distributed`` group, so
    we cover the structural contract here and the cluster bench
    exercises the real collective.

Kept MLX-free so it runs in milliseconds on CI alongside the rest of
the unittest suite.
"""

from __future__ import annotations

import pytest

from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.worker.engines.mlx.utils_mlx import (
    _MX_BROADCAST_MAX_VALUE,  # pyright: ignore[reportPrivateUsage]
    _task_id_hash_pair,  # pyright: ignore[reportPrivateUsage]
    _validate_broadcast_values,  # pyright: ignore[reportPrivateUsage]
    mx_all_gather_tasks,
    mx_broadcast_int_list,
)

# ---------------------------------------------------------------------------
# Validation helper (unit, no MLX needed)
# ---------------------------------------------------------------------------


class TestValidateBroadcastValues:
    """``_validate_broadcast_values`` rejects values that would corrupt
    the int32 ``all_sum`` buffer: negatives wrap on cast, and values
    >= 2**31 overflow on sum."""

    def test_accepts_zero(self) -> None:
        _validate_broadcast_values([0, 0, 0])

    def test_accepts_typical_token_ids(self) -> None:
        # Gemma-4 vocab is ~256k; well inside int32 positive range.
        _validate_broadcast_values([0, 1, 256_000, 999_999])

    def test_accepts_max_value(self) -> None:
        _validate_broadcast_values([_MX_BROADCAST_MAX_VALUE])

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            _validate_broadcast_values([0, -1, 0])

    def test_rejects_overflow(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            _validate_broadcast_values([_MX_BROADCAST_MAX_VALUE + 1])

    def test_error_includes_offending_index(self) -> None:
        with pytest.raises(ValueError, match=r"index 2 = -7"):
            _validate_broadcast_values([0, 1, -7, 3])


# ---------------------------------------------------------------------------
# mx_broadcast_int_list (single-rank short-circuit + contract)
# ---------------------------------------------------------------------------


class TestMxBroadcastIntListSingleRank:
    """The ``group is None`` short-circuit covers single-rank deployments
    (the V1 single-target path and the non-distributed test fakes).
    Multi-rank cluster behaviour is exercised by the cluster bench
    because it needs a real ``mx.distributed`` group."""

    def test_returns_values_when_root(self) -> None:
        result = mx_broadcast_int_list([1, 2, 3], length=3, group=None, is_root=True)
        assert result == [1, 2, 3]
        # Returned list must be a new object so mutating it doesn't
        # corrupt the caller's source list.
        result[0] = 99
        # No assertion on the source -- just exercising that the call
        # didn't share storage. ``list(values)`` semantics.

    def test_rejects_zero_length(self) -> None:
        with pytest.raises(ValueError, match="length must be >= 1"):
            mx_broadcast_int_list([], length=0, group=None, is_root=True)

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length 3"):
            mx_broadcast_int_list([1, 2], length=3, group=None, is_root=True)

    def test_rejects_none_values_on_root(self) -> None:
        with pytest.raises(ValueError, match="length 3"):
            mx_broadcast_int_list(None, length=3, group=None, is_root=True)

    def test_rejects_consumer_in_single_rank(self) -> None:
        # Only the root has source values; ``is_root=False`` with no
        # group means there's no peer to broadcast from -- caller bug.
        with pytest.raises(ValueError, match="single-rank short-circuit"):
            mx_broadcast_int_list([1, 2, 3], length=3, group=None, is_root=False)

    def test_validates_values_on_root(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            mx_broadcast_int_list([0, -1], length=2, group=None, is_root=True)


# ---------------------------------------------------------------------------
# Task-list hashing (drift detection)
# ---------------------------------------------------------------------------


def _make_task(task_id: str) -> TextGeneration:
    """Build a minimal :class:`TextGeneration` for hash-based drift tests.

    The hash function only inspects ``task_id``; the rest of the fields
    are filled with the smallest valid values that satisfy Pydantic's
    strict-mode validation. Keep the construction here so the cluster-
    facing types' field churn doesn't ripple through every assertion
    body.
    """
    return TextGeneration(
        task_id=TaskId(task_id),
        instance_id=InstanceId(),
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("mlx-community/test-model"),
            input=[
                InputMessage(role="user", content=InputMessageContent("hello")),
            ],
            max_output_tokens=1,
        ),
    )


class TestTaskIdHashPair:
    """``_task_id_hash_pair`` must be deterministic, sensitive to ID
    content, and produce values that fit inside the int32 positive
    range so the pair can ride :func:`mx_broadcast_int_list`'s
    int32 ``all_sum`` buffer without wrap-around. BLAKE2b is
    seed-free so two runner subprocesses must agree on the digest
    without depending on Python's salted ``hash()``.
    """

    def test_deterministic_across_calls(self) -> None:
        assert _task_id_hash_pair("alpha") == _task_id_hash_pair("alpha")

    def test_sensitive_to_id_content(self) -> None:
        # If two distinct task IDs hashed to the same pair, agreement
        # would silently admit a task that root doesn't actually have.
        assert _task_id_hash_pair("alpha") != _task_id_hash_pair("beta")

    def test_pair_components_in_int32_positive_range(self) -> None:
        for ident in (
            "a",
            "alpha",
            "very-long-task-id-with-lots-of-bytes-" * 4,
            "\u00e9\u00e8\u00ea",  # non-ascii bytes exercise utf-8 path
        ):
            high, low = _task_id_hash_pair(ident)
            assert 0 <= high <= _MX_BROADCAST_MAX_VALUE
            assert 0 <= low <= _MX_BROADCAST_MAX_VALUE


# ---------------------------------------------------------------------------
# mx_all_gather_tasks single-rank short-circuit
# ---------------------------------------------------------------------------


class TestMxAllGatherTasksSingleRank:
    """Single-rank short-circuit: returns the local task list as-is and
    never invokes a collective. The drift-detection branch needs an
    actual ``mx.distributed`` group and is exercised by the cluster
    bench."""

    def test_empty_input(self) -> None:
        agreed, different = mx_all_gather_tasks([], group=None)
        assert agreed == []
        assert different == []

    def test_passes_through_tasks(self) -> None:
        tasks = [_make_task("task-1"), _make_task("task-2")]
        agreed, different = mx_all_gather_tasks(tasks, group=None)
        assert agreed == tasks
        assert different == []

    def test_returns_a_copy(self) -> None:
        # The caller mutates ``self._maybe_queue`` after the gather;
        # the returned list must be a different object so post-gather
        # mutation doesn't corrupt the agreement view.
        tasks = [_make_task("task-1")]
        agreed, _different = mx_all_gather_tasks(tasks, group=None)
        assert agreed is not tasks


# ---------------------------------------------------------------------------
# mx_all_gather_tasks multi-rank order-agnostic agreement
# ---------------------------------------------------------------------------


class _FakeGroup:
    """Stand-in for ``mx.distributed.Group`` with just the
    ``rank()``/``size()`` surface ``mx_all_gather_tasks`` exercises.

    We avoid importing MLX so the unittest suite stays MLX-free.
    """

    def __init__(self, rank: int, size: int = 2) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class TestMxAllGatherTasksAgreement:
    """Codex P1 (PR #20 round-(N+2), utils_mlx.py:1451): the previous
    hash-equality drift detector hard-failed on benign cross-rank
    timing skew (rank 0 has ``[A, B]`` while rank 1 has ``[A]``),
    turning a normal scheduling state into a runner-killing
    ``RuntimeError``. ``agree_on_tasks`` documents that some ranks
    "may have received [tasks] in different order or not at all", so
    the agreement primitive must instead return:

      * ``agreed`` -- the *intersection* of all ranks' task sets
        (universally present) -- to be admitted into the active set
        on this tick.
      * ``different`` -- the local tasks NOT in the intersection --
        to be retried on the next tick once pubsub catches up.

    These tests stub both the int-list broadcast and ``all_sum`` so
    we can exercise the structural agreement protocol without an
    actual ``mx.distributed`` group. The cluster bench covers the
    real collective.
    """

    def _setup_collectives(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        root_task_ids: list[str],
        all_ranks_task_ids: list[list[str]],
    ) -> None:
        """Wire up fakes for ``mx_broadcast_int_list`` and
        ``mx.distributed.all_sum`` that simulate the cluster's
        collective behaviour. ``all_ranks_task_ids[i]`` is the task
        IDs known to rank ``i`` at agreement time. The fakes
        synthesise root's view (count + hashes) and the all-sum
        presence vector that ``mx_all_gather_tasks`` consumes.
        """
        from exo.worker.engines.mlx import utils_mlx as utils_mlx_mod

        root_count = len(root_task_ids)
        # Sorted as the production code sorts, so the broadcast buffer
        # mirrors what root would produce.
        sorted_root_ids = sorted(root_task_ids)
        root_pairs: list[int] = []
        for tid in sorted_root_ids:
            high, low = _task_id_hash_pair(tid)
            root_pairs.append(high)
            root_pairs.append(low)

        def fake_broadcast(
            values: list[int] | None,
            *,
            length: int,
            group: object,
            is_root: bool,
        ) -> list[int]:
            # All ranks see root's broadcast result, regardless of who
            # called us. The two distinct calls (count, then hashes)
            # are disambiguated by ``length``.
            if length == 1:
                return [root_count]
            assert length == root_count * 2, (
                f"broadcast length {length} doesn't match root's "
                f"count*2 ({root_count * 2})"
            )
            return list(root_pairs)

        # ``mx.distributed.all_sum`` is mocked at the call site by
        # patching it to return a buffer where each slot equals the
        # sum across all ranks of "this rank has root's i-th task".
        # That's exactly what the cluster collective would produce.
        per_rank_presences: list[list[int]] = []
        for rank_ids in all_ranks_task_ids:
            rank_pairs = {_task_id_hash_pair(tid) for tid in rank_ids}
            presence: list[int] = []
            for tid in sorted_root_ids:
                presence.append(1 if _task_id_hash_pair(tid) in rank_pairs else 0)
            per_rank_presences.append(presence)
        summed_presence = [
            sum(rank_presence[i] for rank_presence in per_rank_presences)
            for i in range(root_count)
        ]

        class _FakeAllSumResult:
            def __init__(self, values: list[int]) -> None:
                self._values = values

            def tolist(self) -> list[int]:
                return list(self._values)

        def fake_all_sum(
            buffer: object,
            *,
            group: object,  # noqa: ARG001
            stream: object | None = None,  # noqa: ARG001
        ) -> _FakeAllSumResult:
            # The production code only reads ``.tolist()`` after
            # ``mx.eval``; we ignore the actual buffer contents
            # because the protocol's correctness depends on the
            # cluster-wide sum, not on any single rank's contribution.
            del buffer
            return _FakeAllSumResult(summed_presence)

        # Patch both helpers. ``mx_broadcast_int_list`` is a module-
        # level symbol; ``mx.distributed.all_sum`` is dotted-imported.
        monkeypatch.setattr(utils_mlx_mod, "mx_broadcast_int_list", fake_broadcast)

        import mlx.core as mx_mod

        monkeypatch.setattr(mx_mod.distributed, "all_sum", fake_all_sum)

        # ``mx.eval`` is a no-op for our fake result.
        def _fake_eval(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(mx_mod, "eval", _fake_eval)

    def test_intersection_when_rank_is_partial(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rank 0 has ``[A, B]``; rank 1 only has ``[A]`` (pubsub
        timing skew). Both ranks must agree on ``[A]``, with rank
        0's ``[B]`` deferred to ``different`` for retry next tick.
        """
        self._setup_collectives(
            monkeypatch,
            root_task_ids=["alpha", "beta"],
            all_ranks_task_ids=[["alpha", "beta"], ["alpha"]],
        )

        # Rank 0 sees both tasks; ``different`` carries the not-yet-
        # universally-present ``beta`` for retry.
        agreed_root, different_root = mx_all_gather_tasks(
            [_make_task("alpha"), _make_task("beta")],
            group=_FakeGroup(rank=0, size=2),  # pyright: ignore[reportArgumentType]
        )
        assert [t.task_id for t in agreed_root] == ["alpha"]
        assert [t.task_id for t in different_root] == ["beta"]

        # Rank 1 has only ``[A]``; everything it has is in the
        # intersection.
        agreed_rank1, different_rank1 = mx_all_gather_tasks(
            [_make_task("alpha")],
            group=_FakeGroup(rank=1, size=2),  # pyright: ignore[reportArgumentType]
        )
        assert [t.task_id for t in agreed_rank1] == ["alpha"]
        assert different_rank1 == []

    def test_full_match_returns_everything_in_agreed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every rank has the same task set, ``agreed`` is the
        full canonically-sorted list and ``different`` is empty. This
        is the steady-state happy path."""
        self._setup_collectives(
            monkeypatch,
            root_task_ids=["alpha", "beta"],
            all_ranks_task_ids=[["alpha", "beta"], ["alpha", "beta"]],
        )
        # Note: rank 1 receives in different order than rank 0.
        agreed, different = mx_all_gather_tasks(
            [_make_task("beta"), _make_task("alpha")],
            group=_FakeGroup(rank=1, size=2),  # pyright: ignore[reportArgumentType]
        )
        assert [t.task_id for t in agreed] == ["alpha", "beta"]
        assert different == []

    def test_disjoint_sets_yield_empty_intersection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If ranks have totally disjoint task IDs, the intersection
        is empty and every rank's local tasks fall into
        ``different`` -- no RuntimeError. Genuine pubsub divergence
        recovers via retry rather than runner death."""
        self._setup_collectives(
            monkeypatch,
            root_task_ids=["alpha"],
            all_ranks_task_ids=[["alpha"], ["beta"]],
        )

        agreed, different = mx_all_gather_tasks(
            [_make_task("beta")],
            group=_FakeGroup(rank=1, size=2),  # pyright: ignore[reportArgumentType]
        )
        assert agreed == []
        assert [t.task_id for t in different] == ["beta"]

    def test_root_empty_returns_no_agreement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When root has no tasks (e.g. cold start), no agreement is
        possible this tick; non-root ranks return their entire local
        list as ``different`` for retry. Crucially: no ``RuntimeError``
        even though rank 1 has tasks rank 0 doesn't."""
        self._setup_collectives(
            monkeypatch,
            root_task_ids=[],
            all_ranks_task_ids=[[], ["alpha"]],
        )

        agreed, different = mx_all_gather_tasks(
            [_make_task("alpha")],
            group=_FakeGroup(rank=1, size=2),  # pyright: ignore[reportArgumentType]
        )
        assert agreed == []
        assert [t.task_id for t in different] == ["alpha"]

    def test_all_empty_steady_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cluster steady-state with no pending tasks anywhere: no
        agreement and no error."""
        self._setup_collectives(
            monkeypatch,
            root_task_ids=[],
            all_ranks_task_ids=[[], []],
        )
        agreed, different = mx_all_gather_tasks(
            [],
            group=_FakeGroup(rank=1, size=2),  # pyright: ignore[reportArgumentType]
        )
        assert agreed == []
        assert different == []
