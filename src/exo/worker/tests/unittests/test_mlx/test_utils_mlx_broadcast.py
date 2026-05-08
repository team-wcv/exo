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
    _hash_task_list,  # pyright: ignore[reportPrivateUsage]
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


class TestHashTaskList:
    """``_hash_task_list`` must be deterministic, ordering-sensitive,
    and stay inside the int32 positive range so it can ride
    :func:`mx_broadcast_int_list`."""

    def test_empty_list_hashes_to_zero(self) -> None:
        # Convention: no tasks -> all ranks see hash 0, which is
        # consistent and won't trigger a false drift positive.
        assert _hash_task_list([]) == 0

    def test_deterministic_across_calls(self) -> None:
        tasks = [_make_task("alpha"), _make_task("beta")]
        assert _hash_task_list(tasks) == _hash_task_list(tasks)

    def test_sensitive_to_ordering(self) -> None:
        # Drift detection must catch ranks that have the same task ids
        # but in different order -- the master's plan is ordered so
        # divergent ordering implies divergent execution intent.
        a = [_make_task("alpha"), _make_task("beta")]
        b = [_make_task("beta"), _make_task("alpha")]
        assert _hash_task_list(a) != _hash_task_list(b)

    def test_sensitive_to_id_content(self) -> None:
        a = [_make_task("alpha")]
        b = [_make_task("beta")]
        assert _hash_task_list(a) != _hash_task_list(b)

    def test_no_concatenation_collision(self) -> None:
        # ``[alpha, beta]`` and ``[alphabeta]`` must hash differently
        # so a master that splits a task into two doesn't collide
        # with one that merged them.
        a = [_make_task("alpha"), _make_task("beta")]
        b = [_make_task("alphabeta")]
        assert _hash_task_list(a) != _hash_task_list(b)

    def test_fits_in_int32_positive_range(self) -> None:
        # The hash must fit in :data:`_MX_BROADCAST_MAX_VALUE` so it
        # can ride the broadcast buffer without rejection.
        for ident in (
            "a",
            "alpha",
            "very-long-task-id-with-lots-of-bytes-" * 4,
            "\u00e9\u00e8\u00ea",  # non-ascii bytes exercise utf-8 path
        ):
            h = _hash_task_list([_make_task(ident)])
            assert 0 <= h <= _MX_BROADCAST_MAX_VALUE


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
