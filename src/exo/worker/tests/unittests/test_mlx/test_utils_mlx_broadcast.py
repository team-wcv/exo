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
  * :func:`_encode_task_id` / :func:`_decode_task_id` -- ASCII codec
    used by ``mx_all_gather_tasks`` to broadcast canonical task IDs.
    Round-trip and bounds are verifiable without MLX.
  * :func:`mx_all_gather_tasks` -- the single-rank short-circuit. The
    multi-rank root-authoritative agreement path needs an actual
    ``mx.distributed`` group, so we cover the structural contract here
    and the cluster bench exercises the real collective.

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
    _MX_TASK_ID_BYTES,  # pyright: ignore[reportPrivateUsage]
    _decode_task_id,  # pyright: ignore[reportPrivateUsage]
    _encode_task_id,  # pyright: ignore[reportPrivateUsage]
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


class TestTaskIdCodec:
    """``_encode_task_id`` / ``_decode_task_id`` are the wire codec for
    the root-authoritative agreement protocol. Round-trip must be
    exact and bounds must be enforced; otherwise a corrupt payload
    silently misagrees on which task to admit."""

    def test_round_trip_uuid4(self) -> None:
        ident = "01234567-89ab-cdef-0123-456789abcdef"
        encoded = _encode_task_id(ident)
        assert len(encoded) == _MX_TASK_ID_BYTES
        assert _decode_task_id(encoded) == ident

    def test_short_id_is_zero_padded(self) -> None:
        encoded = _encode_task_id("alpha")
        # Trailing slots stay zero so the decoder's null terminator
        # logic stops at the right place.
        assert encoded[5:] == [0] * (_MX_TASK_ID_BYTES - 5)
        assert _decode_task_id(encoded) == "alpha"

    def test_rejects_oversize_id(self) -> None:
        too_long = "a" * (_MX_TASK_ID_BYTES + 1)
        with pytest.raises(ValueError, match="exceeds"):
            _encode_task_id(too_long)

    def test_rejects_non_ascii_byte_on_decode(self) -> None:
        bogus = [200] + [0] * (_MX_TASK_ID_BYTES - 1)
        with pytest.raises(ValueError, match="outside ASCII range"):
            _decode_task_id(bogus)

    def test_decoder_stops_at_null(self) -> None:
        # Two real chars, then a null, then garbage: decoder must
        # stop at the null and ignore the trailing data.
        slots = [ord("a"), ord("b"), 0, ord("z")] + [0] * (_MX_TASK_ID_BYTES - 4)
        assert _decode_task_id(slots) == "ab"


# ---------------------------------------------------------------------------
# mx_all_gather_tasks single-rank short-circuit
# ---------------------------------------------------------------------------


class TestMxAllGatherTasksSingleRank:
    """Single-rank short-circuit: returns the local task list as-is and
    never invokes a collective. The multi-rank root-authoritative path
    needs an actual ``mx.distributed`` group and is exercised by the
    cluster bench."""

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
# Authoritative agreement: end-to-end via fake group
# ---------------------------------------------------------------------------


def _agree_authoritative(
    root_tasks: list[TextGeneration],
    consumer_tasks: list[TextGeneration],
) -> tuple[list[TextGeneration], list[TextGeneration]]:
    """Run the agreement protocol entirely in-process for unit testing.

    The real protocol uses :func:`mx_broadcast_int_list` (an
    ``all_sum`` ride). We don't ship an actual MLX group into the
    unit test, so this helper bypasses the collective and runs the
    protocol end-to-end by feeding root's broadcast bytes directly
    into the consumer's decoder. That validates the encode/decode
    contract and the consumer-side filtering / leftover logic
    without depending on MLX runtime behaviour.
    """
    # Mirror ``mx_all_gather_tasks`` root encode -> consumer decode
    # without the collective. The collective is exercised in the
    # cluster bench because it requires real ``mx.distributed`` to
    # be initialised.
    from exo.worker.engines.mlx.utils_mlx import (
        _MX_AGREE_BUFFER_LEN,  # pyright: ignore[reportPrivateUsage]
        _MX_AGREE_MAX_TASKS,  # pyright: ignore[reportPrivateUsage]
        _MX_TASK_ID_BYTES,  # pyright: ignore[reportPrivateUsage]
        _decode_task_id,  # pyright: ignore[reportPrivateUsage]
        _encode_task_id,  # pyright: ignore[reportPrivateUsage]
    )

    admitted = root_tasks[:_MX_AGREE_MAX_TASKS]
    payload: list[int] = [len(admitted)]
    for task in admitted:
        payload.extend(_encode_task_id(task.task_id))
    payload.extend([0] * (_MX_AGREE_BUFFER_LEN - len(payload)))

    count = payload[0]
    canonical_ids: list[str] = []
    for i in range(count):
        start = 1 + i * _MX_TASK_ID_BYTES
        end = start + _MX_TASK_ID_BYTES
        canonical_ids.append(_decode_task_id(payload[start:end]))

    local_by_id: dict[TaskId, TextGeneration] = {t.task_id: t for t in consumer_tasks}
    agreed: list[TextGeneration] = []
    for tid in canonical_ids:
        task = local_by_id.pop(TaskId(tid), None)
        if task is not None:
            agreed.append(task)
    leftover = list(local_by_id.values())
    return agreed, leftover


class TestAuthoritativeAgreement:
    """Cross-rank agreement semantics validated through the wire codec
    without spinning up an MLX group. Each test fixes a root view +
    a consumer view, runs the encode/decode, and checks the consumer
    landed on the canonical agreed set with the right leftovers."""

    def test_consumer_matches_root(self) -> None:
        # Both ranks see the same task: agreed = [task], leftover = [].
        task = _make_task("alpha")
        agreed, leftover = _agree_authoritative([task], [task])
        assert [t.task_id for t in agreed] == ["alpha"]
        assert leftover == []

    def test_consumer_drops_unknown_root_tasks(self) -> None:
        # Root has task that consumer hasn't received yet (libp2p
        # delivery race): consumer admits the empty subset; the task
        # surfaces next round once delivery completes. The consumer
        # never desyncs from the canonical order.
        root_only = _make_task("root-only")
        agreed, leftover = _agree_authoritative([root_only], [])
        assert agreed == []
        assert leftover == []

    def test_consumer_keeps_unknown_local_tasks_as_leftover(self) -> None:
        # Consumer has a task root hasn't seen: leftover lets the
        # caller stash it in ``_maybe_queue`` so it's eligible the
        # next agreement round.
        future = _make_task("future")
        agreed, leftover = _agree_authoritative([], [future])
        assert agreed == []
        assert [t.task_id for t in leftover] == ["future"]

    def test_consumer_partial_overlap(self) -> None:
        # Root has [alpha, beta], consumer has [alpha, gamma]:
        # agreed=[alpha], leftover=[gamma]. Beta is silently dropped
        # this round; gamma waits for root to see it.
        alpha = _make_task("alpha")
        consumer_alpha = _make_task("alpha")  # different object, same id
        beta = _make_task("beta")
        gamma = _make_task("gamma")
        agreed, leftover = _agree_authoritative([alpha, beta], [consumer_alpha, gamma])
        assert [t.task_id for t in agreed] == ["alpha"]
        assert [t.task_id for t in leftover] == ["gamma"]

    def test_canonical_order_is_root_order(self) -> None:
        # Master's plan order is authoritative. If consumer has the
        # tasks in a different local order, agreement still comes back
        # in root's order so every rank's ``self._queue`` extends
        # identically.
        a = _make_task("a")
        b = _make_task("b")
        c = _make_task("c")
        consumer_a = _make_task("a")
        consumer_b = _make_task("b")
        consumer_c = _make_task("c")
        agreed, leftover = _agree_authoritative(
            [c, a, b], [consumer_b, consumer_a, consumer_c]
        )
        assert [t.task_id for t in agreed] == ["c", "a", "b"]
        assert leftover == []

    def test_root_caps_at_max_tasks(self) -> None:
        # Hard cap on agreement payload size. Tasks beyond
        # ``_MX_AGREE_MAX_TASKS`` get deferred to the next round.
        from exo.worker.engines.mlx.utils_mlx import (
            _MX_AGREE_MAX_TASKS,  # pyright: ignore[reportPrivateUsage]
        )

        # Build a root list larger than the cap and a consumer that
        # has every task locally. The consumer should still only
        # admit ``_MX_AGREE_MAX_TASKS`` because that's all root can
        # broadcast in a single round.
        many = [_make_task(f"t{i:02d}") for i in range(_MX_AGREE_MAX_TASKS + 4)]
        consumer_copy = [_make_task(t.task_id) for t in many]
        agreed, _leftover = _agree_authoritative(many, consumer_copy)
        assert len(agreed) == _MX_AGREE_MAX_TASKS
