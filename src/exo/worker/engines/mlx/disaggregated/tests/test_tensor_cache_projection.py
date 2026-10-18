"""Cache transport preserves Qwen sparse state and each tensor rank's heads."""

import io
from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from exo.worker.disaggregated.protocol import (
    ArraysState,
    Done,
    IndexerState,
    KVChunk,
    read_header,
    read_message,
)
from exo.worker.engines.mlx.disaggregated.adapter import (
    array_to_bytes,
    bytes_to_array,
    write_cache_to_wire,
)
from exo.worker.engines.mlx.disaggregated.cache_projection import (
    CacheLayout,
    project_kv_heads,
    project_linear_state,
)
from exo.worker.engines.mlx.disaggregated.client import (
    PrefillResult,
    ingest_into_mlx_cache,
)


class _Indexer:
    def __init__(self, state: mx.array | None = None) -> None:
        self.state = state


class _SparseCache(KVCache):
    def __init__(self) -> None:
        super().__init__()
        self.indexer = _Indexer()


def _decode(stream: io.BytesIO) -> PrefillResult:
    stream.seek(0)
    result = PrefillResult(header=read_header(stream))
    while (message := read_message(stream)) is not None:
        if isinstance(message, KVChunk):
            result.kv_chunks.setdefault(message.layer_idx, []).append(message)
        elif isinstance(message, ArraysState):
            result.arrays[message.layer_idx] = message.arrays
        elif isinstance(message, IndexerState):
            result.indexers[message.layer_idx] = message.tensor
        elif isinstance(message, Done):
            result.total_tokens = message.total_tokens
            break
        else:
            raise AssertionError(f"Unexpected wire message: {message}")
    return result


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_integer_history_roundtrip_is_lossless(dtype: str) -> None:
    if dtype == "int32":
        source = mx.array([2_000_000_001, -2_000_000_001], dtype=mx.int32)
        restored = bytes_to_array(array_to_bytes(source), (2,), "int32")
    else:
        source = mx.array([2**50 + 1, -(2**50) + 7], dtype=mx.int64)
        restored = bytes_to_array(array_to_bytes(source), (2,), "int64")
    assert source.dtype == restored.dtype
    assert bool(mx.array_equal(source, restored))


@pytest.mark.parametrize("rank", [0, 1])
def test_linear_projection_preserves_projection_order_and_auxiliary_slots(
    rank: int,
) -> None:
    convolution = mx.arange(14).reshape(1, 1, 14).astype(mx.float32)
    recurrent = mx.arange(24).reshape(1, 4, 2, 3).astype(mx.float32)
    history = mx.array([[248045, 199999]], dtype=mx.int32)
    state = project_linear_state(
        [convolution, recurrent, None, history],
        CacheLayout(linear_key_dim=2, linear_value_dim=3, linear_value_heads=2),
        rank,
        2,
    )
    expected = [[0, 1, 4, 5, 8, 9, 10], [2, 3, 6, 7, 11, 12, 13]][rank]
    assert state[0] is not None and state[1] is not None
    assert state[0].reshape(-1).tolist() == expected
    assert bool(mx.array_equal(state[1], recurrent[:, rank * 2 : (rank + 1) * 2]))
    assert state[2] is None
    assert state[3] is history


def test_replicated_gqa_heads_follow_contiguous_rank_groups() -> None:
    values = mx.array([[[10.0], [20.0]]])
    actual = [project_kv_heads(values, 1, rank, 4).item() for rank in range(4)]
    assert actual == [10.0, 10.0, 20.0, 20.0]


def test_invalid_kv_projection_is_rejected() -> None:
    with pytest.raises(ValueError, match="expects 2 local heads"):
        project_kv_heads(mx.zeros((3, 3, 4)), 2, 0, 2)


@pytest.mark.parametrize("rank", [0, 1])
def test_qwen_cache_wire_roundtrip_to_tensor_rank(rank: int) -> None:
    source = _SparseCache()
    source.keys = mx.arange(60).reshape(1, 4, 5, 3).astype(mx.float32)
    source.values = source.keys + 100
    source.offset = 5
    source.indexer.state = mx.arange(10).reshape(1, 5, 2).astype(mx.float32)
    arrays = ArraysCache(4)
    arrays.state = [
        mx.arange(42).reshape(1, 3, 14).astype(mx.float32),
        mx.arange(24).reshape(1, 4, 2, 3).astype(mx.float32),
        None,
        mx.array([[248045, 200001]], dtype=mx.int32),
    ]
    stream = io.BytesIO()
    write_cache_to_wire(stream, [source, arrays], start_pos=2)
    result = _decode(stream)
    target = _SparseCache()
    target.keys = source.keys[:, rank * 2 : (rank + 1) * 2, :2]
    target.values = source.values[:, rank * 2 : (rank + 1) * 2, :2]
    target.offset = 2
    target_arrays = ArraysCache(4)
    offset = ingest_into_mlx_cache(
        result,
        [target, target_arrays],
        start_pos=2,
        rank=rank,
        world_size=2,
        layouts=[
            CacheLayout(kv_heads=2),
            CacheLayout(linear_key_dim=2, linear_value_dim=3, linear_value_heads=2),
        ],
    )
    assert offset == 5
    assert target.keys is not None
    assert bool(mx.array_equal(target.keys, source.keys[:, rank * 2 : (rank + 1) * 2]))
    assert target.indexer.state is not None
    assert bool(mx.array_equal(target.indexer.state, source.indexer.state))
    state = list(target_arrays.state)
    assert state[2] is None
    history = state[3]
    assert history is not None
    assert history.dtype == mx.int32
    assert history.tolist() == [[248045, 200001]]


def test_missing_sparse_indexer_does_not_mutate_decoder_cache() -> None:
    source = _SparseCache()
    source.keys = mx.ones((1, 2, 3, 4))
    source.values = mx.ones((1, 2, 3, 4))
    source.offset = 3
    source.indexer.state = mx.ones((1, 3, 2))
    stream = io.BytesIO()
    write_cache_to_wire(stream, [source])
    result = _decode(stream)
    result.indexers.clear()
    target = _SparseCache()
    with pytest.raises(ValueError, match="missing indexer state"):
        ingest_into_mlx_cache(result, [target])
    assert target.keys is None
    assert target.offset == 0


@pytest.mark.parametrize("local_failure", [False, True])
def test_any_rank_prefill_failure_prevents_all_ranks_entering_decode(
    monkeypatch: pytest.MonkeyPatch, local_failure: bool
) -> None:
    from exo.worker.engines.mlx.generator.remote_prefill import (
        synchronize_prefill_outcome,
    )

    class Group:
        def size(self) -> int:
            return 2

    group = cast(mx.distributed.Group, cast(object, Group()))
    flags: list[float] = []

    def all_sum(flag: mx.array, *, group: mx.distributed.Group) -> mx.array:
        assert group.size() == 2
        flags.append(float(flag.item()))
        return mx.array(1.0)

    monkeypatch.setattr(mx.distributed, "all_sum", all_sum)
    error = ConnectionError("busy") if local_failure else None
    with pytest.raises(RuntimeError, match="tensor decode rank"):
        synchronize_prefill_outcome(error, group)
    assert flags == [float(local_failure)]
