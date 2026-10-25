import io

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.worker.disaggregated.protocol import (
    ArraysState,
    Done,
    Header,
    KVChunk,
    TensorBlob,
    read_header,
    read_message,
    write_done,
    write_header,
)
from exo.worker.engines.mlx.disaggregated.adapter import (
    array_to_bytes,
    bhsd_to_nhd,
    bytes_to_array,
    chunk_to_mlx_nhd,
    inject_arrays_cache,
    inject_kv_chunk,
    inject_rotating_kv_chunk,
    nhd_to_bhsd,
    send_mlx_kv_cache,
    wire_dtype_from_cache,
)
from exo.worker.engines.mlx.disaggregated.client import (
    PrefillResult,
    ingest_into_mlx_cache,
)


def _equal(a: mx.array, b: mx.array) -> bool:
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    if a.dtype == mx.bfloat16:
        return bool(
            np.array_equal(np.asarray(a.view(mx.uint16)), np.asarray(b.view(mx.uint16)))
        )
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _rand(shape: tuple[int, ...], dtype: mx.Dtype) -> mx.array:
    mx.random.seed(0)
    return (mx.random.uniform(shape=shape) * 10).astype(dtype)


def _make_kv_cache(seq_len: int, n_heads: int, head_dim: int) -> KVCache:
    cache = KVCache()
    cache.keys = _rand((1, n_heads, seq_len, head_dim), mx.bfloat16)
    cache.values = _rand((1, n_heads, seq_len, head_dim), mx.bfloat16)
    cache.offset = seq_len
    return cache


def test_bytes_roundtrip_bf16() -> None:
    x = _rand((2, 3, 4), mx.bfloat16)
    y = bytes_to_array(array_to_bytes(x), (2, 3, 4), "bfloat16")
    assert _equal(x, y)


def test_bytes_roundtrip_f16() -> None:
    x = _rand((5,), mx.float16)
    y = bytes_to_array(array_to_bytes(x), (5,), "float16")
    assert _equal(x, y)


def test_bytes_roundtrip_f32() -> None:
    x = _rand((2, 2), mx.float32)
    y = bytes_to_array(array_to_bytes(x), (2, 2), "float32")
    assert _equal(x, y)


def test_bhsd_nhd_roundtrip() -> None:
    bhsd = _rand((1, 4, 7, 8), mx.float32)
    nhd = bhsd_to_nhd(bhsd)
    assert tuple(nhd.shape) == (7, 4, 8)
    back = nhd_to_bhsd(nhd)
    assert _equal(bhsd, back)


def test_kv_cache_inject_roundtrip() -> None:
    n_heads, seq_len, head_dim = 3, 5, 4
    k_bhsd = _rand((1, n_heads, seq_len, head_dim), mx.float32)
    v_bhsd = _rand((1, n_heads, seq_len, head_dim), mx.float32)
    k_nhd = bhsd_to_nhd(k_bhsd)
    v_nhd = bhsd_to_nhd(v_bhsd)

    cache = KVCache()
    inject_kv_chunk(cache, k_nhd, v_nhd, offset=seq_len)
    assert cache.offset == seq_len
    assert cache.keys is not None and cache.values is not None
    assert _equal(cache.keys, k_bhsd)
    assert _equal(cache.values, v_bhsd)


def test_arrays_cache_inject() -> None:
    a = _rand((3,), mx.float32)
    b = _rand((2, 2), mx.bfloat16)
    blobs = [
        TensorBlob(dtype="float32", shape=(3,), data=array_to_bytes(a)),
        TensorBlob(dtype="bfloat16", shape=(2, 2), data=array_to_bytes(b)),
    ]
    cache = ArraysCache(size=2)
    inject_arrays_cache(cache, blobs)
    s0 = cache.state[0]
    s1 = cache.state[1]
    assert s0 is not None and s1 is not None
    assert _equal(s0, a)
    assert _equal(s1, b)


def test_send_mlx_cache_end_to_end() -> None:
    n_heads, head_dim = 2, 4
    seq_len = 3
    src = _make_kv_cache(seq_len, n_heads, head_dim)
    k_bhsd, v_bhsd = src.keys, src.values
    assert k_bhsd is not None and v_bhsd is not None

    buf = io.BytesIO()
    write_header(buf, Header(num_layers=1, dtype="bfloat16"))
    tokens = send_mlx_kv_cache(buf, [src], dtype="bfloat16")
    write_done(buf, tokens)
    buf.seek(0)

    got_hdr = read_header(buf)
    assert got_hdr.num_layers == 1

    msg = read_message(buf)
    assert isinstance(msg, KVChunk)
    assert msg.num_tokens == seq_len
    k_nhd, v_nhd = chunk_to_mlx_nhd(msg)
    dst = KVCache()
    inject_kv_chunk(dst, k_nhd, v_nhd, offset=msg.num_tokens)

    done = read_message(buf)
    assert isinstance(done, Done)
    assert done.total_tokens == seq_len

    assert dst.offset == seq_len
    assert dst.keys is not None and dst.values is not None
    assert _equal(dst.keys, k_bhsd)
    assert _equal(dst.values, v_bhsd)
    _ = ArraysState


def test_send_with_start_pos_only_ships_suffix() -> None:
    n_heads, head_dim = 2, 4
    seq_len, start_pos = 6, 4
    src = _make_kv_cache(seq_len, n_heads, head_dim)

    buf = io.BytesIO()
    write_header(buf, Header(num_layers=1, dtype="bfloat16", start_pos=start_pos))
    tokens = send_mlx_kv_cache(buf, [src], dtype="bfloat16", start_pos=start_pos)
    write_done(buf, tokens)
    buf.seek(0)

    _ = read_header(buf)
    msg = read_message(buf)
    assert isinstance(msg, KVChunk)
    assert msg.num_tokens == seq_len - start_pos


def test_send_skips_layer_when_offset_below_start_pos() -> None:
    n_heads, head_dim = 2, 4
    seq_len, start_pos = 3, 5
    src = _make_kv_cache(seq_len, n_heads, head_dim)

    buf = io.BytesIO()
    write_header(buf, Header(num_layers=1, dtype="bfloat16", start_pos=start_pos))
    tokens = send_mlx_kv_cache(buf, [src], dtype="bfloat16", start_pos=start_pos)
    write_done(buf, tokens)
    buf.seek(0)

    _ = read_header(buf)
    msg = read_message(buf)
    assert isinstance(msg, Done)
    assert msg.total_tokens == 0
    assert tokens == 0


def test_wire_dtype_from_cache() -> None:
    src = _make_kv_cache(3, 2, 4)
    assert wire_dtype_from_cache([src]) == "bfloat16"

    f32 = KVCache()
    f32.keys = _rand((1, 2, 3, 4), mx.float32)
    f32.values = _rand((1, 2, 3, 4), mx.float32)
    f32.offset = 3
    assert wire_dtype_from_cache([f32]) == "float32"


def _decode_payload(payload: bytes) -> PrefillResult:
    buf = io.BytesIO(payload)
    hdr = read_header(buf)
    result = PrefillResult(header=hdr)
    while True:
        msg = read_message(buf)
        if msg is None:
            break
        if isinstance(msg, KVChunk):
            result.kv_chunks.setdefault(msg.layer_idx, []).append(msg)
        elif isinstance(msg, ArraysState):
            result.arrays[msg.layer_idx] = msg.arrays
        elif isinstance(msg, Done):
            result.total_tokens = msg.total_tokens
            break
    return result


def test_mixed_cache_roundtrip() -> None:
    n_heads, head_dim, seq_len = 2, 4, 6

    src_kv = _make_kv_cache(seq_len, n_heads, head_dim)

    src_rot = RotatingKVCache(max_size=16, keep=0)
    src_rot.keys = _rand((1, n_heads, seq_len, head_dim), mx.bfloat16)
    src_rot.values = _rand((1, n_heads, seq_len, head_dim), mx.bfloat16)
    src_rot.offset = seq_len
    src_rot._idx = seq_len

    src_arr = ArraysCache(size=2)
    arr_a = _rand((3,), mx.bfloat16)
    arr_b = _rand((2, 4), mx.bfloat16)
    src_arr.state = [arr_a, arr_b]

    buf = io.BytesIO()
    write_header(
        buf,
        Header(request_id="req", model_id="m", num_layers=3, dtype="bfloat16"),
    )
    tokens_sent = send_mlx_kv_cache(buf, [src_kv, src_rot, src_arr], dtype="bfloat16")
    write_done(buf, tokens_sent)
    result = _decode_payload(buf.getvalue())

    assert result.header.num_layers == 3
    assert result.total_tokens == seq_len

    dst_kv = KVCache()
    dst_rot = RotatingKVCache(max_size=16, keep=0)
    dst_arr = ArraysCache(size=2)
    final_offset = ingest_into_mlx_cache(result, [dst_kv, dst_rot, dst_arr])

    assert final_offset == seq_len

    assert dst_kv.offset == seq_len
    assert dst_kv.keys is not None and dst_kv.values is not None
    src_kv_k, src_kv_v = src_kv.keys, src_kv.values
    assert src_kv_k is not None and src_kv_v is not None
    assert _equal(dst_kv.keys, src_kv_k)
    assert _equal(dst_kv.values, src_kv_v)

    assert dst_rot.offset == seq_len
    assert dst_rot.keys is not None and dst_rot.values is not None
    src_rot_k, src_rot_v = src_rot.keys, src_rot.values
    assert src_rot_k is not None and src_rot_v is not None
    assert _equal(dst_rot.keys, src_rot_k)
    assert _equal(dst_rot.values, src_rot_v)
    assert dst_rot._idx == seq_len

    assert len(dst_arr.state) == 2
    s0, s1 = dst_arr.state[0], dst_arr.state[1]
    assert s0 is not None and s1 is not None
    assert _equal(s0, arr_a)
    assert _equal(s1, arr_b)
    _ = inject_rotating_kv_chunk
    _ = nhd_to_bhsd


@pytest.mark.parametrize("keep", [0, 2])
@pytest.mark.parametrize("start_pos", [0, 60])
def test_wrapped_rotating_transfer_preserves_position_and_next_update(
    keep: int, start_pos: int
) -> None:
    seq_len, window = 67, 8
    full = _make_kv_cache(seq_len, 2, 4)
    assert full.keys is not None and full.values is not None
    rotating = RotatingKVCache(max_size=window, keep=keep)
    for position in range(seq_len):
        values = mx.full((1, 2, 1, 4), position, dtype=mx.bfloat16)
        rotating.update_and_fetch(values, values + 100)
    positions = [*range(keep), *range(seq_len - (window - keep), seq_len)]
    expected_keys = mx.broadcast_to(
        mx.array(positions, dtype=mx.bfloat16).reshape(1, 1, window, 1),
        (1, 2, window, 4),
    )
    expected_values = expected_keys + 100
    buf = io.BytesIO()
    write_header(buf, Header(num_layers=2, dtype="bfloat16", start_pos=start_pos))
    logical_tokens = send_mlx_kv_cache(
        buf, [full, rotating], dtype="bfloat16", start_pos=start_pos
    )
    write_done(buf, logical_tokens)
    result = _decode_payload(buf.getvalue())
    assert logical_tokens == seq_len - start_pos
    assert result.kv_chunks[1][0].num_tokens == window
    destination_full = _make_kv_cache(start_pos, 2, 4) if start_pos else KVCache()
    if start_pos:
        destination_full.keys = full.keys[:, :, :start_pos, :]
        destination_full.values = full.values[:, :, :start_pos, :]
    destination_rotating = RotatingKVCache(max_size=window, keep=keep)
    offset = ingest_into_mlx_cache(
        result, [destination_full, destination_rotating], start_pos=start_pos
    )
    assert offset == seq_len
    assert destination_rotating.offset == seq_len
    assert destination_rotating._idx == window
    assert destination_rotating.keys is not None
    assert destination_rotating.values is not None
    assert destination_full.keys is not None
    assert _equal(destination_rotating.keys, expected_keys)
    assert _equal(destination_rotating.values, expected_values)
    assert _equal(destination_full.keys, full.keys)
    next_value = mx.full((1, 2, 1, 4), seq_len, dtype=mx.bfloat16)
    rotating.update_and_fetch(next_value, next_value + 100)
    destination_rotating.update_and_fetch(next_value, next_value + 100)
    assert rotating.keys is not None
    next_positions = mx.array(
        [*range(keep), *range(seq_len + 1 - (window - keep), seq_len + 1)],
        dtype=mx.bfloat16,
    )
    assert _equal(mx.sort(rotating.keys[0, 0, :, 0]), next_positions)
    assert _equal(mx.sort(destination_rotating.keys[0, 0, :, 0]), next_positions)


def test_truncated_sliding_window_is_rejected_before_any_cache_mutation() -> None:
    full = _make_kv_cache(64, 2, 4)
    rotating = RotatingKVCache(max_size=8, keep=0)
    rotating.keys = _rand((1, 2, 8, 4), mx.bfloat16)
    rotating.values = _rand((1, 2, 8, 4), mx.bfloat16)
    rotating.offset = 64
    rotating._idx = 8
    buf = io.BytesIO()
    write_header(buf, Header(num_layers=2, dtype="bfloat16"))
    count = send_mlx_kv_cache(buf, [full, rotating], dtype="bfloat16")
    write_done(buf, count)
    result = _decode_payload(buf.getvalue())
    result.kv_chunks[1][0].num_tokens -= 1
    target_full = _make_kv_cache(3, 2, 4)
    target_rotating = RotatingKVCache(max_size=8, keep=0)
    original_keys = target_full.keys
    with pytest.raises(ValueError, match="incomplete token sequence"):
        ingest_into_mlx_cache(result, [target_full, target_rotating])
    assert target_full.keys is original_keys
    assert target_full.offset == 3
    assert target_rotating.keys is None


def test_rotating_export_refuses_an_evicted_earlier_boundary() -> None:
    rotating = RotatingKVCache(max_size=8, keep=0)
    for position in range(64):
        values = mx.full((1, 2, 1, 4), position, dtype=mx.bfloat16)
        rotating.update_and_fetch(values, values)
    with pytest.raises(ValueError, match="Cannot reconstruct rotating cache"):
        send_mlx_kv_cache(io.BytesIO(), [rotating], dtype="bfloat16", max_tokens=60)
