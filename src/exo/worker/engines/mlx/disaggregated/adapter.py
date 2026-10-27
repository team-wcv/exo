from collections.abc import Sequence
from typing import BinaryIO

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache

from exo.worker.disaggregated.protocol import (
    DType,
    Header,
    IndexerState,
    KVChunk,
    TensorBlob,
    write_arrays_state,
    write_done,
    write_header,
    write_kv_chunk,
    write_message,
)
from exo.worker.engines.mlx.types import KVCacheType

_STR_TO_MX: dict[DType, mx.Dtype] = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
    "int32": mx.int32,
    "int64": mx.int64,
}

_MX_TO_STR: dict[mx.Dtype, DType] = {v: k for k, v in _STR_TO_MX.items()}


def mx_dtype_to_str(dtype: mx.Dtype) -> DType:
    if dtype not in _MX_TO_STR:
        raise ValueError(f"Unsupported mlx dtype on wire: {dtype}")
    return _MX_TO_STR[dtype]


def wire_dtype_from_cache(caches: KVCacheType) -> DType:
    for c in caches:
        keys: mx.array | None = getattr(c, "keys", None)
        if keys is None:
            continue
        if keys.dtype in _MX_TO_STR:
            return _MX_TO_STR[keys.dtype]
        break
    return "bfloat16"


def str_to_mx_dtype(dtype: DType) -> mx.Dtype:
    if dtype not in _STR_TO_MX:
        raise ValueError(f"Unsupported wire dtype: {dtype!r}")
    return _STR_TO_MX[dtype]


def array_to_bytes(t: mx.array) -> bytes:
    # bf16 has no native numpy dtype; bitcast through uint16.
    if t.dtype == mx.bfloat16:
        return np.asarray(t.view(mx.uint16)).tobytes()
    if t.dtype in (mx.float16, mx.float32, mx.int32, mx.int64):
        return np.asarray(t).tobytes()
    raise ValueError(f"Unsupported mlx dtype for wire: {t.dtype}")


def bytes_to_array(data: bytes, shape: tuple[int, ...], dtype: DType) -> mx.array:
    match dtype:
        case "bfloat16":
            arr = np.frombuffer(data, dtype=np.uint16).reshape(shape).copy()
            return mx.array(arr).view(mx.bfloat16)
        case "float16":
            arr = np.frombuffer(data, dtype=np.float16).reshape(shape).copy()
            return mx.array(arr)
        case "float32":
            arr = np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
            return mx.array(arr)
        case "int32":
            arr = np.frombuffer(data, dtype=np.int32).reshape(shape).copy()
            return mx.array(arr)
        case "int64":
            arr = np.frombuffer(data, dtype=np.int64).reshape(shape).copy()
            return mx.array(arr)


def bhsd_to_nhd(t: mx.array) -> mx.array:
    if t.ndim != 4 or int(t.shape[0]) != 1:
        raise ValueError(f"Expected BHSD with B=1, got shape={tuple(t.shape)}")
    return mx.transpose(t[0], (1, 0, 2))


def nhd_to_bhsd(t: mx.array) -> mx.array:
    if t.ndim != 3:
        raise ValueError(f"Expected NHD (3D), got shape={tuple(t.shape)}")
    return mx.expand_dims(mx.transpose(t, (1, 0, 2)), 0)


def _rotating_prefix_window(
    cache: RotatingKVCache, offset: int
) -> tuple[mx.array, mx.array]:
    """Normalize the retained ring window without confusing storage with position."""
    if cache.keys is None or cache.values is None:
        raise ValueError("Cannot export an empty rotating attention cache")

    def ordered(values: mx.array) -> mx.array:
        if cache._idx == int(values.shape[2]):
            return values
        if cache._idx < cache.offset:
            return mx.concatenate(
                [
                    values[:, :, : cache.keep, :],
                    values[:, :, cache._idx :, :],
                    values[:, :, cache.keep : cache._idx, :],
                ],
                axis=2,
            )
        return values[:, :, : cache._idx, :]

    keys = ordered(cache.keys)
    values = ordered(cache.values)
    stored = int(keys.shape[2])
    if int(values.shape[2]) != stored:
        raise ValueError("Rotating cache key/value lengths differ")
    retained = min(offset, int(cache.max_size))
    kept = min(int(cache.keep), retained)
    stored_keep = min(int(cache.keep), stored)
    tail_start = int(cache.offset) - (stored - stored_keep)
    needed_start = offset - (retained - kept)
    if offset > cache.offset or (retained > kept and needed_start < tail_start):
        raise ValueError(
            f"Cannot reconstruct rotating cache at offset {offset}: "
            f"available tail starts at {tail_start}, needed {needed_start}"
        )
    begin = stored_keep + needed_start - tail_start
    end = stored_keep + offset - tail_start
    if retained == kept:
        return keys[:, :, :kept, :], values[:, :, :kept, :]
    return (
        mx.concatenate([keys[:, :, :kept, :], keys[:, :, begin:end, :]], axis=2),
        mx.concatenate([values[:, :, :kept, :], values[:, :, begin:end, :]], axis=2),
    )


def send_mlx_kv_cache(
    stream: BinaryIO,
    caches: KVCacheType,
    *,
    dtype: DType,
    start_pos: int = 0,
    max_tokens: int | None = None,
) -> int:
    tokens_sent = 0
    for layer_idx, c in enumerate(caches):
        match c:
            case QuantizedKVCache() | CacheList() | DeepseekV4Cache():
                raise NotImplementedError
            case KVCache() | RotatingKVCache():
                keys = c.keys
                values = c.values
                if keys is None or values is None:
                    continue
                offset = int(c.offset)
                if max_tokens is not None:
                    offset = min(offset, max_tokens)
                if offset <= start_pos:
                    continue
                logical_tokens = offset - start_pos
                if tokens_sent and logical_tokens != tokens_sent:
                    raise ValueError(
                        f"Attention cache layer {layer_idx} covers {logical_tokens} "
                        f"new tokens; expected {tokens_sent}"
                    )
                tokens_sent = logical_tokens
                with mx.stream(mx.Device(mx.cpu)):
                    if isinstance(c, RotatingKVCache):
                        # A sliding layer sends its complete retained window,
                        # including on an incremental full-cache transfer.
                        k, v = _rotating_prefix_window(c, offset)
                        k, v = mx.array(k), mx.array(v)
                    else:
                        k = mx.array(keys[:, :, start_pos:offset, :])
                        v = mx.array(values[:, :, start_pos:offset, :])
                    k_nhd = bhsd_to_nhd(k)
                    v_nhd = bhsd_to_nhd(v)
                    mx.eval(k_nhd, v_nhd)
                num_tokens = int(k_nhd.shape[0])
                n_heads = int(k_nhd.shape[1])
                head_dim = int(k_nhd.shape[2])
                write_kv_chunk(
                    stream,
                    layer_idx=layer_idx,
                    num_tokens=num_tokens,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    keys=array_to_bytes(k_nhd),
                    values=array_to_bytes(v_nhd),
                )
                indexer = getattr(c, "indexer", None)
                indexer_state: object = getattr(indexer, "state", None)
                if isinstance(indexer_state, mx.array):
                    # The indexer is replicated in TP and needs the entire prefix,
                    # even when the attention KV transfer contains only a suffix.
                    indexer_keys = indexer_state[:, :offset]
                    mx.eval(indexer_keys)
                    write_message(
                        stream,
                        IndexerState(
                            layer_idx=layer_idx,
                            tensor=TensorBlob(
                                dtype=mx_dtype_to_str(indexer_keys.dtype),
                                shape=tuple(indexer_keys.shape),
                                data=array_to_bytes(indexer_keys),
                            ),
                        ),
                    )
            case ArraysCache():
                blobs: list[TensorBlob | None] = []
                for a in c.state:
                    if a is None:
                        blobs.append(None)
                        continue
                    with mx.stream(mx.Device(mx.cpu)):
                        a_cpu = mx.array(a)
                        mx.eval(a_cpu)
                    blobs.append(
                        TensorBlob(
                            dtype=mx_dtype_to_str(a_cpu.dtype),
                            shape=tuple(int(d) for d in a_cpu.shape),
                            data=array_to_bytes(a_cpu),
                        )
                    )
                if blobs:
                    write_arrays_state(stream, layer_idx, blobs)
    return tokens_sent


def chunk_to_mlx_nhd(chunk: KVChunk) -> tuple[mx.array, mx.array]:
    shape = chunk.shape
    return (
        bytes_to_array(chunk.keys, shape, chunk.dtype),
        bytes_to_array(chunk.values, shape, chunk.dtype),
    )


def blob_to_mlx(blob: TensorBlob) -> mx.array:
    return bytes_to_array(blob.data, blob.shape, blob.dtype)


def inject_kv_chunk(
    cache: KVCache,
    keys_nhd: mx.array,
    values_nhd: mx.array,
    offset: int,
    *,
    start_pos: int = 0,
    existing_k: mx.array | None = None,
    existing_v: mx.array | None = None,
) -> None:
    k_bhsd = nhd_to_bhsd(keys_nhd)
    v_bhsd = nhd_to_bhsd(values_nhd)
    if start_pos > 0 and existing_k is not None and existing_v is not None:
        cache.keys = mx.concatenate([existing_k[:, :, :start_pos, :], k_bhsd], axis=2)
        cache.values = mx.concatenate([existing_v[:, :, :start_pos, :], v_bhsd], axis=2)
    else:
        cache.keys = k_bhsd
        cache.values = v_bhsd
    cache.offset = offset


def inject_rotating_kv_chunk(
    cache: RotatingKVCache,
    keys_nhd: mx.array,
    values_nhd: mx.array,
    offset: int,
) -> None:
    k_bhsd = nhd_to_bhsd(keys_nhd)
    v_bhsd = nhd_to_bhsd(values_nhd)
    cache.keys = k_bhsd
    cache.values = v_bhsd
    cache.offset = offset
    cache._idx = int(k_bhsd.shape[2])


def inject_arrays_cache(cache: ArraysCache, blobs: Sequence[TensorBlob | None]) -> None:
    cache.state = [blob_to_mlx(b) if b is not None else None for b in blobs]


def write_cache_to_wire(
    wfile: BinaryIO,
    cache: KVCacheType,
    *,
    request_id: str = "",
    model_id: str = "",
    start_pos: int = 0,
) -> int:
    dtype = wire_dtype_from_cache(cache)
    write_header(
        wfile,
        Header(
            request_id=request_id,
            model_id=model_id,
            num_layers=len(cache),
            dtype=dtype,
            start_pos=start_pos,
        ),
    )
    tokens_sent = send_mlx_kv_cache(wfile, cache, dtype=dtype, start_pos=start_pos)
    write_done(wfile, tokens_sent)
    wfile.flush()
    return tokens_sent
