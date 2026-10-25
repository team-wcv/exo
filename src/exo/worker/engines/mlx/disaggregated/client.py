import socket
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import BinaryIO, Protocol, cast

import mlx.core as mx
from loguru import logger
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.worker.disaggregated.protocol import (
    ArraysState,
    Done,
    Header,
    IndexerState,
    KVChunk,
    TensorBlob,
    read_header,
    read_message,
)
from exo.worker.disaggregated.server import PrefillRequest, write_request
from exo.worker.engines.mlx.disaggregated.adapter import (
    blob_to_mlx,
    chunk_to_mlx_nhd,
    inject_kv_chunk,
    inject_rotating_kv_chunk,
)
from exo.worker.engines.mlx.disaggregated.cache_projection import (
    CacheLayout,
    project_kv_heads,
    project_linear_state,
)

_CONNECT_TIMEOUT_SECS = 5
_SOCKET_TIMEOUT_SECS = 300
_RECV_BUFFER_BYTES = 4 * 1024 * 1024


class _IndexerCache(Protocol):
    state: mx.array


@dataclass
class PrefillResult:
    header: Header
    kv_chunks: dict[int, list[KVChunk]] = field(
        default_factory=dict[int, list[KVChunk]]
    )
    arrays: dict[int, list[TensorBlob | None]] = field(
        default_factory=dict[int, list[TensorBlob | None]]
    )
    indexers: dict[int, TensorBlob] = field(default_factory=dict[int, TensorBlob])
    total_tokens: int = 0


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    if ":" in endpoint:
        host, port_str = endpoint.rsplit(":", 1)
        return host, int(port_str)
    raise ValueError(f"Invalid endpoint {endpoint}")


def _connect_prefill(host: str, port: int, timeout_secs: float) -> socket.socket:
    # Connecting to a missing prefill host should fall back quickly, but the
    # first response header is emitted only after the remote model finishes
    # prefill. Large prompts on slower accelerators can legitimately exceed a
    # minute before sending that header.
    sock = socket.create_connection(
        (host, port), timeout=min(timeout_secs, _CONNECT_TIMEOUT_SECS)
    )
    sock.settimeout(timeout_secs)
    return sock


def remote_prefill_fetch(
    endpoint: str,
    request: PrefillRequest,
    on_header: Callable[[Header], None] | None = None,
    on_kv_chunk: Callable[[KVChunk, int], None] | None = None,
    timeout_secs: float = _SOCKET_TIMEOUT_SECS,
) -> PrefillResult:
    host, port = _parse_endpoint(endpoint)
    logger.info(
        f"Connecting to prefill server at {host}:{port} "
        f"({len(request.token_ids)} tokens, start_pos={request.start_pos})"
    )

    sock = _connect_prefill(host, port, timeout_secs)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RECV_BUFFER_BYTES)
    try:
        wfile = sock.makefile("wb", buffering=256 * 1024)
        wstream: BinaryIO = cast(BinaryIO, cast(object, wfile))
        write_request(wstream, request)

        raw_stream = sock.makefile("rb", buffering=256 * 1024)
        stream: BinaryIO = cast(BinaryIO, cast(object, raw_stream))

        header = read_header(stream)
        if on_header is not None:
            on_header(header)

        result = PrefillResult(header=header)
        kv_by_layer: dict[int, list[KVChunk]] = defaultdict(list)
        chunks_received = 0

        while True:
            msg = read_message(stream)
            if msg is None:
                raise ConnectionError(
                    "Prefill stream ended before its completion frame"
                )
            if isinstance(msg, KVChunk):
                kv_by_layer[msg.layer_idx].append(msg)
                chunks_received += 1
                if on_kv_chunk is not None:
                    on_kv_chunk(msg, chunks_received)
            elif isinstance(msg, ArraysState):
                result.arrays[msg.layer_idx] = msg.arrays
            elif isinstance(msg, IndexerState):
                result.indexers[msg.layer_idx] = msg.tensor
            elif isinstance(msg, Done):
                result.total_tokens = msg.total_tokens
                break
            else:
                raise RuntimeError(f"Prefill server error [{msg.code}]: {msg.message}")

        if (header.request_id, header.model_id, header.start_pos) != (
            request.request_id,
            request.model_id,
            request.start_pos,
        ):
            raise ValueError(
                "Prefill response does not match the requested model, ID, or prefix"
            )
        result.kv_chunks = dict(kv_by_layer)
        return result
    finally:
        sock.close()


def ingest_into_mlx_cache(
    result: PrefillResult,
    caches: list[KVCache | RotatingKVCache | ArraysCache],
    *,
    start_pos: int = 0,
    layouts: Sequence[CacheLayout] | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> int:
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(
            f"Invalid remote cache shard rank={rank}, world_size={world_size}"
        )
    if result.header.num_layers != len(caches):
        raise ValueError(
            f"Remote prefill has {result.header.num_layers} layers, "
            f"but the decoder has {len(caches)} caches"
        )
    if world_size > 1 and (layouts is None or len(layouts) != len(caches)):
        raise ValueError("Remote tensor cache import requires every local layer layout")
    if start_pos < 0 or result.header.start_pos != start_pos:
        raise ValueError("Remote prefill prefix offset does not match decoder offset")
    layer_ids = set(result.kv_chunks) | set(result.arrays) | set(result.indexers)
    if layer_ids != set(range(len(caches))):
        raise ValueError("Remote prefill must supply exactly the decoder layer indices")
    for i, cache in enumerate(caches):
        if isinstance(cache, ArraysCache):
            if i not in result.arrays or i in result.kv_chunks:
                raise ValueError(f"Remote recurrent layer {i} is missing array state")
        elif i not in result.kv_chunks or not result.kv_chunks[i]:
            raise ValueError(f"Remote attention layer {i} is missing KV state")
        if (
            isinstance(cache, KVCache)
            and start_pos > 0
            and (cache.keys is None or cache.values is None or cache.offset < start_pos)
        ):
            raise ValueError(f"Decoder layer {i} lacks the requested cached prefix")
    full_lengths = [
        sum(chunk.num_tokens for chunk in result.kv_chunks[i])
        for i, cache in enumerate(caches)
        if isinstance(cache, KVCache)
    ]
    # Full-history layers anchor the logical prefix length when present. Sliding
    # layers carry complete retained windows even for incremental transfers, so
    # validate their physical lengths against the final offset below instead.
    # The caller also checks the logical boundary against its requested prefix.
    if (
        result.total_tokens < 0
        or (
            full_lengths
            and any(length != result.total_tokens for length in full_lengths)
        )
        or (not result.kv_chunks and result.total_tokens != 0)
    ):
        raise ValueError(
            "Remote prefill completion count differs from received KV tokens"
        )
    final_offset = start_pos + result.total_tokens
    for i, chunks in result.kv_chunks.items():
        cache = caches[i]
        expected = (
            min(final_offset, int(cache.max_size))
            if isinstance(cache, RotatingKVCache)
            else result.total_tokens
        )
        if sum(chunk.num_tokens for chunk in chunks) != expected:
            raise ValueError(
                f"Remote attention layer {i} has an incomplete token sequence: "
                f"expected {expected} retained tokens"
            )

    # Validate and materialize every layer before replacing any live cache. A
    # malformed transfer must not leave a mixture of old and imported state.
    prepared_kv: dict[int, tuple[mx.array, mx.array]] = {}
    prepared_arrays: dict[int, list[mx.array | None]] = {}
    prepared_indexers: dict[int, mx.array] = {}

    for i, cache in enumerate(caches):
        if i in result.kv_chunks:
            chunks = result.kv_chunks[i]
            if len(chunks) == 1:
                k_nhd, v_nhd = chunk_to_mlx_nhd(chunks[0])
            else:
                decoded = [chunk_to_mlx_nhd(c) for c in chunks]
                k_nhd = mx.concatenate([k for k, _ in decoded], axis=0)
                v_nhd = mx.concatenate([v for _, v in decoded], axis=0)

            if world_size > 1:
                assert layouts is not None
                k_nhd = project_kv_heads(k_nhd, layouts[i].kv_heads, rank, world_size)
                v_nhd = project_kv_heads(v_nhd, layouts[i].kv_heads, rank, world_size)
            if not isinstance(cache, (KVCache, RotatingKVCache)):
                raise ValueError(f"Remote KV layer {i} targets {type(cache).__name__}")
            prepared_kv[i] = (k_nhd, v_nhd)
            indexer: object = getattr(cache, "indexer", None)
            if indexer is not None and i not in result.indexers:
                raise ValueError(
                    f"Remote sparse-attention layer {i} is missing indexer state"
                )

        if i in result.arrays:
            if not isinstance(cache, ArraysCache):
                raise ValueError(
                    f"Remote array layer {i} targets {type(cache).__name__}"
                )
            state = [
                blob_to_mlx(b) if b is not None else None for b in result.arrays[i]
            ]
            if world_size > 1:
                assert layouts is not None
                state = project_linear_state(state, layouts[i], rank, world_size)
            prepared_arrays[i] = state

        if i in result.indexers:
            indexer = getattr(cache, "indexer", None)
            if indexer is None:
                raise ValueError(f"Remote indexer layer {i} has no decoder indexer")
            keys = blob_to_mlx(result.indexers[i])
            if keys.ndim < 2 or keys.shape[1] != final_offset:
                raise ValueError(
                    f"Remote indexer layer {i} has shape {keys.shape}, "
                    f"expected token length {final_offset}"
                )
            prepared_indexers[i] = keys

    for keys, values in prepared_kv.values():
        mx.eval(keys, values)
    for state in prepared_arrays.values():
        mx.eval([value for value in state if value is not None])
    mx.eval(list(prepared_indexers.values()))

    for i, cache in enumerate(caches):
        if i in prepared_kv:
            k_nhd, v_nhd = prepared_kv[i]

            if isinstance(cache, RotatingKVCache):
                inject_rotating_kv_chunk(cache, k_nhd, v_nhd, final_offset)
            elif isinstance(cache, KVCache):
                if start_pos > 0:
                    inject_kv_chunk(
                        cache,
                        k_nhd,
                        v_nhd,
                        final_offset,
                        start_pos=start_pos,
                        existing_k=cache.keys,
                        existing_v=cache.values,
                    )
                else:
                    inject_kv_chunk(cache, k_nhd, v_nhd, final_offset)

        if i in prepared_arrays:
            assert isinstance(cache, ArraysCache)
            cache.state = prepared_arrays[i]
        if i in prepared_indexers:
            indexer = cast(_IndexerCache, cast(object, getattr(cache, "indexer", None)))
            indexer.state = prepared_indexers[i]

    return final_offset
