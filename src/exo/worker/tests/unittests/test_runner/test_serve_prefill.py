"""Tests for serve_prefill_request — the producer-side path that uses
KVPrefixCache for cross-request prefix sharing."""

import io
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

import exo.worker.engines.mlx.disaggregated.serve as mlx_serve_mod
from exo.worker.disaggregated.protocol import (
    Done,
    Header,
    KVChunk,
    read_header,
    read_message,
)
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.disaggregated.adapter import write_cache_to_wire

N_HEADS = 2
HEAD_DIM = 4


@dataclass
class _FakeTokenizer:
    has_thinking: bool = False
    think_start: object | None = None
    think_end: object | None = None


class _FakeModel:
    def __init__(self) -> None:
        self.layers = [object()]

    def __call__(self, tokens: mx.array, cache: list[KVCache]) -> mx.array:
        # Encoding tokens as KV makes missing or duplicated prefix tokens observable.
        state = tokens.astype(mx.float32)[:, None, :, None]
        for entry in cache:
            entry.update_and_fetch(state, state)
        return state

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]


def _populate_cache_in_place(cache: list[KVCache], n_tokens: int) -> None:
    mx.random.seed(0)
    for c in cache:
        c.keys = (
            mx.random.uniform(shape=(1, N_HEADS, n_tokens, HEAD_DIM)) * 10
        ).astype(mx.bfloat16)
        c.values = (
            mx.random.uniform(shape=(1, N_HEADS, n_tokens, HEAD_DIM)) * 10
        ).astype(mx.bfloat16)
        c.offset = n_tokens


def _patch_prefill(monkeypatch: pytest.MonkeyPatch) -> list[mx.array]:
    """Replace mlx_prefill with a tracker; return the list it appends to."""
    inputs: list[mx.array] = []

    def fake_prefill(_model: object, raw_cache: object, pt: mx.array) -> None:
        inputs.append(pt)
        n = int(pt.shape[0])
        cache = cast(list[KVCache], raw_cache)
        existing = int(cache[0].offset) if cache and cache[0].keys is not None else 0
        _populate_cache_in_place(cache, existing + n)

    monkeypatch.setattr(mlx_serve_mod, "fill_cache_prefix", fake_prefill)
    return inputs


def _decode(payload: bytes) -> tuple[Header, list[KVChunk], int]:
    buf = io.BytesIO(payload)
    hdr = read_header(buf)
    chunks: list[KVChunk] = []
    total = 0
    while True:
        msg = read_message(buf)
        if msg is None:
            break
        if isinstance(msg, KVChunk):
            chunks.append(msg)
        elif isinstance(msg, Done):
            total = msg.total_tokens
            break
    return hdr, chunks, total


def _serve(request: PrefillRequest, kv_prefix_cache: KVPrefixCache | None) -> bytes:
    cache = mlx_serve_mod.run_prefill_for_request(
        model=cast(Any, _FakeModel()),  # pyright: ignore[reportAny]
        tokenizer=cast(Any, _FakeTokenizer()),  # pyright: ignore[reportAny]
        group=None,
        kv_prefix_cache=kv_prefix_cache,
        request=request,
    )
    buf = io.BytesIO()
    write_cache_to_wire(
        buf,
        cache,
        request_id=request.request_id,
        model_id=request.model_id,
        start_pos=request.start_pos,
    )
    return buf.getvalue()


def test_serve_prefill_runs_full_prefill_when_cache_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _patch_prefill(monkeypatch)
    cache = KVPrefixCache(group=None)

    payload = _serve(
        PrefillRequest(
            request_id="r1", model_id="m", token_ids=list(range(20)), start_pos=0
        ),
        cache,
    )

    assert len(inputs) == 1
    assert int(inputs[0].shape[0]) == 19

    hdr, chunks, total = _decode(payload)
    assert hdr.start_pos == 0
    assert total == 19
    assert len(chunks) == 1
    assert chunks[0].num_tokens == 19

    assert len(cache.prompts) == 1
    assert int(cache.prompts[0].shape[0]) == 20


def test_serve_prefill_skips_work_on_exact_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _patch_prefill(monkeypatch)
    cache = KVPrefixCache(group=None)
    tokens = list(range(20))

    _serve(
        PrefillRequest(request_id="r1", model_id="m", token_ids=tokens, start_pos=0),
        cache,
    )
    inputs.clear()

    _serve(
        PrefillRequest(request_id="r2", model_id="m", token_ids=tokens, start_pos=0),
        cache,
    )

    assert inputs == []


def test_serve_prefill_only_runs_suffix_on_partial_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _patch_prefill(monkeypatch)
    cache = KVPrefixCache(group=None)

    base = list(range(15))
    _serve(
        PrefillRequest(request_id="r1", model_id="m", token_ids=base, start_pos=0),
        cache,
    )
    inputs.clear()

    extended = base + [99, 100, 101, 102, 103]
    _serve(
        PrefillRequest(request_id="r2", model_id="m", token_ids=extended, start_pos=0),
        cache,
    )

    assert len(inputs) == 1
    suffix_len = int(inputs[0].shape[0])
    assert suffix_len < len(extended) - 1


def test_serve_prefill_slices_payload_at_client_start_pos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prefill(monkeypatch)
    cache = KVPrefixCache(group=None)

    n_tokens = 20
    client_has = 12

    payload = _serve(
        PrefillRequest(
            request_id="r1",
            model_id="m",
            token_ids=list(range(n_tokens)),
            start_pos=client_has,
        ),
        cache,
    )

    hdr, chunks, total = _decode(payload)
    assert hdr.start_pos == client_has
    expected_sent = (n_tokens - 1) - client_has
    assert total == expected_sent
    assert len(chunks) == 1
    assert chunks[0].num_tokens == expected_sent


def test_serve_prefill_works_without_prefix_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _patch_prefill(monkeypatch)

    payload = _serve(
        PrefillRequest(
            request_id="r1", model_id="m", token_ids=list(range(20)), start_pos=0
        ),
        None,
    )

    assert len(inputs) == 1
    assert int(inputs[0].shape[0]) == 19

    _, _, total = _decode(payload)
    assert total == 19


@pytest.mark.parametrize("length", [20, 4100])
def test_remote_prefix_preserves_every_token_before_decode_seed(length: int) -> None:
    tokens = list(range(length))
    payload = _serve(
        PrefillRequest(request_id="boundary", model_id="m", token_ids=tokens), None
    )
    _, chunks, total = _decode(payload)
    from exo.worker.engines.mlx.disaggregated.adapter import chunk_to_mlx_nhd

    keys, _ = chunk_to_mlx_nhd(chunks[0])
    assert total == length - 1
    assert keys.reshape(-1).tolist() == tokens[:-1]
