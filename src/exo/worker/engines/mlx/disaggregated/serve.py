import time
from typing import cast

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.cache import (
    KVPrefixCache,
    cache_length,
    make_kv_cache,
    snapshot_ssm_states,
)
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import fix_unmatched_think_end_tokens
from exo.worker.runner.bootstrap import logger


def fill_cache_prefix(model: Model, cache: KVCacheType, tokens: mx.array) -> None:
    """Evaluate only prefix state, without sampling or rolling back recurrent caches.

    The ordinary generation prefill helper reserves decode seed tokens itself.
    Calling it on an already shortened remote prefix trims twice and loses
    prompt tokens. Direct chunked forwards keep the requested boundary exact.
    """
    for start in range(0, int(tokens.size), 4096):
        _ = model(tokens[start : start + 4096][None], cache=cache)
        states: list[mx.array] = []
        for entry in cache:
            states.extend(value for value in entry.state if isinstance(value, mx.array))
        for entry in cache:
            indexer = cast(object, getattr(entry, "indexer", None))
            state = cast(object, getattr(indexer, "state", None))
            if isinstance(state, mx.array):
                states.append(state)
        mx.eval(*states)
        mx.clear_cache()


def run_prefill_for_request(
    *,
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    kv_prefix_cache: KVPrefixCache | None,
    request: PrefillRequest,
) -> KVCacheType:
    if group is not None and group.size() > 1:
        raise ValueError(
            "Remote prefill serving requires a single-rank prefill instance"
        )
    prompt_tokens = mx.array(request.token_ids)
    prompt_tokens = fix_unmatched_think_end_tokens(prompt_tokens, tokenizer)
    n_tokens = int(prompt_tokens.shape[0])
    t0 = time.perf_counter()

    matched_index: int | None = None
    prefix_hit_length = 0
    if kv_prefix_cache is not None:
        cache, remaining, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, prompt_tokens
        )
        prefix_hit_length = n_tokens - int(remaining.shape[0])
    else:
        cache = make_kv_cache(model)
        remaining = prompt_tokens

    # The request excludes the final prompt token. The decode loop consumes
    # the final two original tokens, so reserve exactly one more here.
    target_offset = max(0, n_tokens - 1)
    new_tokens = max(0, target_offset - prefix_hit_length)
    prefill_input = remaining[:new_tokens]
    if int(prefill_input.shape[0]) > 0:
        fill_cache_prefix(model, cache, prefill_input)

    if kv_prefix_cache is not None:
        try:
            cache_snapshots = [snapshot_ssm_states(cache)]
            hit_ratio = prefix_hit_length / n_tokens if n_tokens > 0 else 0.0
            if matched_index is not None and hit_ratio >= 0.5:
                kv_prefix_cache.update_kv_cache(
                    matched_index,
                    prompt_tokens,
                    cache,
                    cache_snapshots,
                    restore_pos=prefix_hit_length,
                )
            else:
                kv_prefix_cache.add_kv_cache(prompt_tokens, cache, cache_snapshots)
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to save prefix cache on prefill server"
            )

    elapsed = time.perf_counter() - t0
    final_offset = cache_length(cache)
    logger.info(
        f"Prefill: request_id={request.request_id} "
        f"{n_tokens} tokens (prefix_hit={prefix_hit_length}, "
        f"final_offset={final_offset}) in {elapsed * 1000:.0f}ms"
    )
    return cache
