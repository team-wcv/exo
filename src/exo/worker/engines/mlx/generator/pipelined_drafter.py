"""Pipelined speculative-decoding spec loop.

Implements :class:`PipelinedModelDrafter` -- a custom spec loop that
talks to the drafter through a :class:`DrafterTransport` (in-process,
remote, ...). The win over :class:`ModelDrafter` (which delegates to
``mlx_lm.speculative_generate_step``) is **cross-round speculation**:
while the target rank verifies round ``t``'s drafts, the drafter
speculatively starts round ``t + 1`` by predicting the would-be bonus
token and continuing for ``K`` more forwards. If the target's actual
bonus matches the drafter's predicted bonus, round ``t + 1``'s drafts
are already in hand by the time round ``t``'s verify finishes; if not,
the speculative work is rolled back and the standard non-speculative
path runs.

Apple-Silicon caveat: MLX serialises Metal command queues per device,
so the in-process overlap factor between drafter and target forwards
is ~0.1-0.3 (parallelism is bounded by memory-bandwidth contention,
not GPU saturation). The architecture's payoff scales with topology:
on a multi-machine deployment where target verify includes a network
round-trip, the speculative drafter forward fully overlaps the
network latency and the gain unlocks. Layer B of this PR ships
:class:`RemoteTransport` which is precisely that case.

Cache accounting (drafter side) -- this is the only complex bit, so
spelled out here once and referenced from the code:

  Notation: ``O`` = drafter cache offset before round ``t``'s propose.
  ``K`` = ``num_draft_tokens``.

  Round ``t`` propose, length-1 seed (partial-accept-from-prev case):
    ``forward([seed_t], K)`` -> K outputs. K forwards, each adds 1
    position. Cache offset O+K. Cache content extends with
    ``[seed_t, d_0..d_{K-2}]`` (the K-th draft d_{K-1} is the K-th
    output, *not* fed back as input).

  Round ``t`` propose, length-2 seed (full-accept-from-prev case):
    ``forward([drafts_{t-1}[-1], seed_t], K)`` -> K outputs. K forwards;
    the first has length-2 input, so cache extends by K+1.

  Speculative round ``t + 1`` (cross-round speculation):
    ``forward([drafts_t[-1]], K + 1)`` -> K+1 outputs. K+1 forwards,
    cache extends by K+1. Outputs are
    ``[d^pred_K, d^spec_0, ..., d^spec_{K-1}]``: the first is the
    drafter's prediction of bonus_t (compared against actual bonus_t
    to detect speculation hit); the rest are round t+1's drafts.
    Cache offset after speculation: O+2K+1.

  Round ``t`` accept outcomes:

    * Partial accept (``num_accepted < K_this``): drafter cache trim
      by ``max(K_this - num_accepted - 1, 0)``. If speculation was
      active, also rollback ``K + 1``. Round ``t + 1``'s propose is a
      length-1-seed call.
    * Full accept, speculation MISS (``bonus_t != d^pred_K``): rollback
      ``K + 1``. Round ``t + 1``'s propose is a length-2-seed call.
    * Full accept, speculation HIT: no rollback. Drafter cache
      offset O+2K+1, content matches what mlx_lm's ``_draft_generate``
      would produce after a length-2 first forward + K-1 length-1
      forwards in round t+1. Round ``t + 1``'s drafts come from the
      speculative outputs; round ``t + 1`` skips its own propose call.
    * Truncated last round (``K_this < K``): speculation is disabled
      because there's no round t+1 to feed.

The matching :func:`_pipelined_speculative_step` enforces this
accounting; any divergence between the comments above and the code is
a bug, please flag it.
"""

from __future__ import annotations

import time
from typing import Callable, Generator, Sequence, cast, final

import mlx.core as mx
from mlx_lm.generate import GenerationResponse
from mlx_lm.models.cache import trim_prompt_cache as mlx_trim_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.generator.drafter import DraftMode
from exo.worker.engines.mlx.generator.drafter_transport import (
    DrafterTransport,
    DraftFuture,
)
from exo.worker.engines.mlx.types import KVCacheType, Model


def _get_eos_ids(tokenizer: TokenizerWrapper) -> list[int]:
    eos: list[int] | None = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return []
    return eos


def _process_logits_for_position(
    raw_logits: mx.array,
    prev_tokens: mx.array,
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
) -> mx.array:
    """Apply logits processors and convert to logprobs (single position)."""
    out = raw_logits
    for proc in logits_processors:
        out = proc(prev_tokens, out)
    return out - mx.logsumexp(out, axis=-1, keepdims=True)


@final
class PipelinedModelDrafter:
    """Speculative decoding via a drafter accessed through :class:`DrafterTransport`.

    Owns its own spec loop so the drafter can be remote (different MLX
    rank) without the target rank loading the drafter model. The
    transport-agnostic propose/trim primitives mean swapping
    in-process for remote drafter placement is a one-line construction
    change at :func:`make_drafter`; the spec loop is unaffected.
    """

    def __init__(
        self,
        *,
        transport: DrafterTransport,
        num_draft_tokens: int,
    ) -> None:
        if num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {num_draft_tokens}")
        if num_draft_tokens > transport.num_draft_tokens:
            raise ValueError(
                f"num_draft_tokens ({num_draft_tokens}) exceeds transport's "
                f"max ({transport.num_draft_tokens})"
            )
        self._transport = transport
        self._num_draft_tokens = num_draft_tokens

    @property
    def mode(self) -> DraftMode:
        return "pipelined"

    @property
    def num_draft_tokens(self) -> int:
        return self._num_draft_tokens

    def stream(
        self,
        *,
        model: Model,
        tokenizer: TokenizerWrapper,
        prompt: mx.array,
        context_tokens: Sequence[int],
        prompt_cache: KVCacheType,
        max_tokens: int,
        sampler: Callable[[mx.array], mx.array],
        logits_processors: Sequence[Callable[[mx.array, mx.array], mx.array]],
        prefill_step_size: int = 1,
    ) -> Generator[GenerationResponse, None, None]:
        yield from _pipelined_stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            context_tokens=list(context_tokens),
            prompt_cache=prompt_cache,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=list(logits_processors),
            transport=self._transport,
            num_draft_tokens=self._num_draft_tokens,
            prefill_step_size=prefill_step_size,
        )

    def shutdown(self) -> None:
        """Release transport resources."""
        self._transport.shutdown()


def _pipelined_stream_generate(
    *,
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt: mx.array,
    context_tokens: list[int],
    prompt_cache: KVCacheType,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
    transport: DrafterTransport,
    num_draft_tokens: int,
    prefill_step_size: int,
) -> Generator[GenerationResponse, None, None]:
    """Mirror of ``mlx_lm.stream_generate`` framing for the pipelined drafter.

    The framing (detokenisation, tps tracking, finish reasons) matches
    :func:`exo.worker.engines.mlx.generator.drafter._ngram_stream_generate`
    so the call site in ``mlx_generate`` doesn't branch on drafter type.
    """
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()  # type: ignore[reportUnknownMemberType]
    eos_ids = _get_eos_ids(tokenizer)

    token_iter = _pipelined_speculative_step(
        prompt=prompt,
        model=model,
        transport=transport,
        prompt_cache=prompt_cache,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        num_draft_tokens=num_draft_tokens,
        prefill_step_size=prefill_step_size,
        prompt_token_count=len(context_tokens),
    )

    prompt_size = len(context_tokens)
    tic = time.perf_counter()
    prompt_tps = 0.0
    n = -1
    token = 0
    logprobs = mx.zeros((1,))
    from_draft = False
    finish_reason: str | None = None
    for n, (token, logprobs, from_draft) in enumerate(token_iter):
        if n == 0:
            prompt_time = time.perf_counter() - tic
            prompt_tps = prompt_size / prompt_time if prompt_time > 0 else 0.0
            tic = time.perf_counter()
        if token in eos_ids:
            finish_reason = "stop"
            break
        detokenizer.add_token(token)  # type: ignore[reportUnknownMemberType]
        if (n + 1) == max_tokens:
            finish_reason = "length"
            break
        elapsed = time.perf_counter() - tic
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=from_draft,
            prompt_tokens=prompt_size,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / elapsed if elapsed > 0 else 0.0,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=None,
        )

    detokenizer.finalize()  # type: ignore[reportUnknownMemberType]
    elapsed = time.perf_counter() - tic
    yield GenerationResponse(
        text=detokenizer.last_segment,
        token=token,
        logprobs=logprobs,
        from_draft=from_draft,
        prompt_tokens=prompt_size,
        prompt_tps=prompt_tps,
        generation_tokens=n + 1 if n >= 0 else 0,
        generation_tps=(n + 1) / elapsed if elapsed > 0 and n >= 0 else 0.0,
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason=finish_reason or ("stop" if token in eos_ids else "length"),
    )


def _pipelined_speculative_step(
    *,
    prompt: mx.array,
    model: Model,
    transport: DrafterTransport,
    prompt_cache: KVCacheType,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
    num_draft_tokens: int,
    prefill_step_size: int,
    prompt_token_count: int,
) -> Generator[tuple[int, mx.array, bool], None, None]:
    """Cross-round speculative decoding loop using ``transport``.

    See module docstring for the cache-accounting derivation. This
    function maintains:

      * ``drafts``: list[int] of length K_this -- this round's drafts.
      * ``seed``: int -- the seed token for this round (target verify
        consumes ``[seed, *drafts]``).
      * ``next_round_inputs``: list[int] -- input shape for next round's
        propose call (length 1 for partial-accept-from-this, length 2
        for full-accept-from-this).
      * ``speculative_future``: optional Future from a speculative
        forward issued in parallel with target verify. ``None`` when
        speculation is not in flight.

    ``prompt_token_count`` is captured so logits processors that need
    the running token count (rare, e.g. positional repetition penalty
    that scales with absolute position) get accurate values.
    """
    k = num_draft_tokens
    y = prompt.astype(mx.uint32)

    # Mirror mlx_lm._prefill: caller has aligned ``prompt_cache`` to
    # ``context_tokens[:-2]`` via ``exo.prefill`` + ``trim(2)``; this loop
    # advances the cache by one more token, leaving ``y`` (length 1) as
    # the seed for the spec loop.
    while y.size > 1:
        n_to_process = min(prefill_step_size, y.size - 1)
        model(y[:n_to_process][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])  # type: ignore[reportArgumentType]
        y = y[n_to_process:]
        mx.clear_cache()

    seed = int(y.item())
    # ``prev_tokens`` carries the running token sequence (prompt +
    # emitted) so logits processors with state see consistent context.
    # Mirror :func:`drafter._ngram_speculative_step`: start from prompt.
    prev_tokens = mx.array([seed], dtype=mx.uint32)
    del prompt_token_count  # currently unused; kept for forward-compat

    # Round 0 propose: synchronous, no speculation possible yet because
    # we don't have prior drafts to chain off of.
    drafts_future = transport.forward([seed], k)
    drafts = drafts_future.result()

    speculative_future: DraftFuture | None = None
    ntoks = 0

    while ntoks < max_tokens:
        budget = max_tokens - ntoks
        k_this = min(k, len(drafts), budget)
        if k_this < 1:
            break
        drafts = drafts[:k_this]

        # ----- Cross-round speculation: dispatch in parallel with verify -----
        # Speculate only when:
        #   * full k_this drafts (truncated last rounds have no t+1 to feed),
        #   * budget remains for an entire next round's verify after this one.
        #
        # The speculative forward consumes ``drafts[-1]`` (= drafter's last
        # draft this round) as its first input, doing k+1 forwards. The
        # first output is the drafter's prediction of bonus_t (used to
        # detect speculation hit); the remaining k outputs are round
        # t+1's drafts if speculation hits.
        speculation_active = (
            k_this == k
            and ntoks + (k_this + 1) + k + 1 <= max_tokens
            and speculative_future is None
        )
        if speculation_active:
            speculative_future = transport.forward([drafts[-1]], k + 1)

        # ----- Target verify -----
        seed_arr = mx.array([seed], dtype=mx.uint32)
        draft_arr = mx.array(drafts, dtype=mx.uint32)
        verify_input = mx.concatenate([seed_arr, draft_arr])
        logits = model(verify_input[None], cache=prompt_cache)
        # logits shape: (1, k_this + 1, vocab)

        target_logprobs: list[mx.array]
        target_tokens: list[int]
        if not logits_processors:
            # Fast path: no per-position state. All K+1 sampling decisions
            # are independent, so we batch them into one sampler call and
            # one host-device sync. On a target with ~10ms step time this
            # saves K sync points per round (~2-3ms each), which is the
            # difference between spec-decode net-win and net-loss.
            batched_logits = logits.squeeze(0)
            batched_logprobs = batched_logits - mx.logsumexp(
                batched_logits, axis=-1, keepdims=True
            )
            sampled_batch = sampler(batched_logprobs)
            mx.eval(sampled_batch)
            target_tokens = [int(t) for t in sampled_batch.tolist()]  # type: ignore[reportUnknownArgumentType]
            target_logprobs = [batched_logprobs[i] for i in range(k_this + 1)]
        else:
            # Stateful path: logits processors (e.g. repetition penalty)
            # depend on ``running_prev`` which only resolves between
            # positions, so we can't batch. Per-position sync is the
            # cost of correctness here.
            target_logprobs = []
            target_tokens = []
            running_prev = prev_tokens
            for i in range(k_this + 1):
                position_logits = logits[:, i, :].squeeze(0)
                position_logprobs = _process_logits_for_position(
                    position_logits, running_prev, logits_processors
                )
                sampled = sampler(position_logprobs)
                mx.eval(sampled)
                sampled_token = int(sampled.item())
                target_logprobs.append(position_logprobs)
                target_tokens.append(sampled_token)
                running_prev = mx.concatenate(
                    [running_prev, mx.array([sampled_token], dtype=mx.uint32)]
                )

        # ----- Greedy accept loop -----
        num_accepted = 0
        for i in range(k_this):
            if target_tokens[i] == drafts[i]:
                num_accepted += 1
            else:
                break

        # ----- Emit accepted drafts + correction/bonus -----
        emit_count = num_accepted + 1
        for j in range(emit_count):
            tok = drafts[j] if j < num_accepted else target_tokens[j]
            from_draft = j < num_accepted
            yield tok, target_logprobs[j], from_draft
            prev_tokens = mx.concatenate(
                [prev_tokens, mx.array([tok], dtype=mx.uint32)]
            )
            ntoks += 1
            if ntoks >= max_tokens:
                break

        # ----- Target cache trim (rejected draft positions) -----
        # Verify forward extended target cache by k_this + 1; we keep
        # ``num_accepted + 1`` of those (= emit_count) so trim
        # ``k_this - num_accepted``.
        target_trim = k_this - num_accepted
        if target_trim > 0:
            mlx_trim_prompt_cache(cast(list[object], prompt_cache), target_trim)  # type: ignore[reportArgumentType]

        if ntoks >= max_tokens:
            # Discard any in-flight speculation; we're done. Rolling back
            # the drafter cache isn't strictly necessary (the loop is
            # exiting), but keeps the cache in a consistent state for
            # any subsequent runs that might reuse the transport.
            if speculative_future is not None:
                _drain_future(speculative_future)
                transport.trim_cache(k + 1)
                speculative_future = None
            break

        # ``next_seed`` is the target's chosen token at the rejection
        # point (partial accept) or the bonus position (full accept).
        next_seed: int
        if num_accepted == k_this:
            next_seed = target_tokens[k_this]
        else:
            next_seed = target_tokens[num_accepted]

        # ----- Drafter cache reconciliation + next-round setup -----
        next_drafts: list[int]
        next_round_inputs: list[int]

        if num_accepted < k_this:
            # Partial accept (regardless of speculation state).
            drafter_trim_partial = max(k_this - num_accepted - 1, 0)
            if speculative_future is not None:
                # Speculative work is bound to a different (assumed-full-
                # accept) future; discard it and trim its k+1 positions
                # plus the partial-accept trim.
                _drain_future(speculative_future)
                transport.trim_cache(k + 1 + drafter_trim_partial)
                speculative_future = None
            elif drafter_trim_partial > 0:
                transport.trim_cache(drafter_trim_partial)
            next_round_inputs = [next_seed]
            next_drafts = transport.forward(next_round_inputs, k).result()
        else:
            # Full accept at this round.
            if speculative_future is not None:
                spec_outputs = speculative_future.result()
                speculative_future = None
                bonus_predicted = spec_outputs[0]
                if bonus_predicted == next_seed:
                    # SPECULATION HIT. Round t+1's drafts come for free.
                    # Drafter cache state is correct (offset O+2k+1
                    # matches what a length-2-seed propose for round
                    # t+1 would produce).
                    next_drafts = spec_outputs[1 : k + 1]
                    next_round_inputs = []  # unused, drafts already in hand
                else:
                    # SPECULATION MISS. Rollback the k+1 speculative
                    # positions and run a standard length-2-seed propose
                    # for round t+1.
                    transport.trim_cache(k + 1)
                    next_round_inputs = [drafts[-1], next_seed]
                    next_drafts = transport.forward(next_round_inputs, k).result()
            else:
                # Full accept, speculation was inactive. Standard length-
                # 2-seed propose for round t+1.
                next_round_inputs = [drafts[-1], next_seed]
                next_drafts = transport.forward(next_round_inputs, k).result()

        seed = next_seed
        drafts = next_drafts


def _drain_future(future: DraftFuture) -> None:
    """Block on ``future`` and discard its result.

    Used when speculation misses or the loop exits early: the drafter
    forwards have already executed; we just need to ensure the future
    is resolved before issuing dependent transport operations
    (``trim_cache``, ``shutdown``). Exceptions from the forwards
    surface elsewhere (transport's own error path); we suppress them
    here to avoid double-reporting.
    """
    import contextlib

    with contextlib.suppress(Exception):
        future.result()


__all__ = ["PipelinedModelDrafter"]
