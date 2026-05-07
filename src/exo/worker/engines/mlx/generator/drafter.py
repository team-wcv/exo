"""Drafting strategies for speculative decoding.

The mlx engine has historically supported one drafting mode: a smaller
"drafter" model paired with the target via
``mlx_lm.speculative_generate_step``. That mode (``DraftMode = "model"``)
is the right call for distributed pipeline-parallel runs, where every
generated token pays cross-device communication latency that the
drafter - sitting on a single device - amortises across many tokens.
On fast single-device inference (e.g. Mac Studio M3 Ultra + 4-bit 26B
target at ~76 tok/s), generation is memory-bandwidth-bound and the
``K + 1``-token verify forward costs nearly ``K + 1`` times a
single-token forward; speculative decoding only wins when the
acceptance fraction clears ``K / (K + 1)``, which most workloads don't.
Empirical measurements on that hardware show:

  * model-drafter spec is a net loss across every workload class
    (-25% to -45% tps), even at 65-75% acceptance.
  * n-gram spec is roughly parity on echo-shaped prompts (-0.5%) and
    a 20-30% loss on novel content where suffix matches are weak.

The right call there is ``DraftMode = "none"`` (the default).
``"ngram"`` and ``"model"`` are exposed for slower-target regimes
(distributed inference, larger FP16 models, ASIC-bound targets) where
their economics flip: opt-in via ``EXO_DRAFT_MODE`` env var or per-
request ``TaskParams.draft_mode``.

This module exposes a small ``Drafter`` protocol so ``mlx_generate`` can
dispatch on mode without sprouting branches everywhere, plus three
concrete implementations:

* :class:`NoSpecDrafter` — pass-through to ``mlx_lm.stream_generate``.
* :class:`ModelDrafter` — wraps ``mlx_lm.stream_generate(draft_model=...)``.
* :class:`NgramDrafter` — owns its own spec loop; proposes draft tokens
  by suffix-matching the running context against itself.

The protocol intentionally lives at the *stream factory* level (not at a
finer-grained ``propose / accept`` level), so the well-tested upstream
spec loop keeps owning the model-drafter path. Future additions
(EAGLE/Medusa heads, lookahead with n-gram + Jacobi, drafter-on-other-
device) plug in by adding a new concrete drafter that yields
``GenerationResponse`` the same way ``stream_generate`` does.
"""

from __future__ import annotations

import os
import time
from typing import (
    Callable,
    Final,
    Generator,
    Literal,
    Protocol,
    Sequence,
    cast,
    final,
    runtime_checkable,
)

import mlx.core as mx
from mlx_lm.generate import GenerationResponse, stream_generate
from mlx_lm.models.cache import trim_prompt_cache as mlx_trim_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.constants import KV_BITS, KV_GROUP_SIZE
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.runner.bootstrap import logger


def _get_eos_ids(tokenizer: TokenizerWrapper) -> list[int]:
    """Tokenizer-agnostic EOS lookup matching ``eos_ids_from_tokenizer``."""
    eos: list[int] | None = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return []
    return eos


DraftMode = Literal["model", "pipelined", "ngram", "none"]
"""How to source draft tokens for speculative decoding.

* ``"model"``: small distilled drafter (e.g. Gemma-4 e2b/e4b) via
  ``mlx_lm.speculative_generate_step``. Best for slow targets and
  distributed pipeline-parallel where token latency is dominated by
  cross-device communication. On fast single-device inference this is
  frequently a net loss; benchmark before defaulting to it.
* ``"pipelined"``: same drafter model as ``"model"``, but routed
  through :class:`exo.worker.engines.mlx.generator.pipelined_drafter
  .PipelinedModelDrafter` -- a custom spec loop with cross-round
  speculation (drafter forward for round ``t + 1`` overlaps target
  verify of round ``t``). The transport layer (in-process or remote)
  is selected by ``EXO_DRAFTER_TRANSPORT``; remote (RDMA/TCP via
  ``mx.distributed.send/recv``) is the regime where the pipelining
  win is unambiguous.
* ``"ngram"``: propose drafts by matching the longest suffix of the
  running token context against earlier positions in the same context.
  Zero drafter compute, no extra KV cache, no warmup. Wins on prompts
  the model echoes (RAG, summarisation, structured/code output);
  gracefully degrades to baseline when no match is found.
* ``"none"``: standard non-speculative generation.
"""

ALL_DRAFT_MODES: Final[tuple[DraftMode, ...]] = (
    "model",
    "pipelined",
    "ngram",
    "none",
)

EXO_DRAFT_MODE_ENV: Final[str] = "EXO_DRAFT_MODE"
"""Process-wide default mode. Per-request ``TaskParams`` overrides take precedence."""


def parse_draft_mode(raw: str | None, default: DraftMode) -> DraftMode:
    """Parse an ``EXO_DRAFT_MODE`` value, falling back on unknown values."""
    if raw is None:
        return default
    candidate = raw.strip().lower()
    if candidate == "model":
        return "model"
    if candidate == "pipelined":
        return "pipelined"
    if candidate == "ngram":
        return "ngram"
    if candidate == "none":
        return "none"
    logger.warning(
        f"{EXO_DRAFT_MODE_ENV}={raw!r} not in {ALL_DRAFT_MODES}; falling back to {default!r}"
    )
    return default


def resolve_draft_mode(
    *,
    has_drafter_model: bool,
    request_use_drafter: bool | None,
    request_draft_mode: DraftMode | None,
) -> DraftMode:
    """Compute the effective drafting mode for one request.

    Precedence (highest first):
      1. ``request_draft_mode`` — explicit per-request mode override.
      2. ``request_use_drafter is False`` — opt-out shortcut maps to ``"none"``.
      3. ``EXO_DRAFT_MODE`` env var if recognised.
      4. Implicit default: ``"model"`` if a drafter model was loaded,
         else ``"none"``. ``"ngram"`` and ``"pipelined"`` are opt-in;
         we don't auto-promote because their wins are topology-dependent
         (``"pipelined"``'s gain unlocks at remote-transport scale).

    A ``"model"`` or ``"pipelined"`` mode without a loaded drafter
    degrades to ``"none"`` with a warning, so misconfiguration fails
    loudly instead of silently producing the wrong throughput.
    """
    if request_draft_mode is not None:
        chosen: DraftMode = request_draft_mode
    elif request_use_drafter is False:
        chosen = "none"
    else:
        env_default: DraftMode = "model" if has_drafter_model else "none"
        chosen = parse_draft_mode(os.environ.get(EXO_DRAFT_MODE_ENV), env_default)

    if chosen in ("model", "pipelined") and not has_drafter_model:
        logger.warning(
            f"draft_mode={chosen!r} requested but no drafter model is "
            "loaded; falling back to 'none'."
        )
        return "none"
    return chosen


@runtime_checkable
class Drafter(Protocol):
    """Stream factory that runs one generation with a chosen drafting strategy.

    Concrete drafters yield :class:`mlx_lm.generate.GenerationResponse`
    identically to ``mlx_lm.stream_generate``, so the call site in
    ``mlx_generate`` doesn't change shape across modes.
    """

    @property
    def mode(self) -> DraftMode:
        """The mode this drafter implements (matches :data:`DraftMode`)."""
        ...

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
        """Generate tokens against ``model``.

        Args:
            prompt: Prefill-tail (the last 2 prompt tokens). The caller
                has pre-aligned ``prompt_cache`` to ``full_prompt[:-2]``
                via ``exo.prefill`` + ``trim(2)``; ``mlx_lm``'s
                internal ``_prefill`` advances the cache by one more
                token, and the drafter's spec loop seeds from the last.
            context_tokens: Full prompt as a list of token ids. Used by
                drafters that need the complete history for proposals
                (``NgramDrafter``); other drafters ignore it.
            prompt_cache: Target KV cache, pre-aligned per ``prompt`` above.
            max_tokens: Maximum tokens to generate (including drafter-
                accepted tokens).
            sampler: ``logprobs -> token`` sampler.
            logits_processors: Per-position logits processors (repetition
                penalty, etc.). The drafter applies them before sampling.
            prefill_step_size: Forwarded to ``mlx_lm._prefill``.
        """
        ...


@final
class NoSpecDrafter:
    """Standard non-speculative decoding via ``mlx_lm.stream_generate``."""

    @property
    def mode(self) -> DraftMode:
        return "none"

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
        del context_tokens  # only the n-gram drafter needs it
        yield from stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=list(logits_processors),
            prompt_cache=list(prompt_cache),
            prefill_step_size=prefill_step_size,
            kv_group_size=KV_GROUP_SIZE,
            kv_bits=KV_BITS,
        )


@final
class ModelDrafter:
    """Speculative decoding via a smaller distilled drafter model.

    Delegates to ``mlx_lm.stream_generate(draft_model=...)`` so the
    well-tested upstream spec loop owns the rejection sampling, cache
    trimming, and bonus-token bookkeeping. The target and drafter caches
    must already be aligned to the same offset (handled by
    ``mlx_generate`` via ``exo.prefill`` + ``_spec_drafter_prefill``).
    """

    def __init__(
        self,
        *,
        draft_model: Model,
        draft_cache: KVCacheType,
        num_draft_tokens: int,
    ) -> None:
        if num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {num_draft_tokens}")
        self._draft_model = draft_model
        self._draft_cache = draft_cache
        self._num_draft_tokens = num_draft_tokens

    @property
    def mode(self) -> DraftMode:
        return "model"

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
        del context_tokens  # mlx_lm spec_step manages its own context
        # mlx_lm splits prompt_cache as ``[: len(model.layers)]`` for the
        # target and ``[len(model.layers) :]`` for the drafter, so we just
        # concatenate native cache lists here.
        decode_cache = list(prompt_cache) + list(self._draft_cache)
        yield from stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=list(logits_processors),
            prompt_cache=decode_cache,
            prefill_step_size=prefill_step_size,
            kv_group_size=KV_GROUP_SIZE,
            kv_bits=KV_BITS,
            draft_model=self._draft_model,
            num_draft_tokens=self._num_draft_tokens,
        )


@final
class NgramDrafter:
    """Speculative decoding using in-context n-gram lookup.

    Each spec round looks for the longest suffix (length in
    ``[min_match, max_match]``) of the running token context that
    appeared earlier in the same context, and proposes a continuation
    drawn from the tokens that followed it last time. This is the
    classic "prompt-suffix lookup drafter" used by vLLM
    (``--speculative-model='[ngram]'``) and SGLang
    (``--draft-model n-gram``).

    Match-strength-adaptive K
    -------------------------
    A short (length-``min_match``) match is weak evidence that the
    *next* ``num_draft_tokens`` tokens repeat - it's just two tokens of
    overlap, often coincidental. A long match (length ``max_match``+)
    is strong evidence: the model is genuinely re-emitting a prior
    span. We bias proposal length to match strength via
    ``K_eff = min(num_draft_tokens, match_length)``; that way short
    matches propose few drafts (cheap verify), long matches propose
    many (worth the verify cost). Disable by setting
    ``adaptive_k=False`` to always issue ``num_draft_tokens`` drafts
    when any match is found.

    Cost model: O(context * max_match) per proposal in pure Python -
    microseconds for chats up to a few thousand tokens, zero MLX work,
    zero KV cache, zero warmup. When no match is found we fall through
    to a single-token target step, so worst-case throughput equals the
    no-drafter baseline.
    """

    def __init__(
        self,
        *,
        num_draft_tokens: int,
        max_match: int = 4,
        min_match: int = 2,
        adaptive_k: bool = True,
    ) -> None:
        if num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {num_draft_tokens}")
        if min_match < 1:
            raise ValueError(f"min_match must be >= 1, got {min_match}")
        if max_match < min_match:
            raise ValueError(
                f"max_match ({max_match}) must be >= min_match ({min_match})"
            )
        self._num_draft_tokens = num_draft_tokens
        self._max_match = max_match
        self._min_match = min_match
        self._adaptive_k = adaptive_k

    @property
    def mode(self) -> DraftMode:
        return "ngram"

    @property
    def num_draft_tokens(self) -> int:
        return self._num_draft_tokens

    def propose(self, context: Sequence[int], k: int) -> list[int]:
        """Return up to ``k`` candidate continuations of ``context``.

        Returns an empty list if no suffix of length ``>= min_match``
        appears earlier in ``context``. The match is right-anchored at
        ``context[-n:]`` (we don't search inside the suffix itself, to
        avoid trivial self-overlap). When ``adaptive_k`` is enabled,
        the proposal length is capped at the match length so weak
        (short) matches don't trigger expensive K-token verifies.
        """
        if k < 1 or len(context) < self._min_match + 1:
            return []
        # Walk match length from longest to shortest, biasing toward
        # stronger matches (and earlier exit on the first match).
        upper = min(self._max_match, len(context) - 1)
        for n in range(upper, self._min_match - 1, -1):
            suffix = list(context[-n:])
            # Search backwards (most-recent match wins) through earlier
            # positions; locality of reference means the model is most
            # likely to repeat its recent self.
            for start in range(len(context) - n - 1, -1, -1):
                if list(context[start : start + n]) == suffix:
                    # Adaptive K: cap proposal length to match strength.
                    # Match length n -> at most n drafts (a 2-gram match
                    # gets 2 drafts; a 4-gram match gets up to 4).
                    cap = min(k, n) if self._adaptive_k else k
                    proposal = list(context[start + n : start + n + cap])
                    return proposal
        return []

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
        yield from _ngram_stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            context_tokens=list(context_tokens),
            prompt_cache=prompt_cache,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=list(logits_processors),
            drafter=self,
            prefill_step_size=prefill_step_size,
        )


def make_drafter(
    *,
    mode: DraftMode,
    num_draft_tokens: int,
    draft_model: Model | None,
    draft_cache: KVCacheType | None,
) -> Drafter:
    """Build a :class:`Drafter` for the resolved mode.

    Raises ``ValueError`` if ``mode in ("model", "pipelined")`` is
    requested without a loaded drafter; callers should resolve that via
    :func:`resolve_draft_mode` (which downgrades silently).

    For ``mode == "pipelined"`` the transport kind defaults to
    ``EXO_DRAFTER_TRANSPORT`` (``"inprocess"`` if unset). The remote
    transport requires asymmetric instance topology (drafter on a
    different MLX rank); see :mod:`pipelined_drafter` for details.
    """
    if mode == "none":
        return NoSpecDrafter()
    if mode == "ngram":
        return NgramDrafter(num_draft_tokens=num_draft_tokens)
    if mode == "model":
        if draft_model is None or draft_cache is None:
            raise ValueError(
                "draft_mode='model' requires both draft_model and draft_cache"
            )
        return ModelDrafter(
            draft_model=draft_model,
            draft_cache=draft_cache,
            num_draft_tokens=num_draft_tokens,
        )
    if mode == "pipelined":
        # Imported here to keep the module's import surface minimal in
        # the common (model/ngram/none) paths.
        from exo.worker.engines.mlx.generator.drafter_transport import (
            EXO_DRAFTER_TRANSPORT_ENV,
            parse_transport_kind,
            transport_factory_for,
        )
        from exo.worker.engines.mlx.generator.pipelined_drafter import (
            PipelinedModelDrafter,
        )

        transport_kind = parse_transport_kind(
            os.environ.get(EXO_DRAFTER_TRANSPORT_ENV), default="inprocess"
        )
        factory = transport_factory_for(transport_kind)
        # The factory accepts ``draft_model`` / ``draft_cache`` for the
        # in-process transport; the remote transport ignores those (they
        # live on the drafter rank).
        if transport_kind == "inprocess" and (
            draft_model is None or draft_cache is None
        ):
            raise ValueError(
                "draft_mode='pipelined' with EXO_DRAFTER_TRANSPORT='inprocess' "
                "requires both draft_model and draft_cache"
            )
        transport = factory(
            draft_model=draft_model,
            draft_cache=draft_cache,
            num_draft_tokens=num_draft_tokens,
        )
        return PipelinedModelDrafter(
            transport=transport,
            num_draft_tokens=num_draft_tokens,
        )
    # Exhaustiveness: DraftMode is a closed Literal. Any other value is a
    # programming error at the call site, so raise loudly.
    raise ValueError(f"Unknown DraftMode: {mode!r}")


def _ngram_stream_generate(
    *,
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt: mx.array,
    context_tokens: list[int],
    prompt_cache: KVCacheType,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
    drafter: NgramDrafter,
    prefill_step_size: int,
) -> Generator[GenerationResponse, None, None]:
    """Mirror of ``mlx_lm.stream_generate`` for the n-gram drafter.

    Replicates only the framing (detokenisation, tps tracking, finish
    reasons) that ``mlx_lm.stream_generate`` does for the model-drafter
    path; the actual spec loop is :func:`_ngram_speculative_step`.
    ``prompt`` is the prefill-tail (size 2 in production, but any size
    >=1 works); ``context_tokens`` is the full prompt as a Python list
    (used for n-gram lookups, not fed to the model).
    """
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()  # type: ignore[reportUnknownMemberType]
    eos_ids = _get_eos_ids(tokenizer)

    token_iter = _ngram_speculative_step(
        prompt=prompt,
        context_tokens=context_tokens,
        model=model,
        drafter=drafter,
        prompt_cache=prompt_cache,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        prefill_step_size=prefill_step_size,
    )

    # Telemetry: report the *full* prompt size (which is what the user
    # paid prefill on), not the prefill-tail we were handed.
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


def _process_logits_for_position(
    raw_logits: mx.array,
    prev_tokens: mx.array,
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
) -> mx.array:
    """Apply logits processors and convert to logprobs (single position).

    ``raw_logits`` has shape ``(vocab,)`` (already squeezed from a
    ``(1, vocab)`` per-position slice). ``prev_tokens`` is the running
    sequence of tokens emitted so far, used by repetition-penalty etc.
    """
    out = raw_logits
    for proc in logits_processors:
        out = proc(prev_tokens, out)
    return out - mx.logsumexp(out, axis=-1, keepdims=True)


def _ngram_speculative_step(
    *,
    prompt: mx.array,
    context_tokens: list[int],
    model: Model,
    drafter: NgramDrafter,
    prompt_cache: KVCacheType,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
    prefill_step_size: int,
) -> Generator[tuple[int, mx.array, bool], None, None]:
    """Custom speculative-decoding loop using an :class:`NgramDrafter`.

    Yields ``(token, logprobs, from_draft)`` tuples to match the shape
    ``mlx_lm.stream_generate`` expects from its inner token generator.

    Algorithm (greedy accept; matches the temperature-0 case our warmup
    and most code paths use):

      1. Prefill: feed ``prompt[:-1]`` to ``model`` so the cache covers
         the prompt minus its last token.
      2. Each round, ask the drafter for up to ``num_draft_tokens``
         candidates given the running context.
      3. Build a verify input ``[y, *drafts]`` (y = the last emitted
         token) and run ``model`` on it once. The cache extends by
         ``len(drafts) + 1``.
      4. Sample target's preferred token at each position. Walk the
         drafts and accept any that match the target's choice; on the
         first mismatch, also emit the target's choice at that position
         and stop. If all drafts match, emit the bonus token from the
         final position.
      5. Trim the cache by ``len(drafts) - num_accepted`` so its offset
         lines up with the emitted tokens.
      6. If the drafter declined to propose, fall back to a single-
         token target step (cost identical to non-spec generation).
    """
    y = prompt.astype(mx.uint32)

    # Mirror mlx_lm._prefill: the caller has aligned ``prompt_cache`` to
    # ``context_tokens[:-2]`` via ``exo.prefill`` + ``trim(2)``; this loop
    # advances the cache by one more token (offset N-1), leaving ``y``
    # as the seed for the spec loop.
    while y.size > 1:
        n_to_process = min(prefill_step_size, y.size - 1)
        model(y[:n_to_process][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])  # type: ignore[reportArgumentType]
        y = y[n_to_process:]
        mx.clear_cache()

    # Running context for n-gram lookup and logits processors. We start
    # from the full prompt (so the n-gram drafter can match against
    # prefix-cached portions) and append every emitted token.
    running_context: list[int] = list(context_tokens)
    prev_tokens = mx.array(running_context, dtype=mx.uint32)
    ntoks = 0

    while ntoks < max_tokens:
        # ``num_draft_tokens`` is the upper bound; cap to remaining budget
        # so the verify forward never overruns ``max_tokens``.
        num_drafts = min(max_tokens - ntoks, drafter.num_draft_tokens)
        if num_drafts < 1:
            break

        drafts = drafter.propose(running_context, num_drafts)

        if not drafts:
            # Single-token fallback: identical to non-spec generation.
            logits = model(y[None], cache=prompt_cache)
            logprobs = _process_logits_for_position(
                logits[:, -1, :].squeeze(0), prev_tokens, logits_processors
            )
            sampled = sampler(logprobs)
            mx.eval(sampled)
            sampled_token = int(sampled.item())
            yield sampled_token, logprobs, False
            running_context.append(sampled_token)
            prev_tokens = mx.concatenate(
                [prev_tokens, mx.array([sampled_token], dtype=mx.uint32)]
            )
            y = mx.array([sampled_token], dtype=mx.uint32)
            ntoks += 1
            continue

        # The proposer's contract is *up to* ``num_drafts`` tokens; the
        # rest of the loop is sized off the actual proposal length so we
        # never index past the verify forward's output.
        actual_drafts = len(drafts)

        # Verify pass: target forward on [y, *drafts]
        draft_arr = mx.array(drafts, dtype=mx.uint32)
        verify_input = mx.concatenate([y, draft_arr])
        logits = model(verify_input[None], cache=prompt_cache)
        # logits shape: (1, actual_drafts + 1, vocab)

        target_logprobs: list[mx.array] = []
        target_tokens: list[int] = []
        running_prev = prev_tokens
        for i in range(actual_drafts + 1):
            position_logits = logits[:, i, :].squeeze(0)
            position_logprobs = _process_logits_for_position(
                position_logits, running_prev, logits_processors
            )
            sampled = sampler(position_logprobs)
            mx.eval(sampled)
            sampled_token = int(sampled.item())
            target_logprobs.append(position_logprobs)
            target_tokens.append(sampled_token)
            # Speculatively assume position i was kept for the next
            # logits-processor call; this matches what
            # ``speculative_generate_step`` does internally.
            running_prev = mx.concatenate(
                [running_prev, mx.array([sampled_token], dtype=mx.uint32)]
            )

        # Greedy accept
        num_accepted = 0
        for i in range(actual_drafts):
            if target_tokens[i] == drafts[i]:
                num_accepted += 1
            else:
                break

        # Emit accepted drafts + 1 (target's choice at first mismatch
        # or bonus token after a full accept).
        emit_count = num_accepted + 1
        trim = actual_drafts - num_accepted

        for j in range(emit_count):
            tok = drafts[j] if j < num_accepted else target_tokens[j]
            from_draft = j < num_accepted
            yield tok, target_logprobs[j], from_draft
            running_context.append(tok)
            prev_tokens = mx.concatenate(
                [prev_tokens, mx.array([tok], dtype=mx.uint32)]
            )
            ntoks += 1
            if ntoks >= max_tokens:
                break

        # Cache cleanup: we appended ``actual_drafts + 1`` tokens (the seed
        # plus the proposed drafts); only the first ``num_accepted + 1``
        # of those are correct, so trim the rest.
        if trim > 0:
            # mlx_lm types the cache as ``List[Cache]``; exo's ``KVCacheType``
            # is a structural subset, so the cast + ignore mirrors the
            # pattern used in ``mlx_generate``'s drafter cache trimming.
            mlx_trim_prompt_cache(cast(list[object], prompt_cache), trim)  # type: ignore[reportArgumentType]

        y = mx.array([running_context[-1]], dtype=mx.uint32)
