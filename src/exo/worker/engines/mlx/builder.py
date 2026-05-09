import contextlib
import os
from collections.abc import Generator
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    DEFAULT_DRAFTER_MIN_OUTPUT_TOKENS,
    DEFAULT_NUM_DRAFT_TOKENS,
    EXO_ADAPTIVE_DRAFT_TOKENS,
    EXO_DRAFTER_MIN_OUTPUT_TOKENS,
    EXO_NUM_DRAFT_TOKENS,
    BatchGenerator,
    SequentialGenerator,
    parse_env_int,
)
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser

from .cache import KVPrefixCache
from .generator.drafter import EXO_DRAFT_MODE_ENV, parse_draft_mode
from .types import Model
from .utils_mlx import (
    initialize_mlx,
    load_mlx_items,
)
from .vision import VisionProcessor


@dataclass
class MlxBuilder(Builder):
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    group: mx.distributed.Group | None = None
    vision_processor: VisionProcessor | None = None
    draft_model: Model | None = None
    draft_model_id: ModelId | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        self.group = initialize_mlx(bound_instance)

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        (
            self.inference_model,
            self.tokenizer,
            self.vision_processor,
            self.draft_model,
            self.draft_model_id,
        ) = yield from load_mlx_items(bound_instance, self.group)

    def close(self) -> None:
        with contextlib.suppress(NameError, AttributeError):
            del self.inference_model
        with contextlib.suppress(NameError, AttributeError):
            del self.tokenizer
        with contextlib.suppress(NameError, AttributeError):
            del self.group
        with contextlib.suppress(NameError, AttributeError):
            del self.draft_model

    def build(
        self,
    ) -> Engine:
        assert self.inference_model
        assert self.tokenizer

        vision_processor = self.vision_processor

        tool_parser = None
        logger.info(
            f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
        )
        if (
            self.tokenizer.tool_call_start
            and self.tokenizer.tool_call_end
            and self.tokenizer.tool_parser  # type: ignore
        ):
            tool_parser = make_mlx_parser(
                self.tokenizer.tool_call_start,
                self.tokenizer.tool_call_end,
                self.tokenizer.tool_parser,  # type: ignore
            )

        kv_prefix_cache = KVPrefixCache(self.group)
        # Item 6: dedicated KVPrefixCache for the drafter so multi-turn
        # workloads don't repeatedly prefill the drafter on the same prefix.
        # Allocated only when a drafter is actually loaded; None means
        # mlx_generate falls back to the per-request drafter prefill.
        drafter_kv_prefix_cache: KVPrefixCache | None = (
            KVPrefixCache(self.group) if self.draft_model is not None else None
        )

        device_rank = 0 if self.group is None else self.group.rank()

        # Speculative decoding (model or n-gram) currently flows only through
        # SequentialGenerator -> mlx_generate. Upstream BatchGenerator does
        # not accept a draft model and has no hook for n-gram drafting, so
        # force the sequential path whenever speculative decoding could
        # plausibly run for any request: a drafter model is loaded *or*
        # ``EXO_DRAFT_MODE=ngram`` is set process-wide *or* the operator
        # opted into request-level draft overrides via
        # ``EXO_ALLOW_REQUEST_DRAFTING``. Per-request overrides
        # (``TaskParams.draft_mode``) only apply within the surface that
        # the chosen generator exposes.
        #
        # Codex P2 (PR #19 round 2): without ``EXO_ALLOW_REQUEST_DRAFTING``
        # a node started in normal batch mode silently dropped
        # ``draft_mode="ngram"`` request overrides because BatchGenerator
        # has no spec-decoding hook. This broke the newly added
        # API-level override path for A/B tests and mixed traffic. The
        # opt-in trades batching for per-request spec-decoding control;
        # operators who don't need request-level spec stay on
        # BatchGenerator with the default settings.
        #
        # Codex P1 (PR #19 round-(N+3), builder.py:136): on multi-device
        # runners, ``mlx_generate`` unconditionally demotes
        # ``draft_mode`` to ``"none"`` (see ``generate.py``: ``if group
        # is not None: draft_mode = "none"``), so swapping to
        # ``SequentialGenerator`` for drafting buys nothing and only
        # loses batching. PR #20 reintroduces speculative decoding for
        # asymmetric placements, but PR #19 stand-alone has no
        # multi-device drafter path. Gate the sequential fallback on
        # single-device runners; multi-device nodes keep
        # ``BatchGenerator`` regardless of ``EXO_DRAFT_MODE`` /
        # ``EXO_ALLOW_REQUEST_DRAFTING`` so concurrent traffic doesn't
        # silently lose throughput.
        #
        # Codex P1 (PR #19 round-(N+6), builder.py:151): drop
        # ``configured_draft_mode == "ngram"`` from the
        # force-sequential trigger. ``mlx_generate`` now demotes
        # ``draft_mode="ngram"`` to ``"none"`` for any non-greedy
        # request (see :func:`_request_is_greedy_sampling`), and the
        # default sampler path uses ``temperature=0.7`` when the
        # request omits temperature. So a worker booted with
        # ``EXO_DRAFT_MODE=ngram`` against mixed traffic would
        # disable batching for the entire worker yet only run
        # speculation for the (rare) greedy subset -- a strict
        # throughput regression for the common case. n-gram remains
        # opt-in via ``EXO_NO_BATCH=1`` (operators who explicitly
        # want greedy-only n-gram acceleration) or
        # ``EXO_ALLOW_REQUEST_DRAFTING=1`` (per-request override
        # path); without either, ngram requests fall back to plain
        # decode under BatchGenerator and the worker keeps full
        # batching throughput. Emit a warning when this condition
        # holds so operators know n-gram won't actually run.
        configured_draft_mode = parse_draft_mode(
            os.environ.get(EXO_DRAFT_MODE_ENV),
            default="model" if self.draft_model is not None else "none",
        )
        allow_request_drafting = os.environ.get(
            "EXO_ALLOW_REQUEST_DRAFTING", ""
        ).lower() in {"1", "true", "yes"}
        is_single_device = self.group is None or self.group.size() == 1
        drafting_can_run_here = is_single_device
        # Codex P1 (PR #19 round-(N+8), builder.py:169): the gate
        # MUST honour ``EXO_DRAFT_MODE=none``. Pre-fix the gate
        # forced ``SequentialGenerator`` whenever a drafter model was
        # loaded, even if the operator explicitly opted out of
        # speculation via ``EXO_DRAFT_MODE=none``. In that
        # configuration ``mlx_generate`` resolves
        # ``draft_mode="none"`` for every request (the env var
        # overrides the loaded-drafter default), so we'd lose
        # batching with zero spec-decode benefit -- a worker-wide
        # throughput regression. ``allow_request_drafting`` still
        # forces sequential because per-request overrides may
        # legitimately raise ``draft_mode`` above ``"none"``.
        drafter_loaded_will_run = (
            self.draft_model is not None and configured_draft_mode != "none"
        )
        force_sequential_for_drafter = drafting_can_run_here and (
            drafter_loaded_will_run or allow_request_drafting
        )
        ngram_configured_without_force_sequential = (
            drafting_can_run_here
            and configured_draft_mode == "ngram"
            and not force_sequential_for_drafter
        )
        # Codex P1 (PR #19 round-(N+8), builder.py:169): when the
        # operator loaded a drafter model AND opted out of
        # speculation via ``EXO_DRAFT_MODE=none``, log it explicitly
        # so the choice is visible in startup logs (otherwise the
        # operator might think the loaded drafter is participating
        # while every request silently runs without speculation).
        drafter_loaded_but_explicitly_disabled = (
            drafting_can_run_here
            and self.draft_model is not None
            and configured_draft_mode == "none"
            and not allow_request_drafting
        )

        if os.environ.get("EXO_NO_BATCH") or force_sequential_for_drafter:
            if force_sequential_for_drafter:
                if allow_request_drafting and self.draft_model is None:
                    logger.info(
                        "using SequentialGenerator (EXO_ALLOW_REQUEST_DRAFTING set; "
                        "BatchGenerator has no spec-decoding hook for request "
                        "overrides)"
                    )
                else:
                    logger.info(
                        f"using SequentialGenerator (draft_mode={configured_draft_mode!r}; "
                        f"BatchGenerator has no spec-decoding hook)"
                    )
            else:
                logger.info("using SequentialGenerator (batching disabled)")

            num_draft_tokens = parse_env_int(
                EXO_NUM_DRAFT_TOKENS, DEFAULT_NUM_DRAFT_TOKENS
            )
            drafter_min_output_tokens = parse_env_int(
                EXO_DRAFTER_MIN_OUTPUT_TOKENS,
                DEFAULT_DRAFTER_MIN_OUTPUT_TOKENS,
                minimum=0,
            )
            adaptive_draft_tokens = os.environ.get(
                EXO_ADAPTIVE_DRAFT_TOKENS, ""
            ).lower() in {"1", "true", "yes"}
            if force_sequential_for_drafter:
                logger.info(
                    f"speculative decoding: mode={configured_draft_mode}, "
                    f"K={num_draft_tokens} (adaptive={adaptive_draft_tokens}), "
                    f"skip_drafter_when_max_tokens<={drafter_min_output_tokens}"
                )

            return SequentialGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
                draft_model=self.draft_model,
                draft_model_id=self.draft_model_id,
                drafter_kv_prefix_cache=drafter_kv_prefix_cache,
                num_draft_tokens=num_draft_tokens,
                drafter_min_output_tokens=drafter_min_output_tokens,
                adaptive_draft_tokens=adaptive_draft_tokens,
            )
        else:
            # Codex P1 (PR #19 round-(N+3), builder.py:136): make the
            # multi-device drafting-disabled path explicit so operators
            # don't silently observe missing speculative decoding.
            drafting_was_requested = (
                self.draft_model is not None
                or configured_draft_mode == "ngram"
                or allow_request_drafting
            )
            if not drafting_can_run_here and drafting_was_requested:
                logger.info(
                    f"using BatchGenerator (drafting unavailable on multi-device "
                    f"runner: group.size={self.group.size() if self.group is not None else 1}; "
                    f"mlx_generate would demote draft_mode='none' anyway, keeping "
                    f"batching for throughput)"
                )
            elif drafter_loaded_but_explicitly_disabled:
                # Codex P1 (PR #19 round-(N+8), builder.py:169): a
                # drafter model is loaded but the operator set
                # ``EXO_DRAFT_MODE=none``, so every request resolves
                # to ``draft_mode="none"`` in ``mlx_generate``.
                # SequentialGenerator would lose batching for
                # nothing in this configuration. Keep
                # BatchGenerator and surface the choice loudly so
                # operators see why their loaded drafter weights
                # appear inactive.
                logger.info(
                    f"using BatchGenerator (drafter weights loaded "
                    f"({self.draft_model_id}) but EXO_DRAFT_MODE='none' "
                    f"explicitly disables speculation; keeping batching "
                    f"for throughput. Set EXO_DRAFT_MODE='model' or "
                    f"clear the env var to re-enable spec decode)"
                )
            elif ngram_configured_without_force_sequential:
                # Codex P1 (PR #19 round-(N+6), builder.py:151): make
                # the n-gram-on-BatchGenerator no-op path explicit so
                # operators see that ``EXO_DRAFT_MODE=ngram`` alone
                # has no runtime effect. To actually run n-gram set
                # ``EXO_NO_BATCH=1`` (greedy-only deployments) or
                # ``EXO_ALLOW_REQUEST_DRAFTING=1`` (per-request
                # override path).
                logger.warning(
                    "using BatchGenerator with EXO_DRAFT_MODE='ngram' set: "
                    "BatchGenerator has no spec-decoding hook so n-gram "
                    "drafting will be a no-op for every request. To run "
                    "n-gram set EXO_NO_BATCH=1 (forces SequentialGenerator) "
                    "or EXO_ALLOW_REQUEST_DRAFTING=1 (per-request override "
                    "path); batching is preserved here because the prior "
                    "behaviour disabled batching worker-wide for non-greedy "
                    "traffic that mlx_generate now demotes to 'none' anyway."
                )
            else:
                logger.info("using BatchGenerator")
            return BatchGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
            )
