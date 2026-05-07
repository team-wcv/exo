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
        # ``EXO_DRAFT_MODE=ngram`` is set process-wide. Per-request
        # overrides (``TaskParams.draft_mode``) only apply within the
        # surface that the chosen generator exposes.
        configured_draft_mode = parse_draft_mode(
            os.environ.get(EXO_DRAFT_MODE_ENV),
            default="model" if self.draft_model is not None else "none",
        )
        force_sequential_for_drafter = (
            self.draft_model is not None or configured_draft_mode == "ngram"
        )

        if os.environ.get("EXO_NO_BATCH") or force_sequential_for_drafter:
            if force_sequential_for_drafter:
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
