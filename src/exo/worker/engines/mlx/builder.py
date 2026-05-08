import contextlib
import os
import socket
from collections.abc import Generator
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS
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
    # ``group`` is the target ranks' ``mx.distributed.Group``: pipeline
    # / tensor / batch collectives all run on it. Under the v3+ wire
    # the drafter is NOT a member of this group (asymmetric drafters
    # talk to target rank 0 over a TCP socket; see ``drafter_socket``
    # below).
    group: mx.distributed.Group | None = None
    # Connected TCP socket from target rank 0 to the drafter rank.
    # Set ONLY on target rank 0 of an asymmetric placement; ``None``
    # everywhere else (other target ranks don't drive drafter IPC, and
    # single-device / symmetric multi-rank builds have no drafter
    # wire at all).
    drafter_socket: socket.socket | None = None
    drafter_rank_in_parent: int | None = None
    vision_processor: VisionProcessor | None = None
    draft_model: Model | None = None
    draft_model_id: ModelId | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        split = initialize_mlx(bound_instance)
        self.group = split.target_subgroup
        # Only target rank 0 in an asymmetric placement holds a drafter
        # socket; every other rank sees ``None`` here. ``MlxGroupSplit``
        # types it as ``object | None`` to keep the dataclass importable
        # without ``socket``; cast back to the concrete type for
        # consumers.
        if split.drafter_socket is not None:
            self.drafter_socket = cast(socket.socket, split.drafter_socket)
        else:
            self.drafter_socket = None
        self.drafter_rank_in_parent = split.drafter_rank_in_parent

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
        if self.drafter_socket is not None:
            with contextlib.suppress(OSError):
                self.drafter_socket.close()
            self.drafter_socket = None
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
            self.draft_model is not None
            or configured_draft_mode in ("ngram", "pipelined")
        )

        # Asymmetric placement: drafter lives on a separate node; only
        # target rank 0 owns the drafter wire (``drafter_socket``).
        # Force the SequentialGenerator path (BatchGenerator has no
        # spec-decoding hook) and build a long-lived RemoteTransport
        # that the spec loop reuses across requests.
        #
        # Other target ranks in an asymmetric placement (rank >= 1) see
        # ``drafter_socket is None`` and treat their build the same as
        # symmetric multi-rank: they participate in target collectives
        # but never call drafter ops directly. The spec loop's
        # rank-0-only sampling decision keeps that invariant.
        is_asymmetric_target_rank_zero = self.drafter_socket is not None
        # Long-lived ``RemoteTransport`` (NOT a per-task DrafterTransport).
        # Each in-flight request opens its own session via
        # :meth:`RemoteTransport.open_session`; the session handle is the
        # actual DrafterTransport consumed by the spec loop. See
        # ``remote_drafter.py`` module docstring for the wire-protocol
        # session multiplexing rationale.
        from exo.worker.engines.mlx.generator.remote_drafter import RemoteTransport

        remote_drafter_transport: RemoteTransport | None = None
        if is_asymmetric_target_rank_zero:
            assert self.drafter_socket is not None
            from exo.worker.engines.mlx.generator.remote_drafter import (
                make_remote_transport,
            )

            num_draft_tokens_remote = parse_env_int(
                EXO_NUM_DRAFT_TOKENS, DEFAULT_NUM_DRAFT_TOKENS
            )
            target_world_size = self.group.size() if self.group is not None else 1
            logger.info(
                "Allocating long-lived RemoteTransport: "
                f"target_world_size={target_world_size} "
                f"drafter_rank={self.drafter_rank_in_parent} "
                f"K={num_draft_tokens_remote} "
                f"transport=tcp_socket"
            )
            remote_drafter_transport = make_remote_transport(
                draft_model=None,
                draft_cache=None,
                num_draft_tokens=num_draft_tokens_remote,
                sock=self.drafter_socket,
            )

        # Asymmetric "is the cluster speculative-decoding-aware" check.
        # Used below to force ``SequentialGenerator`` and to log mode
        # selection. Non-zero ranks of an asymmetric instance do NOT
        # set this flag (they don't own the wire) but they still enter
        # the same generator path because the placement-time decision
        # to enable the drafter is uniform across target ranks.
        is_asymmetric = (
            is_asymmetric_target_rank_zero or self.drafter_rank_in_parent is not None
        )

        if (
            os.environ.get("EXO_NO_BATCH")
            or force_sequential_for_drafter
            or is_asymmetric
        ):
            if is_asymmetric:
                logger.info(
                    "using SequentialGenerator (asymmetric placement: "
                    "drafter lives on a separate MLX rank, pipelined+remote spec)"
                )
            elif force_sequential_for_drafter:
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
            if force_sequential_for_drafter or is_asymmetric:
                logger.info(
                    f"speculative decoding: mode={'pipelined+remote' if is_asymmetric else configured_draft_mode}, "
                    f"K={num_draft_tokens} (adaptive={adaptive_draft_tokens}), "
                    f"skip_drafter_when_max_tokens<={drafter_min_output_tokens}"
                )

            # Concurrent in-flight tasks. Asymmetric pipelined+remote
            # rides the same ``EXO_MAX_CONCURRENT_REQUESTS`` cap as every
            # other config now that the wire protocol carries a
            # ``session_id`` slot: each in-flight target request opens
            # its own ``_SessionHandle`` via
            # ``RemoteTransport.open_session()`` and the drafter rank
            # multiplexes per-session KV caches. The wire stays serial
            # (single ``ThreadPoolExecutor`` on the target, single recv
            # loop on the drafter) so ``mx.distributed.send/recv``
            # ordering is preserved; concurrency comes from interleaving
            # forward / verify rounds across sessions, which is the
            # whole point of asymmetric placement -- keep the drafter
            # rank busy serving session A while the target verifies
            # session B's drafts.
            max_concurrent_tasks = EXO_MAX_CONCURRENT_REQUESTS
            if max_concurrent_tasks > 1:
                logger.info(
                    f"SequentialGenerator round-robin concurrency: "
                    f"max_concurrent_tasks={max_concurrent_tasks} "
                    f"(EXO_MAX_CONCURRENT_REQUESTS)"
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
                drafter_rank_in_parent=self.drafter_rank_in_parent,
                remote_drafter_transport=remote_drafter_transport,
                max_concurrent_tasks=max_concurrent_tasks,
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
