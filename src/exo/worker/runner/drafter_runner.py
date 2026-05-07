"""Runner for an asymmetric drafter rank.

The asymmetric placement layer (``master.placement``) appends a drafter
rank to the parent ``mx.distributed`` group whenever a model card lists
:attr:`ModelCard.drafter_eligible_nodes` and an eligible host is
reachable from every target rank. The drafter loads its own (smaller)
drafter model on a separate node and runs :func:`drafter_serve_loop`
to field forwards from target rank 0 over ``RemoteTransport``.

This module follows the same lifecycle as :class:`exo.worker.runner.runner.Runner`
(``Idle -> Connecting -> Connected -> Loading -> Loaded -> WarmingUp ->
Ready -> Running``) so the worker plan's readiness checks (which iterate
``Instance.all_runner_ids``) treat the drafter rank like any other rank.
The internals differ:

  * No target shard, no tokenizer, no chat-completion handling. The
    drafter has its own ``ModelCard`` and only loads the drafter
    weights.
  * No ``Engine`` wrapper. ``StartWarmup`` does a single forward to
    JIT-compile Metal kernels, then the drafter steps directly into
    :func:`drafter_serve_loop`, which blocks on
    :func:`mx.distributed.recv` until the target rank sends
    ``OP_SHUTDOWN``.
  * ``Shutdown`` arrives via the worker plan after target ranks have
    already sent ``OP_SHUTDOWN``; on the drafter side we just clean up
    state.

The module is import-cheap: it does not pull in any target-side
generator code (``generate.py``, ``batch_generator.py``, etc.). The
drafter runs in its own process with its own model, so memory and
import time stay tight.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast, final

import mlx.core as mx
from loguru import logger as loguru_logger
from mlx_lm.utils import load_model

from exo.download.download_utils import build_model_path, resolve_existing_model
from exo.shared.types.events import (
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
)
from exo.shared.types.worker.instances import BoundInstance, DrafterPlacement
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import ClosedResourceError, EndOfStream, MpReceiver, MpSender

if TYPE_CHECKING:
    from exo.worker.engines.mlx.types import KVCacheType, Model


@final
class DrafterRunner:
    """Lifecycle manager for the drafter rank in an asymmetric instance.

    Same task-driven state machine as the target runner -- the worker
    plan dispatches ``ConnectToGroup``, ``LoadModel``, ``StartWarmup``,
    and ``Shutdown`` in order; readiness gates iterate
    ``Instance.all_runner_ids`` so the drafter participates in
    barriers exactly like a target rank.
    """

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
    ) -> None:
        assert bound_instance.is_drafter_rank, (
            "DrafterRunner can only be constructed for an asymmetric drafter "
            "rank; check `bound_instance.is_drafter_rank` before instantiation."
        )
        placement = bound_instance.instance.drafter_placement
        assert placement is not None
        self._placement: DrafterPlacement = placement

        self.bound_instance = bound_instance
        self.runner_id = bound_instance.bound_runner_id
        self.event_sender = event_sender
        self.task_receiver = task_receiver

        self.parent_group: mx.distributed.Group | None = None
        self.target_subgroup: mx.distributed.Group | None = None
        self.draft_model: Model | None = None
        self.draft_cache: KVCacheType | None = None

        self._setup_start = time.perf_counter()
        self._update_status(RunnerIdle())
        loguru_logger.info(
            f"DrafterRunner created (runner_id={self.runner_id} "
            f"node={bound_instance.bound_node_id} "
            f"drafter_model_id={self._placement.drafter_model_id} "
            f"drafter_rank={self._placement.drafter_rank})"
        )

    def main(self) -> None:
        try:
            with self.task_receiver:
                for task in self.task_receiver:
                    if not self._dispatch(task):
                        return
        except (EndOfStream, ClosedResourceError):
            loguru_logger.warning("DrafterRunner task stream closed")

    def _dispatch(self, task: Task) -> bool:
        """Process one task; return ``False`` to exit the main loop."""
        self._send_task_status(task.task_id, TaskStatus.Running)
        match task:
            case ConnectToGroup() if isinstance(self.current_status, RunnerIdle):
                self._handle_connect(task)
            case LoadModel() if isinstance(self.current_status, RunnerConnected):
                self._handle_load(task)
            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                self._handle_start_warmup(task)
            case Shutdown():
                self._handle_shutdown(task)
                return False
            case _:
                raise ValueError(
                    f"DrafterRunner received {task.__class__.__name__} outside "
                    f"of state machine in {self.current_status=}"
                )
        return True

    def _handle_connect(self, task: Task) -> None:
        from exo.worker.engines.mlx.utils_mlx import initialize_mlx

        self._update_status(RunnerConnecting())
        self._acknowledge(task)
        split = initialize_mlx(self.bound_instance)
        assert split.is_asymmetric, (
            "DrafterRunner reached ConnectToGroup but MlxGroupSplit reports "
            "symmetric placement; placement layer should have set "
            "drafter_placement on this instance"
        )
        self.parent_group = split.parent
        self.target_subgroup = split.target_subgroup
        loguru_logger.info(
            f"DrafterRunner connected: parent_size={split.parent.size()} "
            f"my_rank={split.parent.rank()} "
            f"drafter_rank_in_parent={split.drafter_rank_in_parent}"
        )
        self._send_task_status(task.task_id, TaskStatus.Complete)
        self._update_status(RunnerConnected())

    def _handle_load(self, task: Task) -> None:
        from exo.worker.engines.mlx.cache import make_kv_cache

        drafter_id = self._placement.drafter_model_id
        drafter_path = resolve_existing_model(drafter_id)
        if drafter_path is None:
            # Build a fallback path so the error message points at where
            # the operator should drop the weights.
            expected_path = build_model_path(drafter_id)
            raise FileNotFoundError(
                f"Drafter weights for {drafter_id} not found on this node "
                f"(expected at {expected_path}). Asymmetric drafter "
                "placement requires pre-downloading the drafter model "
                "on every drafter-eligible node; auto-download is not "
                "yet implemented for the drafter rank."
            )

        self._update_status(RunnerLoading(layers_loaded=0, total_layers=0))
        self._acknowledge(task)

        load_start = time.perf_counter()
        loguru_logger.info(f"DrafterRunner loading {drafter_id} from {drafter_path}")
        model, _ = load_model(drafter_path, lazy=True, strict=False)
        mx.eval(model)
        self.draft_model = cast("Model", model)
        self.draft_cache = make_kv_cache(model=self.draft_model)
        loguru_logger.info(
            f"DrafterRunner loaded {drafter_id} in "
            f"{(time.perf_counter() - load_start):.2f}s"
        )

        self._send_task_status(task.task_id, TaskStatus.Complete)
        self._update_status(RunnerLoaded())

    def _handle_start_warmup(self, task: Task) -> None:
        assert self.parent_group is not None
        assert self.draft_model is not None
        assert self.draft_cache is not None

        self._update_status(RunnerWarmingUp())
        self._acknowledge(task)

        # JIT-compile drafter Metal kernels with a single forward, so
        # the first real spec-decode round on the target rank doesn't
        # eat the compile latency.
        warmup_start = time.perf_counter()
        seed = mx.array([[0]], dtype=mx.uint32)
        _ = self.draft_model(seed, cache=self.draft_cache)
        mx.eval([c.state for c in self.draft_cache])  # type: ignore[reportArgumentType]
        # Reset the cache so we don't carry the warmup token into real
        # generation.
        from typing import cast as _cast

        from mlx_lm.models.cache import trim_prompt_cache as mlx_trim_prompt_cache

        mlx_trim_prompt_cache(_cast(list[object], self.draft_cache), 1)  # type: ignore[reportArgumentType]
        loguru_logger.info(
            f"DrafterRunner warmup complete in "
            f"{(time.perf_counter() - warmup_start):.2f}s; "
            f"setup_total={(time.perf_counter() - self._setup_start):.2f}s"
        )

        self._send_task_status(task.task_id, TaskStatus.Complete)
        # The drafter has no prefill server, so prefill_server_port is None.
        self._update_status(RunnerReady(prefill_server_port=None))
        self._update_status(RunnerRunning())

        # Enter the drafter serve loop. This blocks until the target
        # rank sends OP_SHUTDOWN. The serve loop's send/recv use the
        # parent group; target rank 0 is conventionally the only target
        # rank that drives drafter IPC.
        self._serve_loop()

        # OP_SHUTDOWN arrived; transition back to Ready so the worker
        # plan's Shutdown task can drive us to RunnerShutdown.
        self._update_status(RunnerReady(prefill_server_port=None))

    def _serve_loop(self) -> None:
        from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

        assert self.parent_group is not None
        assert self.draft_model is not None
        assert self.draft_cache is not None

        # Target rank that drives drafter IPC. By placement convention
        # rank 0 of the parent group owns sampling decisions and so is
        # the rank that calls RemoteTransport.forward / trim_cache /
        # shutdown.
        target_rank = 0
        # ``num_draft_tokens`` here only sizes the response buffer; the
        # spec loop on the target side may issue forwards with
        # ``num_forwards`` up to K+1, so we mirror exactly its config.
        num_draft_tokens = self._num_draft_tokens()
        loguru_logger.info(
            f"DrafterRunner entering serve_loop "
            f"(K={num_draft_tokens}, target_rank={target_rank})"
        )
        drafter_serve_loop(
            draft_model=self.draft_model,
            draft_cache=self.draft_cache,
            num_draft_tokens=num_draft_tokens,
            group=self.parent_group,
            target_rank=target_rank,
        )
        loguru_logger.info("DrafterRunner serve_loop exited via OP_SHUTDOWN")

    @staticmethod
    def _num_draft_tokens() -> int:
        # Same default the target-side builder uses; reading the env
        # var keeps drafter and target in lock-step without an explicit
        # IPC message at warmup time.
        from exo.worker.runner.llm_inference.batch_generator import (
            DEFAULT_NUM_DRAFT_TOKENS,
            EXO_NUM_DRAFT_TOKENS,
            parse_env_int,
        )

        return parse_env_int(EXO_NUM_DRAFT_TOKENS, default=DEFAULT_NUM_DRAFT_TOKENS)

    def _handle_shutdown(self, task: Task) -> None:
        loguru_logger.info("DrafterRunner shutting down")
        self._update_status(RunnerShuttingDown())
        self._acknowledge(task)
        # Release the model and cache so the drafter rank's process
        # frees its drafter weights before exiting.
        self.draft_model = None
        self.draft_cache = None
        self.parent_group = None
        self.target_subgroup = None
        import gc

        gc.collect()
        self._send_task_status(task.task_id, TaskStatus.Complete)
        self._update_status(RunnerShutdown())

    # -- helpers ---------------------------------------------------------

    def _update_status(self, status: RunnerStatus) -> None:
        self.current_status: RunnerStatus = status
        self.event_sender.send(
            RunnerStatusUpdated(runner_id=self.runner_id, runner_status=status)
        )

    def _send_task_status(self, task_id: TaskId, status: TaskStatus) -> None:
        self.event_sender.send(TaskStatusUpdated(task_id=task_id, task_status=status))

    def _acknowledge(self, task: Task) -> None:
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))


__all__ = ["DrafterRunner"]
