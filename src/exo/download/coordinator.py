from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import anyio
from anyio import BrokenResourceError, ClosedResourceError, current_time, to_thread
from loguru import logger

from exo.download.download_utils import (
    RepoDownloadProgress,
    delete_model,
    is_read_only_model_dir,
    map_repo_download_progress_to_download_progress_data,
    resolve_existing_model,
)
from exo.download.shard_downloader import ShardDownloader
from exo.shared.constants import EXO_DEFAULT_MODELS_DIR, EXO_MODELS_READ_ONLY_DIRS
from exo.shared.models.model_cards import ModelCard, ModelId, get_model_cards
from exo.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    NodeDownloadProgress,
)
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender
from exo.utils.task_group import TaskGroup

# Mirrors the same env var consumed by the worker's MLX loader. Keeping the
# string literal in lockstep so users only need to set one variable to opt
# out of speculative decoding entirely (skips both download and load).
_DRAFTER_DISABLED_VALUES = frozenset({"1", "true", "yes"})


def _drafter_disabled_by_env() -> bool:
    return os.environ.get("EXO_DISABLE_DRAFTER", "").lower() in _DRAFTER_DISABLED_VALUES


@dataclass
class DownloadCoordinator:
    node_id: NodeId
    shard_downloader: ShardDownloader
    download_command_receiver: Receiver[ForwarderDownloadCommand]
    event_sender: Sender[Event]
    offline: bool = False

    # Local state
    download_status: dict[ModelId, DownloadProgress] = field(default_factory=dict)
    active_downloads: dict[ModelId, anyio.CancelScope] = field(default_factory=dict)

    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _stopped: anyio.Event = field(init=False, default_factory=anyio.Event)

    # Per-model throttle for download progress events
    _last_progress_time: dict[ModelId, float] = field(default_factory=dict)

    # Map of target model_id -> drafter model_ids spawned alongside it.
    # When the user cancels or deletes the target, we propagate the
    # cancellation/deletion to its chained drafters so they don't keep
    # consuming network/disk after the user revoked the original
    # download intent. Populated only when the drafter chain actually
    # runs (offline/disabled-by-env paths short-circuit and add no
    # children).
    _drafter_children: dict[ModelId, list[ModelId]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.shard_downloader.on_progress(self._download_progress_callback)

    @staticmethod
    def _default_model_dir(model_id: ModelId) -> str:
        return str(EXO_DEFAULT_MODELS_DIR / model_id.normalize())

    def _completed_from_path(
        self,
        shard: ShardMetadata,
        found: Path,
        total: Memory,
    ) -> DownloadCompleted:
        return DownloadCompleted(
            shard_metadata=shard,
            node_id=self.node_id,
            total=total,
            model_directory=str(found),
            read_only=is_read_only_model_dir(found),
        )

    async def _download_progress_callback(
        self, callback_shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        model_id = callback_shard.model_card.model_id
        throttle_interval_secs = 1.0

        try:
            if progress.status == "complete":
                found = await to_thread.run_sync(
                    resolve_existing_model, model_id, callback_shard.model_card
                )
                if found is not None:
                    completed = self._completed_from_path(
                        callback_shard, found, progress.total
                    )
                else:
                    completed = DownloadCompleted(
                        shard_metadata=callback_shard,
                        node_id=self.node_id,
                        total=progress.total,
                        model_directory=self._default_model_dir(model_id),
                    )
                self.download_status[model_id] = completed
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=completed)
                )
                self._last_progress_time.pop(model_id, None)
            elif (
                progress.status == "in_progress"
                and current_time() - self._last_progress_time.get(model_id, 0.0)
                > throttle_interval_secs
            ):
                ongoing = DownloadOngoing(
                    node_id=self.node_id,
                    shard_metadata=callback_shard,
                    download_progress=map_repo_download_progress_to_download_progress_data(
                        progress
                    ),
                    model_directory=self._default_model_dir(model_id),
                )
                self.download_status[model_id] = ongoing
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=ongoing)
                )
                self._last_progress_time[model_id] = current_time()
        except (BrokenResourceError, ClosedResourceError):
            logger.debug(
                f"Event stream closed while sending download progress for {model_id}, skipping update"
            )

    async def run(self) -> None:
        logger.info(
            f"Starting DownloadCoordinator{' (offline mode)' if self.offline else ''}"
        )
        try:
            async with self._tg as tg:
                tg.start_soon(self._command_processor)
                tg.start_soon(self._emit_existing_download_progress)
        finally:
            self._stopped.set()

    async def shutdown(self) -> None:
        self._tg.cancel_tasks()
        await self._stopped.wait()

    async def _command_processor(self) -> None:
        with self.download_command_receiver as commands:
            async for cmd in commands:
                # Only process commands targeting this node
                if cmd.command.target_node_id != self.node_id:
                    continue

                match cmd.command:
                    case StartDownload(shard_metadata=shard):
                        await self._start_download(shard)
                    case DeleteDownload(model_id=model_id):
                        await self._delete_download(model_id)
                    case CancelDownload(model_id=model_id):
                        await self._cancel_download(model_id)

    async def _cancel_download(self, model_id: ModelId) -> None:
        if model_id in self.active_downloads and model_id in self.download_status:
            logger.info(f"Cancelling download for {model_id}")
            self.active_downloads[model_id].cancel()
            current_status = self.download_status[model_id]
            downloaded = Memory()
            total = Memory()
            if isinstance(current_status, DownloadOngoing):
                downloaded = current_status.download_progress.downloaded
                total = current_status.download_progress.total
            pending = DownloadPending(
                shard_metadata=current_status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._default_model_dir(model_id),
                downloaded=downloaded,
                total=total,
            )
            self.download_status[model_id] = pending
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )
        # Codex flagged (P2, PR #18 round 2) that cancelling a target
        # left chained drafters running in the background, consuming
        # network/disk after the user revoked the original download
        # intent. Pop the parent->children mapping (so we don't
        # double-cancel on a follow-up cancel of the same target) and
        # cascade the cancel.
        children = self._drafter_children.pop(model_id, [])
        for child_model_id in children:
            if child_model_id in self.active_downloads:
                logger.info(
                    f"Cancelling chained drafter {child_model_id} alongside "
                    f"target {model_id}"
                )
                await self._cancel_download(child_model_id)

    async def _start_download(self, shard: ShardMetadata) -> None:
        model_id = shard.model_card.model_id

        # Check if already downloading, complete, or recently failed
        if model_id in self.download_status:
            status = self.download_status[model_id]
            if isinstance(status, (DownloadOngoing, DownloadCompleted, DownloadFailed)):
                logger.debug(
                    f"Download for {model_id} already in progress, complete, or failed, skipping"
                )
                # Even if the target was already in flight, the user may have
                # just upgraded exo onto a build that knows about its drafter.
                # Make sure the drafter download is chained either way.
                self._spawn_drafter_chain(shard)
                return

        # Check all model directories for pre-existing complete models
        found_path = await to_thread.run_sync(
            resolve_existing_model, model_id, shard.model_card
        )
        if found_path is not None:
            logger.info(f"DownloadCoordinator: Model {model_id} found at {found_path}")
            completed = self._completed_from_path(
                shard, found_path, shard.model_card.storage_size
            )
            self.download_status[model_id] = completed
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=completed)
            )
            self._spawn_drafter_chain(shard)
            return

        # Emit pending status
        progress = DownloadPending(
            shard_metadata=shard,
            node_id=self.node_id,
            model_directory=self._default_model_dir(model_id),
        )
        self.download_status[model_id] = progress
        await self.event_sender.send(NodeDownloadProgress(download_progress=progress))

        # Check initial status from downloader
        initial_progress = (
            await self.shard_downloader.get_shard_download_status_for_shard(shard)
        )

        if initial_progress.status == "complete":
            found = await to_thread.run_sync(
                resolve_existing_model, model_id, shard.model_card
            )
            if found is not None:
                completed = self._completed_from_path(
                    shard, found, initial_progress.total
                )
            else:
                completed = DownloadCompleted(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    total=initial_progress.total,
                    model_directory=self._default_model_dir(model_id),
                )
            self.download_status[model_id] = completed
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=completed)
            )
            self._spawn_drafter_chain(shard)
            return

        if self.offline:
            logger.warning(
                f"Offline mode: model {model_id} is not fully available locally, cannot download"
            )
            failed = DownloadFailed(
                shard_metadata=shard,
                node_id=self.node_id,
                error_message=f"Model files not found locally in offline mode: {model_id}",
                model_directory=self._default_model_dir(model_id),
            )
            self.download_status[model_id] = failed
            await self.event_sender.send(NodeDownloadProgress(download_progress=failed))
            return

        # Start actual download
        self._start_download_task(shard, initial_progress)
        self._spawn_drafter_chain(shard)

    def _spawn_drafter_chain(self, target_shard: ShardMetadata) -> None:
        """Background the drafter chain so command processing doesn't block.

        Codex flagged (P1, PR #18 round 2) that
        ``_maybe_chain_drafter_download`` ran inline during
        ``StartDownload`` handling. ``ModelCard.load`` falls through
        to ``ModelCard.fetch_from_hf`` whenever the drafter card
        isn't already in ``_card_cache``, and a slow/unreachable HF
        fetch would block the command loop and delay unrelated
        ``CancelDownload``/``DeleteDownload`` commands until the
        client timeout. That turns a best-effort drafter step into
        control-plane backpressure whenever drafter metadata is cold.

        Fix: dispatch the chain on the coordinator's own task group
        via ``start_soon`` so the command processor returns
        immediately and remains responsive. Errors inside the chain
        are still logged-and-swallowed (best-effort semantics
        preserved); the only difference is that they no longer hold
        up unrelated commands.
        """
        self._tg.start_soon(self._maybe_chain_drafter_download, target_shard)

    async def _maybe_chain_drafter_download(self, target_shard: ShardMetadata) -> None:
        """Enqueue downloads for every drafter declared on ``target_shard``'s
        model card.

        We download *all* candidate drafters so the runner can switch between
        them at startup time via ``EXO_DRAFTER_PREFERENCE`` without an
        on-demand fetch. Drafters are small (typically <2GB) so the storage
        overhead is fine.

        Drafter downloads are silent best-effort: anything that fails (no
        cards, env opt-out, HF unreachable, drafter already tracked) is
        logged and swallowed. The target download is the source of truth for
        the user's intent; speculative decoding is best-effort.

        Each drafter is downloaded as a single ``PipelineShardMetadata`` for
        the entire model. Speculative decoding is single-device today (see
        ``mlx_generate``), so we never need a sharded drafter.
        """
        drafter_ids = list(target_shard.model_card.drafter_model_ids)
        if not drafter_ids:
            return
        if _drafter_disabled_by_env():
            logger.debug(
                f"EXO_DISABLE_DRAFTER set; skipping drafter downloads "
                f"{drafter_ids} for {target_shard.model_card.model_id}"
            )
            return
        if self.offline:
            # Offline mode: ``ModelCard.load`` falls through to
            # ``fetch_from_hf`` whenever the drafter card isn't already
            # in ``_card_cache``, which is an outbound HuggingFace
            # request. Drafter downloads are silent best-effort, so the
            # subsequent ``DownloadFailed`` would have been swallowed
            # anyway, but the HF call itself can stall command
            # processing for the full client timeout (the upstream
            # offline guard at ``_start_download`` line 263 only fires
            # *after* this path has already issued the network call).
            # Skip drafter chaining outright in offline mode -- if the
            # operator wants a drafter, they need to ship it locally.
            logger.debug(
                f"Offline mode: skipping drafter card resolution "
                f"{drafter_ids} for {target_shard.model_card.model_id}"
            )
            return

        target_model_id = target_shard.model_card.model_id
        chained: list[ModelId] = []
        for drafter_id in drafter_ids:
            if drafter_id in self.download_status:
                # Already tracked: still record the parent->child link
                # so a subsequent target cancel propagates to the
                # already-running drafter download. Avoids the case
                # where a drafter started by an earlier target stays
                # alive after the user cancels the only target that
                # references it. (We don't check for OTHER targets
                # also referencing this drafter -- if needed, the
                # drafter is small enough that re-downloading it
                # later is cheap, and tracking a many-to-many graph
                # would balloon the coordinator state.)
                chained.append(drafter_id)
                continue

            try:
                drafter_card = await ModelCard.load(drafter_id)
            except Exception as exc:
                logger.warning(
                    f"Could not load drafter card {drafter_id} for "
                    f"{target_model_id}; skipping drafter download: {exc}"
                )
                continue

            drafter_shard = PipelineShardMetadata(
                model_card=drafter_card,
                device_rank=0,
                world_size=1,
                start_layer=0,
                end_layer=drafter_card.n_layers,
                n_layers=drafter_card.n_layers,
            )
            logger.info(
                f"Chaining drafter download {drafter_id} for {target_model_id}"
            )
            await self._start_download(drafter_shard)
            chained.append(drafter_id)

        if chained:
            # Replace any prior children list rather than extending so
            # repeated chain runs (e.g. coordinator restart, drafter
            # field upgrade) reflect the current set of chained drafters
            # rather than accumulating stale entries.
            self._drafter_children[target_model_id] = chained

    def _start_download_task(
        self, shard: ShardMetadata, initial_progress: RepoDownloadProgress
    ) -> None:
        model_id = shard.model_card.model_id

        # Emit ongoing status
        status = DownloadOngoing(
            node_id=self.node_id,
            shard_metadata=shard,
            download_progress=map_repo_download_progress_to_download_progress_data(
                initial_progress
            ),
            model_directory=self._default_model_dir(model_id),
        )
        self.download_status[model_id] = status
        self.event_sender.send_nowait(NodeDownloadProgress(download_progress=status))

        async def download_wrapper(cancel_scope: anyio.CancelScope) -> None:
            try:
                with cancel_scope:
                    await self.shard_downloader.ensure_shard(shard)
            except Exception as e:
                logger.error(f"Download failed for {model_id}: {e}")
                failed = DownloadFailed(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    error_message=str(e),
                    model_directory=self._default_model_dir(model_id),
                )
                self.download_status[model_id] = failed
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=failed)
                )
            except anyio.get_cancelled_exc_class():
                # ignore cancellation - let cleanup do its thing
                pass
            finally:
                self.active_downloads.pop(model_id, None)

        scope = anyio.CancelScope()
        self._tg.start_soon(download_wrapper, scope)
        self.active_downloads[model_id] = scope

    async def _delete_download(self, model_id: ModelId) -> None:
        # Protect read-only models from deletion
        if model_id in self.download_status:
            current = self.download_status[model_id]
            if isinstance(current, DownloadCompleted) and current.read_only:
                logger.warning(f"Refusing to delete read-only model {model_id}")
                return

        # Cancel if active
        if model_id in self.active_downloads:
            logger.info(f"Cancelling active download for {model_id} before deletion")
            self.active_downloads[model_id].cancel()

        # Cascade cancellation/deletion to chained drafters: the user
        # is removing the target's download intent, so the drafters
        # spawned alongside it should not keep running or stay on disk
        # past the target's lifetime. Pop the mapping so we don't
        # double-cascade on a subsequent delete of the same target.
        children = self._drafter_children.pop(model_id, [])
        for child_model_id in children:
            if (
                child_model_id in self.active_downloads
                or child_model_id in self.download_status
            ):
                logger.info(
                    f"Deleting chained drafter {child_model_id} alongside "
                    f"target {model_id}"
                )
                await self._delete_download(child_model_id)

        # Delete from disk
        logger.info(f"Deleting model files for {model_id}")
        try:
            deleted = await delete_model(model_id)
        except Exception:
            logger.exception(f"Failed to delete model files for {model_id}")
            return

        if deleted:
            logger.info(f"Successfully deleted model {model_id}")
        else:
            logger.warning(f"Model {model_id} was not found on disk")

        # Emit pending status to reset UI state, then remove from local tracking
        if model_id in self.download_status:
            current_status = self.download_status[model_id]
            pending = DownloadPending(
                shard_metadata=current_status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._default_model_dir(model_id),
            )
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )
            del self.download_status[model_id]

    async def _emit_existing_download_progress(self) -> None:
        while True:
            try:
                logger.debug(
                    "DownloadCoordinator: Fetching and emitting existing download progress..."
                )
                async for (
                    _,
                    progress,
                ) in self.shard_downloader.get_shard_download_status():
                    model_id = progress.shard.model_card.model_id

                    # Active downloads emit progress via the callback — don't overwrite
                    if model_id in self.active_downloads:
                        continue

                    if progress.status == "complete":
                        found = await to_thread.run_sync(
                            resolve_existing_model,
                            model_id,
                            progress.shard.model_card,
                        )
                        if found is not None:
                            status: DownloadProgress = self._completed_from_path(
                                progress.shard, found, progress.total
                            )
                        else:
                            status = DownloadCompleted(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                total=progress.total,
                                model_directory=self._default_model_dir(model_id),
                            )
                    elif progress.status in ["in_progress", "not_started"]:
                        # TODO(ciaran): temporary solution
                        # Don't downgrade a model that is already confirmed complete.
                        if isinstance(
                            self.download_status.get(model_id), DownloadCompleted
                        ):
                            continue
                        # The per-file size check compares local files against
                        # the latest HF "main" revision, which is a moving
                        # target.  When HF updates text files (README, YAML,
                        # jinja) in a new commit, the cached file list has new
                        # sizes while local files still match the old revision.
                        # Fall back to the authoritative completeness check
                        # (is_model_directory_complete) which validates that all
                        # safetensors weight files are present.
                        found = await to_thread.run_sync(
                            resolve_existing_model,
                            model_id,
                            progress.shard.model_card,
                        )
                        if found is not None:
                            status = self._completed_from_path(
                                progress.shard, found, progress.total
                            )
                        elif progress.downloaded.in_bytes == 0:
                            continue
                        elif progress.downloaded_this_session.in_bytes == 0:
                            status = DownloadPending(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                model_directory=self._default_model_dir(model_id),
                                downloaded=progress.downloaded,
                                total=progress.total,
                            )
                        else:
                            status = DownloadOngoing(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                download_progress=map_repo_download_progress_to_download_progress_data(
                                    progress
                                ),
                                model_directory=self._default_model_dir(model_id),
                            )
                    else:
                        continue

                    self.download_status[progress.shard.model_card.model_id] = status
                    await self.event_sender.send(
                        NodeDownloadProgress(download_progress=status)
                    )
                # Scan read-only directories for pre-downloaded models
                if EXO_MODELS_READ_ONLY_DIRS:
                    for card in await get_model_cards():
                        mid = card.model_id
                        if mid in self.active_downloads:
                            continue
                        if isinstance(
                            self.download_status.get(mid),
                            (DownloadCompleted, DownloadOngoing, DownloadFailed),
                        ):
                            continue
                        found = await to_thread.run_sync(
                            resolve_existing_model, mid, card
                        )
                        if found is not None and is_read_only_model_dir(found):
                            path_shard = PipelineShardMetadata(
                                model_card=card,
                                device_rank=0,
                                world_size=1,
                                start_layer=0,
                                end_layer=card.n_layers,
                                n_layers=card.n_layers,
                            )
                            path_completed: DownloadProgress = (
                                self._completed_from_path(
                                    path_shard, found, card.storage_size
                                )
                            )
                            self.download_status[mid] = path_completed
                            await self.event_sender.send(
                                NodeDownloadProgress(download_progress=path_completed)
                            )

                logger.debug(
                    "DownloadCoordinator: Done emitting existing download progress."
                )
            except Exception as e:
                logger.error(
                    f"DownloadCoordinator: Error emitting existing download progress: {e}"
                )
            await anyio.sleep(60)
