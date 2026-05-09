"""Tests for chained drafter downloads in :class:`DownloadCoordinator`.

When a target model card declares ``drafter_model_id``, kicking off a
download for the target should also kick off a download for the matching
drafter so single-device speculative decoding works without manual setup.

These tests stub the underlying ``ShardDownloader`` and the model-card
loader so they can run in CI without touching HuggingFace or the disk.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable
from datetime import timedelta
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

from exo.download.coordinator import DownloadCoordinator
from exo.download.download_utils import RepoDownloadProgress
from exo.download.shard_downloader import ShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.commands import (
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId, SystemId
from exo.shared.types.events import Event, NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender, channel

NODE_ID = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
TARGET_ID = ModelId("test-org/test-target")
DRAFTER_ID = ModelId("test-org/test-drafter")


def _make_target_card(drafters: list[ModelId]) -> ModelCard:
    return ModelCard(
        model_id=TARGET_ID,
        storage_size=Memory.from_mb(500),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=drafters,
    )


def _make_drafter_card() -> ModelCard:
    return ModelCard(
        model_id=DRAFTER_ID,
        storage_size=Memory.from_mb(50),
        n_layers=12,
        hidden_size=768,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )


def _make_shard(card: ModelCard) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )


class _RecordingShardDownloader(ShardDownloader):
    """Records every shard ``ensure_shard`` is called on and reports
    ``status='complete'`` immediately so the coordinator advances to a
    terminal state."""

    def __init__(self) -> None:
        self.ensured: list[ModelId] = []
        self._progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self._progress_callbacks.append(callback)

    async def ensure_shard(
        self,
        shard: ShardMetadata,
        config_only: bool = False,  # noqa: ARG002
    ) -> Path:
        self.ensured.append(shard.model_card.model_id)
        progress = RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision="main",
            shard=shard,
            completed_files=1,
            total_files=1,
            downloaded=shard.model_card.storage_size,
            downloaded_this_session=shard.model_card.storage_size,
            total=shard.model_card.storage_size,
            overall_speed=0,
            overall_eta=timedelta(seconds=0),
            status="complete",
        )
        for cb in self._progress_callbacks:
            await cb(shard, progress)
        return Path("/fake/models") / shard.model_card.model_id.normalize()

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        if False:  # noqa: SIM108  # empty async generator
            yield (
                Path(),
                RepoDownloadProgress(  # pyright: ignore[reportUnreachable]
                    repo_id="",
                    repo_revision="",
                    shard=_make_shard(_make_drafter_card()),
                    completed_files=0,
                    total_files=0,
                    downloaded=Memory.from_bytes(0),
                    downloaded_this_session=Memory.from_bytes(0),
                    total=Memory.from_bytes(0),
                    overall_speed=0,
                    overall_eta=timedelta(seconds=0),
                    status="not_started",
                ),
            )

    async def get_shard_download_status_for_shard(
        self,
        shard: ShardMetadata,
    ) -> RepoDownloadProgress:
        return RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision="main",
            shard=shard,
            completed_files=0,
            total_files=1,
            downloaded=Memory.from_bytes(0),
            downloaded_this_session=Memory.from_bytes(0),
            total=shard.model_card.storage_size,
            overall_speed=0,
            overall_eta=timedelta(seconds=0),
            status="not_started",
        )


async def _wait_for_completed(
    event_recv: Receiver[Event], model_id: ModelId, timeout: float = 2.0
) -> DownloadCompleted | None:
    try:
        async with asyncio.timeout(timeout):
            while True:
                event = await event_recv.receive()
                if (
                    isinstance(event, NodeDownloadProgress)
                    and isinstance(event.download_progress, DownloadCompleted)
                    and event.download_progress.shard_metadata.model_card.model_id
                    == model_id
                ):
                    return event.download_progress
    except TimeoutError:
        return None


@contextlib.asynccontextmanager
async def _running_coordinator(
    downloader: _RecordingShardDownloader,
    *,
    offline: bool = False,
) -> AsyncIterator[
    tuple[
        DownloadCoordinator,
        Sender[ForwarderDownloadCommand],
        Receiver[Event],
    ]
]:
    cmd_send: Sender[ForwarderDownloadCommand]
    cmd_send, cmd_recv = channel[ForwarderDownloadCommand]()
    event_send, event_recv = channel[Event]()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=downloader,
        download_command_receiver=cmd_recv,
        event_sender=event_send,
        offline=offline,
    )
    coordinator_task = asyncio.create_task(coordinator.run())
    try:
        yield coordinator, cmd_send, event_recv
    finally:
        await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task


async def test_target_with_drafter_chains_drafter_download() -> None:
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))
    drafter_card = _make_drafter_card()

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == DRAFTER_ID:
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    with patch.object(ModelCard, "load", side_effect=fake_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (_, cmd_send, event_recv):
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None
            assert await _wait_for_completed(event_recv, DRAFTER_ID) is not None

    assert TARGET_ID in downloader.ensured
    assert DRAFTER_ID in downloader.ensured


async def test_target_without_drafter_does_not_chain() -> None:
    target_shard = _make_shard(_make_target_card([]))

    async def fail_load(_: ModelId) -> ModelCard:
        raise AssertionError("ModelCard.load should not be called when no drafter")

    with patch.object(ModelCard, "load", side_effect=fail_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (_, cmd_send, event_recv):
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None
            await asyncio.sleep(0.05)

    assert downloader.ensured == [TARGET_ID]


async def test_drafter_chain_skipped_when_disabled_by_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_DISABLE_DRAFTER", "1")
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))

    async def fail_load(_: ModelId) -> ModelCard:
        raise AssertionError(
            "ModelCard.load should not be called when EXO_DISABLE_DRAFTER set"
        )

    with patch.object(ModelCard, "load", side_effect=fail_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (_, cmd_send, event_recv):
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None
            await asyncio.sleep(0.05)

    assert downloader.ensured == [TARGET_ID]


async def test_drafter_chain_swallows_card_load_error() -> None:
    """If the drafter's ModelCard cannot be loaded (e.g. HF unreachable, card
    not in repo), the target download must still complete and the coordinator
    must not crash."""
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))

    async def boom(_: ModelId) -> ModelCard:
        raise RuntimeError("simulated card load failure")

    with patch.object(ModelCard, "load", side_effect=boom):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (_, cmd_send, event_recv):
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None
            await asyncio.sleep(0.05)

    assert downloader.ensured == [TARGET_ID]


async def test_drafter_chain_skipped_in_offline_mode() -> None:
    """Offline-mode coordinators must NOT call ``ModelCard.load`` for
    declared drafters even when the target download itself is locally
    complete.

    ``ModelCard.load`` falls through to ``ModelCard.fetch_from_hf``
    whenever the drafter card isn't already in ``_card_cache``. Under
    ``EXO_OFFLINE=true`` that's an outbound HuggingFace request that
    can stall command processing for the full client timeout before
    the eventual ``DownloadFailed`` is swallowed by the silent
    best-effort drafter chain. The fix short-circuits
    ``_maybe_chain_drafter_download`` when ``self.offline`` is True
    so no card resolution is attempted.

    Test calls ``_maybe_chain_drafter_download`` directly so the
    assertion is precise: ``ModelCard.load`` is the network entry
    point, and the test fails immediately if the offline guard
    regresses to letting it fire.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))

    async def fail_load(_: ModelId) -> ModelCard:
        raise AssertionError(
            "ModelCard.load must not be called in offline mode "
            "(would trigger a HuggingFace fetch)"
        )

    with patch.object(ModelCard, "load", side_effect=fail_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader, offline=True) as (
            coordinator,
            _,
            _,
        ):
            await coordinator._maybe_chain_drafter_download(  # pyright: ignore[reportPrivateUsage]
                target_shard
            )
            await asyncio.sleep(0.05)

    # No drafter download was ever queued because the chain
    # short-circuited before ``ModelCard.load``.
    assert downloader.ensured == []
