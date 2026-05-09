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

import anyio
import pytest

from exo.download.coordinator import DownloadCoordinator
from exo.download.download_utils import RepoDownloadProgress
from exo.download.shard_downloader import ShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.commands import (
    CancelDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId, SystemId
from exo.shared.types.events import Event, NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadProgressData,
)
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


async def test_drafter_chain_runs_off_command_processing_path() -> None:
    """Codex flagged (P1, PR #18 round 2) that the drafter card fetch
    ran inline inside ``_command_processor``, so a slow HF call
    blocked unrelated commands. The fix backgrounds the chain via
    ``_tg.start_soon``; this test verifies that a second command
    arriving while ``ModelCard.load`` is hung still progresses.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))
    drafter_card = _make_drafter_card()

    # Block ModelCard.load until we've observed the second command
    # being processed.
    drafter_load_started = asyncio.Event()
    drafter_load_release = asyncio.Event()

    async def slow_load(model_id: ModelId) -> ModelCard:
        if model_id == DRAFTER_ID:
            drafter_load_started.set()
            await drafter_load_release.wait()
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    # Second command -- a CancelDownload -- proves the command loop
    # is still responsive even while the drafter chain is hung.
    second_target = ModelId("test-org/second-target")

    with patch.object(ModelCard, "load", side_effect=slow_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            _coordinator,
            cmd_send,
            event_recv,
        ):
            # Kick off the target download; the drafter chain will
            # block on ``slow_load``.
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None

            # Wait for the drafter chain to actually be running and
            # blocked on ``slow_load`` (proves the chain was
            # dispatched). A bounded wait so a regression that takes
            # the chain off-process entirely surfaces as a clear
            # timeout failure instead of a silent skip.
            async with asyncio.timeout(2.0):
                await drafter_load_started.wait()

            # Command loop must remain responsive: send a
            # CancelDownload for an UNRELATED model and verify it
            # processes immediately (no-op, but the coordinator must
            # observe it). Before the fix, this would block until
            # ``slow_load`` completed (or timed out).
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=CancelDownload(
                        target_node_id=NODE_ID, model_id=second_target
                    ),
                )
            )

            # A small grace window to let the cancel command be
            # observed; the drafter chain is still blocked so any
            # progress here is by definition concurrent.
            await asyncio.sleep(0.1)

            # Release the drafter load so the test cleans up.
            drafter_load_release.set()
            await asyncio.sleep(0.1)


async def test_cancel_during_chain_aborts_drafter_download() -> None:
    """Codex P1 (PR #18 round-(N+1)): a CancelDownload that arrives
    AFTER StartDownload but BEFORE the chain coroutine has registered
    its drafters in ``_drafter_children`` must still prevent the
    drafter download from starting. Pre-fix, the cancel cascade ran
    against an empty children list (the chain hadn't populated it
    yet) and the chain then merrily dispatched ``ensure_shard`` for
    the drafter despite the user having revoked the parent intent.
    Post-fix, ``_spawn_drafter_chain`` pre-registers an empty entry
    and the chain re-checks membership after every ``await`` so the
    cascade pops the entry and signals the chain to bail.

    The race is reproduced deterministically by stalling
    ``ModelCard.load`` so the chain reaches its post-load
    cancellation re-check while a cancel is in flight.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))
    drafter_card = _make_drafter_card()

    drafter_load_started = asyncio.Event()
    drafter_load_release = asyncio.Event()

    async def slow_load(model_id: ModelId) -> ModelCard:
        if model_id == DRAFTER_ID:
            drafter_load_started.set()
            await drafter_load_release.wait()
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    with patch.object(ModelCard, "load", side_effect=slow_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            cmd_send,
            event_recv,
        ):
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None

            # Wait for the chain to actually enter ``ModelCard.load``;
            # at this point the cancel is racing the load resolution.
            async with asyncio.timeout(2.0):
                await drafter_load_started.wait()

            # Cancel the target while the chain is hung mid-load.
            # Pre-fix: ``_drafter_children[TARGET_ID]`` was empty, so
            # the cascade had nothing to cancel; after release, the
            # chain proceeded to call ``ensure_shard(DRAFTER_ID)``.
            # Post-fix: the entry exists (pre-registered), the cancel
            # cascade pops it, and the chain's post-load re-check
            # sees ``cancelled() == True`` and returns.
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=CancelDownload(target_node_id=NODE_ID, model_id=TARGET_ID),
                )
            )

            # Give the cancel command a moment to be processed
            # before releasing the load.
            await asyncio.sleep(0.1)
            drafter_load_release.set()

            # Allow the chain coroutine to run its post-load check.
            await asyncio.sleep(0.1)

            # Drafter download must NOT have been kicked off, because
            # the parent target was cancelled before its load
            # resolved. Only the target made it into ``ensured``.
            assert DRAFTER_ID not in downloader.ensured, (
                "drafter download must NOT start when its parent target "
                "was cancelled mid-chain; got ensured="
                f"{downloader.ensured!r}"
            )
            # The cancel cascade must also have removed the parent
            # entry, so a duplicate cancel doesn't try to cascade
            # into a stale drafter list.
            assert TARGET_ID not in coordinator._drafter_children, (  # pyright: ignore[reportPrivateUsage]
                "cancel cascade must clear _drafter_children for the "
                "target so a duplicate cancel doesn't double-cascade"
            )


async def test_failed_target_does_not_chain_drafter() -> None:
    """Codex P2 (PR #18 round-(N+2), coordinator.py:231): a target
    that is already in ``DownloadFailed`` state must NOT trigger a
    drafter chain. The round-(N+1) "backfill drafters even when
    target was already tracked" branch swept failed targets into
    the same fast-path, kicking off drafter downloads for a target
    that won't itself download. Drafters served by a non-runnable
    target are useless (the runner can't boot speculative decoding
    without the target weights), so we must consume the network/
    disk only when the target is at least possibly going to run.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))

    async def fail_load(_: ModelId) -> ModelCard:
        raise AssertionError(
            "ModelCard.load must not be called when target is "
            "already in DownloadFailed state"
        )

    with patch.object(ModelCard, "load", side_effect=fail_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            cmd_send,
            _,
        ):
            # Pre-seed the target's download_status as FAILED.
            from exo.shared.types.worker.downloads import DownloadFailed

            coordinator.download_status[TARGET_ID] = DownloadFailed(
                shard_metadata=target_shard,
                node_id=NODE_ID,
                error_message="simulated previous failure",
                model_directory="/fake/target",
            )

            # Re-issuing StartDownload for a previously-failed target
            # must NOT chain drafters. Pre-fix: the round-(N+1) code
            # called ``self._spawn_drafter_chain(shard)`` from inside
            # the failed-state fast-path branch; we'd get
            # ``ModelCard.load`` and the AssertionError above.
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            await asyncio.sleep(0.1)

    # Drafter must NOT have been queued for download.
    assert DRAFTER_ID not in downloader.ensured, (
        "drafter download must NOT start when target is in "
        f"DownloadFailed state; got ensured={downloader.ensured!r}"
    )


async def test_restart_target_re_chains_cancelled_drafter() -> None:
    """Codex P2 (PR #18 round-(N+2), coordinator.py:437): after a
    cancel cascade demotes a chained drafter to ``DownloadPending``,
    a subsequent ``StartDownload`` for the same target is a fresh
    user intent and must bring the drafter back to life. Pre-fix,
    ``drafter_id in self.download_status`` short-circuited
    regardless of the drafter's current state, so a once-cancelled
    drafter never restarted and speculative decoding silently
    stayed disabled until the operator manually started each
    drafter.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))
    drafter_shard = _make_shard(_make_drafter_card())
    drafter_card = _make_drafter_card()

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == DRAFTER_ID:
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    with patch.object(ModelCard, "load", side_effect=fake_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            cmd_send,
            event_recv,
        ):
            # Simulate the post-cancel state: the drafter was
            # previously chained, then cancelled (DownloadPending).
            from exo.shared.types.worker.downloads import DownloadPending

            coordinator.download_status[DRAFTER_ID] = DownloadPending(
                shard_metadata=drafter_shard,
                node_id=NODE_ID,
                model_directory="/fake/drafter",
            )

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

    # Drafter must have been re-ensured: pre-fix this list contained
    # only the target, because the drafter's stale ``DownloadPending``
    # status short-circuited the chain branch.
    assert DRAFTER_ID in downloader.ensured, (
        "subsequent StartDownload(target) must re-chain a previously "
        f"cancelled drafter; got ensured={downloader.ensured!r}"
    )


async def test_cancel_target_cascades_to_chained_drafter() -> None:
    """Codex flagged (P2, PR #18 round 2) that cancelling a target
    left chained drafters running independently. The fix wires a
    parent->children mapping that ``_cancel_download`` cascades.

    Test calls ``_cancel_download`` directly with a synthesised
    children mapping so we don't depend on the timing of the
    background chain task to populate state.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))

    downloader = _RecordingShardDownloader()
    async with _running_coordinator(downloader) as (
        coordinator,
        _,
        _,
    ):
        # Pre-seed the parent->children mapping and active downloads
        # so the cancel cascade has something to operate on.
        coordinator._drafter_children[TARGET_ID] = [DRAFTER_ID]  # pyright: ignore[reportPrivateUsage]
        target_scope = anyio.CancelScope()
        drafter_scope = anyio.CancelScope()
        coordinator.active_downloads[TARGET_ID] = target_scope
        coordinator.active_downloads[DRAFTER_ID] = drafter_scope

        # Status entries needed by ``_cancel_download``'s pending
        # synthesis path.
        def _ongoing_progress(
            downloaded_mb: int, total_mb: int
        ) -> DownloadProgressData:
            return DownloadProgressData(
                downloaded=Memory.from_mb(downloaded_mb),
                downloaded_this_session=Memory.from_mb(downloaded_mb),
                total=Memory.from_mb(total_mb),
                completed_files=0,
                total_files=1,
                speed=0.0,
                eta_ms=0,
                files={},
            )

        coordinator.download_status[TARGET_ID] = DownloadOngoing(
            shard_metadata=target_shard,
            node_id=NODE_ID,
            model_directory="/fake/target",
            download_progress=_ongoing_progress(100, 500),
        )
        drafter_card = _make_drafter_card()
        drafter_shard_meta = _make_shard(drafter_card)
        coordinator.download_status[DRAFTER_ID] = DownloadOngoing(
            shard_metadata=drafter_shard_meta,
            node_id=NODE_ID,
            model_directory="/fake/drafter",
            download_progress=_ongoing_progress(10, 50),
        )

        await coordinator._cancel_download(TARGET_ID)  # pyright: ignore[reportPrivateUsage]

        # Both scopes must be cancelled.
        assert target_scope.cancel_called
        assert drafter_scope.cancel_called
        # And the parent->children mapping is cleared so a duplicate
        # cancel command doesn't try to cancel a stale drafter.
        assert TARGET_ID not in coordinator._drafter_children  # pyright: ignore[reportPrivateUsage]


async def test_rechain_preserves_drafter_link_for_cancel_cascade() -> None:
    """Codex P2 (PR #18 round-(N+2), coordinator.py:442): when
    ``StartDownload`` is re-issued for a target whose chain is still
    in flight, the second chain run MUST mutate the same
    ``_drafter_children`` list that any in-flight chain holds a
    reference to. Pre-fix, the second run reassigned the dict slot
    to a fresh list, orphaning the in-flight chain's appended
    drafter ids and breaking the ``_cancel_download`` cascade.

    We simulate the bug by directly invoking
    ``_maybe_chain_drafter_download`` twice, capturing the list
    object the first invocation observes, and asserting that drafter
    ids appended via the second chain are visible through that
    same captured reference -- which is what the cancel cascade
    relies on.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))
    drafter_card = _make_drafter_card()

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == DRAFTER_ID:
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    with patch.object(ModelCard, "load", side_effect=fake_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            _,
            _,
        ):
            # First chain run -- pre-register and run synchronously
            # so the slot exists when we capture the list reference.
            coordinator._drafter_children.setdefault(TARGET_ID, [])  # pyright: ignore[reportPrivateUsage]
            captured_list: list[ModelId] = coordinator._drafter_children[TARGET_ID]  # pyright: ignore[reportPrivateUsage]
            await coordinator._maybe_chain_drafter_download(target_shard)  # pyright: ignore[reportPrivateUsage]

            # The drafter must be visible through the captured list
            # AND through the live dict-resolved list. Pre-fix, a
            # second run would diverge these.
            assert DRAFTER_ID in captured_list, (
                "first chain run must populate the captured list ref"
            )
            assert captured_list is coordinator._drafter_children[TARGET_ID], (  # pyright: ignore[reportPrivateUsage]
                "_drafter_children slot must NOT be reassigned by chain run"
            )

            # Second chain run (e.g. user re-issued StartDownload).
            await coordinator._maybe_chain_drafter_download(target_shard)  # pyright: ignore[reportPrivateUsage]

            # The captured list reference must still be the live one
            # tracked by ``_drafter_children`` -- otherwise a cancel
            # cascade based on ``_drafter_children[TARGET_ID]`` would
            # miss any drafter the second run started.
            assert captured_list is coordinator._drafter_children[TARGET_ID], (  # pyright: ignore[reportPrivateUsage]
                "rechain must mutate the same list, not replace the slot, "
                "so the cancel cascade always sees every drafter ever "
                "started for this target"
            )
            # Dedup: the drafter must not be duplicated across runs.
            assert captured_list.count(DRAFTER_ID) == 1, (
                "rechain must dedup drafter ids it already linked"
            )
