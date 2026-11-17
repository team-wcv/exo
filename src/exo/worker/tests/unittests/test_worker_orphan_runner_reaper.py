# pyright: reportPrivateUsage=false

from typing import cast

import pytest

from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.events import Event, IndexedEvent, RunnerStatusUpdated
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerReady,
    RunnerShutdown,
    RunnerShuttingDown,
)
from exo.utils.channels import channel
from exo.worker.main import ORPHAN_RUNNER_GRACE_SECONDS, Worker
from exo.worker.runner.supervisor import RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


@pytest.mark.anyio
async def test_reaper_reports_only_true_orphans_after_grace_period() -> None:
    _, receiver = channel[IndexedEvent]()
    event_sender, event_receiver = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        node_id=NodeId("node-a"),
        event_receiver=receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        api_port=52415,
        peer_download_port=24216,
    )

    referenced = RunnerId("runner-a")
    instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=referenced,
        node_id=NodeId("node-a"),
    ).instance
    orphan = RunnerId("orphan-runner")
    supervised = RunnerId("supervised-runner")
    ready = RunnerId("ready-runner")
    worker.state = State(
        instances={instance.instance_id: instance},
        runners={
            orphan: RunnerShuttingDown(),
            referenced: RunnerShuttingDown(),
            supervised: RunnerShuttingDown(),
            ready: RunnerReady(),
        },
    )
    worker.runners = {supervised: cast(RunnerSupervisor, cast(object, object()))}

    orphan_since: dict[RunnerId, float] = {}
    await worker._reap_orphan_runners_pass(100.0, orphan_since)
    await worker._reap_orphan_runners_pass(
        100.0 + ORPHAN_RUNNER_GRACE_SECONDS - 1, orphan_since
    )
    assert event_receiver.collect() == []

    await worker._reap_orphan_runners_pass(
        100.0 + ORPHAN_RUNNER_GRACE_SECONDS, orphan_since
    )
    reported = event_receiver.collect()
    assert len(reported) == 1
    assert isinstance(reported[0], RunnerStatusUpdated)
    assert reported[0].runner_id == orphan
    assert isinstance(reported[0].runner_status, RunnerShutdown)
    assert orphan_since == {}
