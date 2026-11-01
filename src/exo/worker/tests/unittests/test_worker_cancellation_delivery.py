# pyright: reportPrivateUsage=false

from types import SimpleNamespace
from typing import cast

import pytest

from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.events import Event, IndexedEvent, TaskDeleted, TaskStatusUpdated
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.utils.channels import channel
from exo.worker.main import Worker
from exo.worker.runner.supervisor import RunnerSupervisor


class Runner:
    def __init__(self, instance_id: InstanceId):
        self.bound_instance = SimpleNamespace(
            instance=SimpleNamespace(instance_id=instance_id)
        )
        self.received: list[TaskId] = []

    async def cancel_task(self, task_id: TaskId) -> None:
        self.received.append(task_id)


@pytest.mark.anyio
async def test_cancellation_survives_immediate_task_deletion() -> None:
    events, receiver = channel[IndexedEvent]()
    event_sender, _ = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        node_id=NodeId("worker"),
        event_receiver=receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        api_port=52415,
        peer_download_port=24216,
    )
    instance_id = InstanceId("instance")
    task = TextGeneration(
        task_id=TaskId("task"),
        instance_id=instance_id,
        command_id=CommandId("command"),
        task_params=TextGenerationTaskParams(model=ModelId("test/model"), input=[]),
    )
    worker.state = State(tasks={task.task_id: task})
    runner = Runner(instance_id)
    unrelated = Runner(InstanceId("other"))
    worker.runners = {
        RunnerId("runner"): cast(RunnerSupervisor, cast(object, runner)),
        RunnerId("unrelated"): cast(RunnerSupervisor, cast(object, unrelated)),
    }
    events.send_nowait(
        IndexedEvent(
            idx=0,
            event=TaskStatusUpdated(
                task_id=task.task_id, task_status=TaskStatus.Cancelled
            ),
        )
    )
    events.send_nowait(IndexedEvent(idx=1, event=TaskDeleted(task_id=task.task_id)))
    events.close()
    # No planner tick occurs between these ordered events, as when the API
    # closes a cancelled response and immediately announces TaskFinished.
    async with worker._tg:
        await worker._event_applier()
    assert task.task_id not in worker.state.tasks
    assert runner.received == [task.task_id]
    assert unrelated.received == []
