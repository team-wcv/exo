# pyright: reportPrivateUsage=false

import anyio
import pytest

from exo.routing.event_router import EventRouter
from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    GlobalForwarderEvent,
    LocalForwarderEvent,
    TestEvent,
)
from exo.utils.channels import channel


@pytest.mark.asyncio
async def test_nack_requests_only_the_missing_event() -> None:
    command_sender, command_receiver = channel[ForwarderCommand]()
    global_sender, global_receiver = channel[GlobalForwarderEvent]()
    local_sender, local_receiver = channel[LocalForwarderEvent]()
    router = EventRouter(
        session_id=SessionId(master_node_id=NodeId("master"), election_clock=0),
        command_sender=command_sender,
        external_inbound=global_receiver,
        external_outbound=local_sender,
    )
    router._nack_base_seconds = 0

    await router._nack_request(since_idx=42)

    commands = command_receiver.collect()
    assert len(commands) == 1
    command = commands[0].command
    assert isinstance(command, RequestEventLog)
    assert command.since_idx == 42
    assert command.max_events == 1

    global_sender.close()
    local_receiver.close()


@pytest.mark.asyncio
async def test_gap_replay_chains_when_buffer_still_has_future_events() -> None:
    command_sender, command_receiver = channel[ForwarderCommand]()
    global_sender, global_receiver = channel[GlobalForwarderEvent]()
    local_sender, local_receiver = channel[LocalForwarderEvent]()
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=0)
    router = EventRouter(
        session_id=session_id,
        command_sender=command_sender,
        external_inbound=global_receiver,
        external_outbound=local_sender,
    )
    router._nack_base_seconds = 0
    internal_receiver = router.receiver()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        await global_sender.send(
            GlobalForwarderEvent(
                origin=NodeId("master"),
                origin_idx=2,
                session=session_id,
                event=TestEvent(),
            )
        )
        first = await command_receiver.receive()
        assert isinstance(first.command, RequestEventLog)
        assert first.command.since_idx == 0
        assert first.command.max_events == 1

        await global_sender.send(
            GlobalForwarderEvent(
                origin=NodeId("master"),
                origin_idx=0,
                session=session_id,
                event=TestEvent(),
            )
        )
        second = await command_receiver.receive()
        assert isinstance(second.command, RequestEventLog)
        assert second.command.since_idx == 1
        assert second.command.max_events == 1
        applied = await internal_receiver.receive()
        assert applied.idx == 0

        router.shutdown()
        task_group.cancel_scope.cancel()

    global_sender.close()
    local_receiver.close()
