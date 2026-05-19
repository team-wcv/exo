# pyright: reportPrivateUsage=false

import pytest

from exo.routing.event_router import EventRouter
from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import GlobalForwarderEvent, LocalForwarderEvent
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
