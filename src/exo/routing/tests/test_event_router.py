# pyright: reportPrivateUsage=false

import anyio
import pytest

from exo.routing.event_router import (
    EventDeliveryStalledError,
    EventRouter,
    _PendingDelivery,
)
from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    TestEvent,
    TracesCollected,
)
from exo.shared.types.tasks import TaskId
from exo.utils.channels import channel


def _pending_test_event(session_id: SessionId) -> _PendingDelivery:
    event = TestEvent()
    return _PendingDelivery(
        first_sent_at=10.0,
        last_sent_at=24.0,
        event=LocalForwarderEvent(
            origin_idx=0,
            origin=SystemId("worker"),
            session=session_id,
            event=event,
        ),
    )


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

    await router._nack_request(since_idx=42, max_events=1)

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
        assert first.command.max_events == 3

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
        assert second.command.max_events == 2
        applied = await internal_receiver.receive()
        assert applied.idx == 0

        router.shutdown()
        task_group.cancel_scope.cancel()

    global_sender.close()
    local_receiver.close()


@pytest.mark.asyncio
async def test_gap_replay_batches_are_capped() -> None:
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
    router._nack_max_events = 2

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        await global_sender.send(
            GlobalForwarderEvent(
                origin=NodeId("master"),
                origin_idx=10,
                session=session_id,
                event=TestEvent(),
            )
        )

        command = (await command_receiver.receive()).command
        assert isinstance(command, RequestEventLog)
        assert command.since_idx == 0
        assert command.max_events == 2

        router.shutdown()
        task_group.cancel_scope.cancel()

    global_sender.close()
    local_receiver.close()


def test_delivery_watchdog_uses_original_send_time() -> None:
    command_sender, _ = channel[ForwarderCommand]()
    _, global_receiver = channel[GlobalForwarderEvent]()
    local_sender, _ = channel[LocalForwarderEvent]()
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=0)
    router = EventRouter(
        session_id=session_id,
        command_sender=command_sender,
        external_inbound=global_receiver,
        external_outbound=local_sender,
    )
    router._delivery_stall_seconds = 15.0
    pending = _pending_test_event(session_id)
    router.out_for_delivery[pending.event.event.event_id] = pending

    with pytest.raises(EventDeliveryStalledError, match="half-open routing session"):
        router._raise_if_delivery_stalled(now=25.0)


def test_delivery_watchdog_allows_recent_or_disabled_events() -> None:
    command_sender, _ = channel[ForwarderCommand]()
    _, global_receiver = channel[GlobalForwarderEvent]()
    local_sender, _ = channel[LocalForwarderEvent]()
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=0)
    router = EventRouter(
        session_id=session_id,
        command_sender=command_sender,
        external_inbound=global_receiver,
        external_outbound=local_sender,
    )
    pending = _pending_test_event(session_id)
    router.out_for_delivery[pending.event.event.event_id] = pending

    router._delivery_stall_seconds = 15.0
    router._raise_if_delivery_stalled(now=24.9)
    router._delivery_stall_seconds = 0.0
    router._raise_if_delivery_stalled(now=100.0)


@pytest.mark.asyncio
async def test_trace_only_events_are_not_tracked_for_acknowledgement() -> None:
    command_sender, command_receiver = channel[ForwarderCommand]()
    global_sender, global_receiver = channel[GlobalForwarderEvent]()
    local_sender, local_receiver = channel[LocalForwarderEvent]()
    event_sender, event_receiver = channel[Event]()
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=0)
    router = EventRouter(
        session_id=session_id,
        command_sender=command_sender,
        external_inbound=global_receiver,
        external_outbound=local_sender,
    )
    trace_event = TracesCollected(task_id=TaskId(), rank=0, traces=[])

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router._ingest, SystemId("worker"), event_receiver)
        await event_sender.send(trace_event)
        forwarded = await local_receiver.receive()
        assert forwarded.event == trace_event
        assert not router.out_for_delivery
        event_sender.close()

    command_receiver.close()
    global_sender.close()
    local_receiver.close()
