# pyright: reportPrivateUsage=false

import anyio
import pytest

from exo.routing.event_router import (
    EventDeliveryStalledError,
    EventRouter,
    _PendingDelivery,
)
from exo.shared.apply import apply
from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    StateSnapshot,
    TestEvent,
    TracesCollected,
)
from exo.shared.types.state import State
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
@pytest.mark.parametrize(("gap", "limit", "expected"), [(10, 2, 2), (240000, None, 64)])
async def test_gap_replay_batches_are_capped(
    gap: int, limit: int | None, expected: int
) -> None:
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
    if limit is not None:
        router._nack_max_events = limit

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        await global_sender.send(
            GlobalForwarderEvent(
                origin=NodeId("master"),
                origin_idx=gap,
                session=session_id,
                event=TestEvent(),
            )
        )

        command = (await command_receiver.receive()).command
        assert isinstance(command, RequestEventLog)
        assert command.since_idx == 0
        assert command.max_events == expected
        assert command.allow_initial_snapshot

        router.shutdown()
        task_group.cancel_scope.cancel()

    global_sender.close()
    local_receiver.close()


@pytest.mark.asyncio
async def test_initial_snapshot_skips_history_and_preserves_live_event_order() -> None:
    commands, _ = channel[ForwarderCommand]()
    incoming, receiver = channel[GlobalForwarderEvent]()
    outgoing, _ = channel[LocalForwarderEvent]()
    session = SessionId(master_node_id=NodeId("master"), election_clock=3)
    router = EventRouter(session, commands, receiver, outgoing)
    internal = router.receiver()
    pending = _pending_test_event(session)
    future = _PendingDelivery(
        first_sent_at=10,
        last_sent_at=24,
        event=pending.event.model_copy(update={"origin_idx": 1, "event": TestEvent()}),
    )
    router.out_for_delivery = {
        pending.event.event.event_id: pending,
        future.event.event.event_id: future,
    }
    snapshot = StateSnapshot(
        target=router._system_id,
        state=State(last_event_applied_idx=240000),
        acknowledged_origins={SystemId("worker"): 1},
    )

    async def send(idx: int, event: Event) -> None:
        await incoming.send(
            GlobalForwarderEvent(
                origin=session.master_node_id,
                origin_idx=idx,
                session=session,
                event=event,
            )
        )

    async with anyio.create_task_group() as tg:
        tg.start_soon(router.run)
        await send(240001, TestEvent())
        await send(240000, snapshot.model_copy(update={"target": SystemId("other")}))
        await send(240000, snapshot)
        with anyio.fail_after(2):
            first = await internal.receive()
            second = await internal.receive()
        assert isinstance(first.event, StateSnapshot)
        assert first.idx == 240000 and second.idx == 240001
        state = apply(apply(State(), first), second)
        assert state.last_event_applied_idx == 240001
        assert pending.event.event.event_id not in router.out_for_delivery
        assert future.event.event.event_id in router.out_for_delivery
        # A delayed snapshot cannot overwrite a consumer that has already started.
        await send(240000, snapshot)
        await send(240002, TestEvent())
        with anyio.fail_after(2):
            assert (await internal.receive()).idx == 240002
        router.shutdown()
        tg.cancel_scope.cancel()


def test_snapshot_round_trip_and_empty_consumer_requirement() -> None:
    snapshot = StateSnapshot(
        target=SystemId("worker"),
        state=State(last_event_applied_idx=2000),
        acknowledged_origins={SystemId("origin"): 4},
    )
    wrapped = GlobalForwarderEvent(
        origin=NodeId("master"),
        origin_idx=2000,
        session=SessionId(master_node_id=NodeId("master"), election_clock=1),
        event=snapshot,
    )
    restored = GlobalForwarderEvent.model_validate_json(wrapped.model_dump_json())
    assert isinstance(restored.event, StateSnapshot)
    assert restored.event.acknowledged_origins == {SystemId("origin"): 4}
    from exo.shared.types.events import IndexedEvent

    indexed = IndexedEvent(idx=2000, event=restored.event)
    with pytest.raises(ValueError, match="empty consumer"):
        apply(State(last_event_applied_idx=10), indexed)
    with pytest.raises(ValueError, match="differs from its envelope"):
        apply(State(), indexed.model_copy(update={"idx": 1999}))


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
async def test_trace_only_events_retry_until_producer_progress() -> None:
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
        assert trace_event.event_id in router.out_for_delivery
        router._delivery_stall_seconds = 1.0
        router._raise_if_delivery_stalled(now=anyio.current_time() + 60.0)

        router._acknowledge_prior_trace_events(
            LocalForwarderEvent(
                origin_idx=forwarded.origin_idx + 1,
                origin=forwarded.origin,
                session=session_id,
                event=TestEvent(),
            )
        )
        assert trace_event.event_id not in router.out_for_delivery
        event_sender.close()

    command_receiver.close()
    global_sender.close()
    local_receiver.close()
