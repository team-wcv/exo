import os
from dataclasses import dataclass, field
from random import random

import anyio
from anyio import BrokenResourceError, ClosedResourceError
from anyio.abc import CancelScope
from loguru import logger

from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import SessionId, SystemId
from exo.shared.types.events import (
    Event,
    EventId,
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
)
from exo.utils import channels
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.event_buffer import OrderedBuffer
from exo.utils.task_group import TaskGroup


class EventDeliveryStalledError(RuntimeError):
    """Raised when pubsub stays half-open long enough to require a restart."""


@dataclass(frozen=True)
class _PendingDelivery:
    first_sent_at: float
    last_sent_at: float
    event: LocalForwarderEvent


def _delivery_stall_seconds() -> float:
    raw_value = os.getenv("EXO_EVENT_DELIVERY_STALL_SECONDS", "0")
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        logger.warning(
            "Invalid EXO_EVENT_DELIVERY_STALL_SECONDS value "
            f"{raw_value!r}; delivery watchdog disabled"
        )
        return 0.0


@dataclass
class EventRouter:
    session_id: SessionId
    command_sender: Sender[ForwarderCommand]
    external_inbound: Receiver[GlobalForwarderEvent]
    external_outbound: Sender[LocalForwarderEvent]
    _system_id: SystemId = field(init=False, default_factory=SystemId)
    internal_outbound: list[Sender[IndexedEvent]] = field(
        init=False, default_factory=list
    )
    event_buffer: OrderedBuffer[Event] = field(
        init=False, default_factory=OrderedBuffer
    )
    out_for_delivery: dict[EventId, _PendingDelivery] = field(
        init=False, default_factory=dict
    )
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)

    _nack_cancel_scope: CancelScope | None = field(init=False, default=None)
    _nack_attempts: int = field(init=False, default=0)
    _nack_base_seconds: float = field(init=False, default=0.5)
    _nack_cap_seconds: float = field(init=False, default=10.0)
    _nack_max_events: int = field(init=False, default=64)
    _last_outbound_warning_size: int = field(init=False, default=0)
    _delivery_stall_seconds: float = field(
        init=False, default_factory=_delivery_stall_seconds
    )

    async def run(self):
        try:
            async with self._tg as tg:
                tg.start_soon(self._run_ext_in)
                tg.start_soon(self._simple_retry)
        finally:
            self.external_outbound.close()
            for send in self.internal_outbound:
                send.close()

    # can make this better in future
    async def _simple_retry(self):
        while True:
            await anyio.sleep(1 + random())
            now = anyio.current_time()
            self._raise_if_delivery_stalled(now)
            # list here is a shallow clone for shared mutation
            for e_id, pending in list(self.out_for_delivery.items()):
                if now > pending.last_sent_at + 5:
                    self.out_for_delivery[e_id] = _PendingDelivery(
                        first_sent_at=pending.first_sent_at,
                        last_sent_at=now,
                        event=pending.event,
                    )
                    logger.debug(
                        "Retrying unacknowledged local event "
                        f"event_id={e_id} origin_idx={pending.event.origin_idx} "
                        f"event_type={type(pending.event.event).__name__} "
                        f"out_for_delivery={len(self.out_for_delivery)}"
                    )
                    await self.external_outbound.send(pending.event)

    def _raise_if_delivery_stalled(self, now: float) -> None:
        if self._delivery_stall_seconds <= 0:
            return
        for event_id, pending in self.out_for_delivery.items():
            age_seconds = now - pending.first_sent_at
            if age_seconds < self._delivery_stall_seconds:
                continue
            message = (
                "Event delivery stalled; terminating Exo so the service supervisor "
                "can rebuild the half-open routing session "
                f"event_id={event_id} "
                f"event_type={type(pending.event.event).__name__} "
                f"age_seconds={age_seconds:.3f} "
                f"out_for_delivery={len(self.out_for_delivery)} "
                f"threshold_seconds={self._delivery_stall_seconds:.3f}"
            )
            logger.critical(message)
            raise EventDeliveryStalledError(message)

    def sender(self) -> Sender[Event]:
        send, recv = channel[Event](error_override_config=_ERROR_CFG)
        if self._tg.is_running():
            self._tg.start_soon(self._ingest, SystemId(), recv)
        else:
            self._tg.queue(self._ingest, SystemId(), recv)
        return send

    def receiver(self) -> Receiver[IndexedEvent]:
        assert not self._tg.is_running()
        send, recv = channel[IndexedEvent](error_override_config=_ERROR_CFG)
        self.internal_outbound.append(send)
        return recv

    def shutdown(self) -> None:
        self._tg.cancel_tasks()

    async def _ingest(self, system_id: SystemId, recv: Receiver[Event]):
        idx = 0
        with recv as events:
            async for event in events:
                f_ev = LocalForwarderEvent(
                    origin_idx=idx,
                    origin=system_id,
                    session=self.session_id,
                    event=event,
                )
                idx += 1
                await self.external_outbound.send(f_ev)
                now = anyio.current_time()
                self.out_for_delivery[event.event_id] = _PendingDelivery(
                    first_sent_at=now,
                    last_sent_at=now,
                    event=f_ev,
                )
                self._log_outbound_pressure()

    async def _run_ext_in(self):
        buf = OrderedBuffer[Event]()
        with self.external_inbound as events:
            async for event in events:
                if event.session != self.session_id:
                    continue
                if event.origin != self.session_id.master_node_id:
                    continue

                buf.ingest(event.origin_idx, event.event)
                event_id = event.event.event_id
                if event_id in self.out_for_delivery:
                    self.out_for_delivery.pop(event_id)
                    logger.debug(
                        "Acknowledged local event from global stream "
                        f"event_id={event_id} origin_idx={event.origin_idx} "
                        f"remaining_out_for_delivery={len(self.out_for_delivery)}"
                    )

                drained = buf.drain_indexed()
                if drained:
                    self._nack_attempts = 0
                    if self._nack_cancel_scope:
                        self._nack_cancel_scope.cancel()

                if not drained and (
                    self._nack_cancel_scope is None
                    or self._nack_cancel_scope.cancel_called
                ):
                    logger.warning(
                        "Global event stream gap detected "
                        f"received_idx={event.origin_idx} "
                        f"next_expected_idx={buf.next_idx_to_release} "
                        f"event_type={type(event.event).__name__}"
                    )
                    # Request the next index.
                    self._start_nack_request(
                        buf.next_idx_to_release,
                        max_events=self._nack_replay_size(buf),
                    )
                    continue

                for idx, event in drained:
                    to_clear = set[int]()
                    for i, sender in enumerate(self.internal_outbound):
                        try:
                            await sender.send(IndexedEvent(idx=idx, event=event))
                        except (ClosedResourceError, BrokenResourceError):
                            to_clear.add(i)
                    for i in sorted(to_clear, reverse=True):
                        self.internal_outbound.pop(i)
                if drained and buf.store:
                    # A one-event replay can close the first hole while
                    # leaving later buffered events behind another gap.
                    # Schedule the next replay immediately instead of
                    # waiting for unrelated future global traffic.
                    self._start_nack_request(
                        buf.next_idx_to_release,
                        max_events=self._nack_replay_size(buf),
                    )

    def _nack_replay_size(self, buf: OrderedBuffer[Event]) -> int:
        if not buf.store:
            return 1
        buffered_gap = max(buf.store) - buf.next_idx_to_release + 1
        return max(1, min(buffered_gap, self._nack_max_events))

    async def _nack_request(self, since_idx: int, max_events: int) -> None:
        # We request all events after (and including) the missing index.
        # This function is started whenever we receive an event that is out of sequence.
        # It is cancelled as soon as we receiver an event that is in sequence.

        if since_idx < 0:
            logger.warning(f"Negative value encountered for nack request {since_idx=}")
            since_idx = 0

        with CancelScope() as scope:
            self._nack_cancel_scope = scope
            delay: float = self._nack_base_seconds * (2.0**self._nack_attempts)
            delay = min(self._nack_cap_seconds, delay)
            self._nack_attempts += 1
            try:
                await anyio.sleep(delay)
                logger.info(
                    "Requesting event log replay "
                    f"nack_attempt={self._nack_attempts} since_idx={since_idx} "
                    f"session={self.session_id} "
                    f"out_for_delivery={len(self.out_for_delivery)}"
                )
                await self.command_sender.send(
                    ForwarderCommand(
                        origin=self._system_id,
                        command=RequestEventLog(
                            since_idx=since_idx, max_events=max_events
                        ),
                    )
                )
            finally:
                if self._nack_cancel_scope is scope:
                    self._nack_cancel_scope = None

    def _start_nack_request(self, since_idx: int, *, max_events: int) -> None:
        if (
            self._nack_cancel_scope is not None
            and not self._nack_cancel_scope.cancel_called
        ):
            return
        self._tg.start_soon(self._nack_request, since_idx, max_events)

    def _log_outbound_pressure(self) -> None:
        size = len(self.out_for_delivery)
        if size < 10:
            self._last_outbound_warning_size = 0
            return
        if size >= self._last_outbound_warning_size + 10:
            self._last_outbound_warning_size = size
            logger.warning(
                "Local events awaiting master acknowledgement "
                f"out_for_delivery={size} session={self.session_id}"
            )
