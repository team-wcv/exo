# pyright: reportPrivateUsage=false

from unittest.mock import AsyncMock, Mock, patch

import pytest

from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.events import CustomModelCardAdded, Event, IndexedEvent
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.utils.channels import Receiver, channel
from exo.worker.main import Worker


def _custom_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("custom/persisted"),
        n_layers=1,
        storage_size=Memory.from_bytes(1),
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
        is_custom=True,
    )


def _worker() -> tuple[Worker, Receiver[Event]]:
    event_sender, emitted_events = channel[Event]()
    _, event_receiver = channel[IndexedEvent]()
    command_sender, _ = channel[ForwarderCommand]()
    download_command_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        node_id=NodeId("worker"),
        event_receiver=event_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_command_sender,
        api_port=52415,
        peer_download_port=52416,
    )
    return worker, emitted_events


@pytest.mark.anyio
async def test_persisted_custom_cards_are_published_on_startup() -> None:
    card = _custom_card()
    worker, emitted_events = _worker()

    with patch(
        "exo.worker.main.card_cache.list_all",
        new_callable=AsyncMock,
        return_value=[card],
    ):
        pending = await worker._publish_persisted_custom_cards()

    event = await emitted_events.receive()
    assert isinstance(event, CustomModelCardAdded)
    assert event.model_card == card
    assert pending == {card.model_id}


@pytest.mark.anyio
async def test_startup_cards_are_not_deleted_before_state_replays() -> None:
    card = _custom_card()
    worker, _ = _worker()
    pending = {card.model_id}

    with (
        patch("exo.worker.main.card_cache.get", new=Mock(return_value=card)),
        patch(
            "exo.worker.main.card_cache.list_all",
            new_callable=AsyncMock,
            return_value=[card],
        ),
        patch("exo.worker.main.card_cache.pop", new_callable=AsyncMock) as pop_card,
    ):
        await worker._reconcile_custom_cards_once(pending)

    pop_card.assert_not_awaited()
    assert pending == {card.model_id}


@pytest.mark.anyio
async def test_replayed_then_deleted_custom_card_is_removed_from_disk() -> None:
    card = _custom_card()
    worker, _ = _worker()
    worker.state = State(custom_model_cards={card.model_id: card})
    pending = {card.model_id}

    with (
        patch("exo.worker.main.card_cache.get", new=Mock(return_value=card)),
        patch(
            "exo.worker.main.card_cache.is_persisted",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "exo.worker.main.card_cache.list_all",
            new_callable=AsyncMock,
            return_value=[card],
        ),
        patch("exo.worker.main.card_cache.pop", new_callable=AsyncMock) as pop_card,
    ):
        await worker._reconcile_custom_cards_once(pending)
        worker.state = State(last_event_applied_idx=0)
        await worker._reconcile_custom_cards_once(pending)

    pop_card.assert_awaited_once_with(card.model_id)
    assert pending == set()


@pytest.mark.anyio
async def test_state_card_is_saved_when_cached_but_not_persisted() -> None:
    card = _custom_card()
    worker, _ = _worker()
    worker.state = State(custom_model_cards={card.model_id: card})

    with (
        patch("exo.worker.main.card_cache.get", new=Mock(return_value=card)),
        patch(
            "exo.worker.main.card_cache.is_persisted",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch("exo.worker.main.card_cache.save", new_callable=AsyncMock) as save_card,
        patch(
            "exo.worker.main.card_cache.list_all",
            new_callable=AsyncMock,
            return_value=[card],
        ),
    ):
        await worker._reconcile_custom_cards_once(set())

    save_card.assert_awaited_once_with(card)
