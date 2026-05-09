"""Tests for the optional `drafter_model_ids` field on ModelCard.

The field declares a preference-ordered list of speculative-decoding draft
models that runners may load alongside the target. Coverage:
- ModelCard accepts and serialises the field.
- Cards with no drafters declared default to an empty list.
- Gemma 4 large-instruct cards declare both e2b and e4b drafters at matching
  quantisation, in fastest-first order.
"""

from pathlib import Path

import pytest
from anyio import Path as AsyncPath

from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelCard, ModelId, get_model_cards
from exo.shared.types.memory import Memory


@pytest.fixture(autouse=True)
def _isolate_custom_cards(  # pyright: ignore[reportUnusedFunction]
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Insulate these tests from operator-local custom card overrides.

    ``_custom_cards_dir`` resolves to ``$EXO_DATA_HOME/custom_model_cards``,
    which on dev workstations holds operator-edited cards (e.g. trimmed
    drafter lists for memory-constrained clusters). Those overrides are
    layered on top of the shipped TOML, so without isolation the assertions
    below describe whatever the operator last wrote, not the shipped data
    the gate is supposed to protect. Reset the in-memory cache too so the
    next test refreshes from the now-empty custom dir.
    """
    custom_dir = tmp_path / "custom_model_cards"
    custom_dir.mkdir()
    monkeypatch.setattr(model_cards, "_custom_cards_dir", AsyncPath(custom_dir))
    monkeypatch.setattr(model_cards, "_card_cache", {})


@pytest.mark.asyncio
async def test_drafter_model_ids_defaults_to_empty_list() -> None:
    cards = {card.model_id: card for card in await get_model_cards()}
    qwen_id = ModelId("mlx-community/Qwen3-30B-A3B-4bit")
    if qwen_id in cards:
        assert cards[qwen_id].drafter_model_ids == []


def _gemma4_31b_expectations() -> dict[str, list[str]]:
    return {
        "mlx-community/gemma-4-31b-it-4bit": [
            "mlx-community/gemma-4-e2b-it-4bit",
            "mlx-community/gemma-4-e4b-it-4bit",
        ],
        "mlx-community/gemma-4-31b-it-6bit": [
            "mlx-community/gemma-4-e2b-it-6bit",
            "mlx-community/gemma-4-e4b-it-6bit",
        ],
        "mlx-community/gemma-4-31b-it-8bit": [
            "mlx-community/gemma-4-e2b-it-8bit",
            "mlx-community/gemma-4-e4b-it-8bit",
        ],
        "mlx-community/gemma-4-31b-it-bf16": [
            "mlx-community/gemma-4-e2b-it-bf16",
            "mlx-community/gemma-4-e4b-it-bf16",
        ],
    }


def _gemma4_26b_expectations() -> dict[str, list[str]]:
    return {
        "mlx-community/gemma-4-26b-a4b-it-4bit": [
            "mlx-community/gemma-4-e2b-it-4bit",
            "mlx-community/gemma-4-e4b-it-4bit",
        ],
        "mlx-community/gemma-4-26b-a4b-it-6bit": [
            "mlx-community/gemma-4-e2b-it-6bit",
            "mlx-community/gemma-4-e4b-it-6bit",
        ],
        "mlx-community/gemma-4-26b-a4b-it-8bit": [
            "mlx-community/gemma-4-e2b-it-8bit",
            "mlx-community/gemma-4-e4b-it-8bit",
        ],
        "mlx-community/gemma-4-26b-a4b-it-bf16": [
            "mlx-community/gemma-4-e2b-it-bf16",
            "mlx-community/gemma-4-e4b-it-bf16",
        ],
    }


@pytest.mark.asyncio
async def test_gemma4_31b_cards_declare_e2b_then_e4b_drafters() -> None:
    cards = {card.model_id: card for card in await get_model_cards()}
    for target_str, expected_drafters in _gemma4_31b_expectations().items():
        target_id = ModelId(target_str)
        assert target_id in cards, f"{target_id} card missing"
        card = cards[target_id]
        assert card.drafter_model_ids == [ModelId(d) for d in expected_drafters], (
            f"{target_id} drafter mismatch: got {card.drafter_model_ids!r}"
        )


@pytest.mark.asyncio
async def test_gemma4_26b_cards_declare_e2b_then_e4b_drafters() -> None:
    cards = {card.model_id: card for card in await get_model_cards()}
    for target_str, expected_drafters in _gemma4_26b_expectations().items():
        target_id = ModelId(target_str)
        assert target_id in cards, f"{target_id} card missing"
        card = cards[target_id]
        assert card.drafter_model_ids == [ModelId(d) for d in expected_drafters], (
            f"{target_id} drafter mismatch: got {card.drafter_model_ids!r}"
        )


def test_model_card_explicit_drafters_round_trip() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/test-target"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
        drafter_model_ids=[
            ModelId("mlx-community/test-drafter-fast"),
            ModelId("mlx-community/test-drafter-accurate"),
        ],
    )
    assert card.drafter_model_ids == [
        ModelId("mlx-community/test-drafter-fast"),
        ModelId("mlx-community/test-drafter-accurate"),
    ]
    dump = card.model_dump(exclude_none=True)
    assert dump["drafter_model_ids"] == [
        "mlx-community/test-drafter-fast",
        "mlx-community/test-drafter-accurate",
    ]
