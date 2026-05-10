"""Tests for the optional `drafter_model_ids` field on ModelCard.

The field declares a preference-ordered list of speculative-decoding draft
models that runners may load alongside the target. Coverage:
- ModelCard accepts and serialises the field.
- Cards with no drafters declared default to an empty list.
- Gemma 4 large-instruct cards declare both e2b and e4b drafters at matching
  quantisation, in fastest-first order.

Also covers the asymmetric placement opt-in field
``drafter_eligible_nodes``: empty by default (legacy in-process drafter),
populated to designate per-deployment hosts for drafter-only ranks. The
field round-trips through Pydantic serialisation.
"""

from pathlib import Path

import pytest
from anyio import Path as AsyncPath

from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelCard, ModelId, get_model_cards
from exo.shared.types.common import NodeId
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


def test_drafter_eligible_nodes_defaults_to_empty() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/test-target-2"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
    )
    assert card.drafter_eligible_nodes == []
    dump = card.model_dump(exclude_none=True)
    assert dump["drafter_eligible_nodes"] == []


def test_drafter_eligible_nodes_round_trip() -> None:
    eligible = [NodeId(), NodeId()]
    card = ModelCard(
        model_id=ModelId("mlx-community/test-target-3"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
        drafter_model_ids=[ModelId("mlx-community/test-drafter")],
        drafter_eligible_nodes=eligible,
    )
    assert card.drafter_eligible_nodes == eligible
    dump = card.model_dump(exclude_none=True)
    assert dump["drafter_eligible_nodes"] == eligible
    rehydrated = ModelCard.model_validate(dump)
    assert rehydrated.drafter_eligible_nodes == eligible


def test_coupled_drafter_defaults_to_none() -> None:
    """Cards that don't declare a coupled drafter retain legacy behaviour.

    Phase-1 invariant: the field is purely additive. Existing cards that omit
    ``coupled_drafter`` must validate and serialise as if the field weren't
    there (``model_dump(exclude_none=True)`` drops the ``None`` so the TOML
    on disk stays untouched for the steady-state of cards that haven't been
    updated).
    """
    card = ModelCard(
        model_id=ModelId("mlx-community/test-target-no-coupled"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
    )
    assert card.coupled_drafter is None
    dump = card.model_dump(exclude_none=True)
    assert "coupled_drafter" not in dump


def test_coupled_drafter_round_trip() -> None:
    """``coupled_drafter`` accepts a ModelId and round-trips through dump/validate.

    Drafter-kind resolution happens at *load* time (Phase 2) via
    ``mlx_vlm.speculative.drafters.resolve_drafter_kind`` reading the
    drafter's HF ``config.json``; the card stores only the model id so it
    stays decoupled from the mlx-vlm runtime API surface.
    """
    coupled = ModelId("mlx-community/gemma-4-E2B-it-assistant-bf16")
    card = ModelCard(
        model_id=ModelId("mlx-community/test-target-coupled"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
        coupled_drafter=coupled,
    )
    assert card.coupled_drafter == coupled
    dump = card.model_dump(exclude_none=True)
    assert dump["coupled_drafter"] == coupled
    rehydrated = ModelCard.model_validate(dump)
    assert rehydrated.coupled_drafter == coupled


def test_coupled_drafter_composes_with_standard_drafter_list() -> None:
    """A card may declare both a coupled drafter AND a standard sibling list.

    The two fields are not mutually exclusive: placement chooses between them
    based on topology (asymmetric placement → standard list; single-node →
    coupled). The card schema must accept both side-by-side without
    validation error so a single Gemma 4 31B card can serve every deployment
    shape from "one Mac" to "asymmetric pipeline across a Thunderbolt RDMA
    cluster."
    """
    standard_list = [
        ModelId("mlx-community/gemma-4-e2b-it-4bit"),
        ModelId("mlx-community/gemma-4-e4b-it-4bit"),
    ]
    coupled = ModelId("mlx-community/gemma-4-E2B-it-assistant-bf16")
    card = ModelCard(
        model_id=ModelId("mlx-community/test-target-hybrid"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
        drafter_model_ids=standard_list,
        coupled_drafter=coupled,
    )
    assert card.drafter_model_ids == standard_list
    assert card.coupled_drafter == coupled
    dump = card.model_dump(exclude_none=True)
    assert dump["drafter_model_ids"] == standard_list
    assert dump["coupled_drafter"] == coupled
    rehydrated = ModelCard.model_validate(dump)
    assert rehydrated.drafter_model_ids == standard_list
    assert rehydrated.coupled_drafter == coupled


@pytest.mark.asyncio
async def test_shipped_gemma4_cards_omit_coupled_drafter_in_phase1() -> None:
    """Gate against accidentally landing card updates in the Phase-1 PR.

    Updating shipped Gemma 4 cards to declare ``coupled_drafter`` is the job
    of Phase 3 (cards + placement + smoke), once the loader (Phase 2) can
    actually consume the field. Landing the card update earlier would cause
    the loader to silently ignore the field on every Phase-1 deployment,
    masking real bugs in the Phase-2 dispatch logic when it eventually ships.
    This test removes itself in Phase 3.
    """
    cards = {card.model_id: card for card in await get_model_cards()}
    for target_str in {
        *_gemma4_31b_expectations(),
        *_gemma4_26b_expectations(),
    }:
        target_id = ModelId(target_str)
        if target_id not in cards:
            continue
        assert cards[target_id].coupled_drafter is None, (
            f"{target_id} declares coupled_drafter in shipped TOML; that "
            "should land in the Phase 3 PR after the loader can consume it."
        )
