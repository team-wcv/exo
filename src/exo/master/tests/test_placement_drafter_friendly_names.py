"""Tests for friendly-name resolution in drafter eligibility.

A model card can opt into asymmetric drafter placement either by listing
raw libp2p ``NodeId`` peer identifiers in ``drafter_eligible_nodes`` or
by listing human-readable hostnames in
``drafter_eligible_friendly_names``. The placement layer resolves friendly
names against the live ``state.node_identities`` map and merges the
result with the raw-NodeId list before asymmetric-drafter selection
runs.

Coverage:
- ``resolve_drafter_eligible_nodes`` returns raw NodeIds unchanged
  (legacy behaviour).
- ``resolve_drafter_eligible_nodes`` resolves friendly names against the
  identities map.
- The merge dedupes when the same node appears in both fields.
- An unmatched friendly name is silently dropped (no exception, no
  spurious degradation event).
- Resolution treats a missing identities map (``None``) as "skip
  friendly-name resolution" rather than raising.
- End-to-end through ``place_instance``: a friendly-name-only card
  produces a drafter placement on the matched node.
"""

from collections.abc import Mapping

from exo.master.placement import (
    place_instance,
    resolve_drafter_eligible_nodes,
)
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import NodeIdentity
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import InstanceMeta, MlxRingInstance
from exo.shared.types.worker.shards import Sharding


def _identity(friendly_name: str) -> NodeIdentity:
    return NodeIdentity(
        model_id="MacBookPro18,1",
        chip_id="Apple M5 Max",
        friendly_name=friendly_name,
        os_version="macOS 26.5",
        os_build_version="25F00",
    )


def _drafter_card(
    *,
    eligible_nodes: list[NodeId] | None = None,
    eligible_friendly_names: list[str] | None = None,
) -> ModelCard:
    return ModelCard(
        model_id=ModelId("mlx-community/gemma-4-31b-it-8bit"),
        storage_size=Memory.from_bytes(20_000_000_000),
        n_layers=60,
        hidden_size=5376,
        num_key_value_heads=16,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="gemma",
        base_model="Gemma 4 31B",
        drafter_model_ids=[ModelId("mlx-community/gemma-4-e2b-it-8bit")],
        drafter_eligible_nodes=eligible_nodes or [],
        drafter_eligible_friendly_names=eligible_friendly_names or [],
    )


def test_resolve_returns_raw_node_ids_unchanged() -> None:
    """Legacy callers populate only `drafter_eligible_nodes`; resolver passes through."""
    raw_node = NodeId()
    card = _drafter_card(eligible_nodes=[raw_node])

    result = resolve_drafter_eligible_nodes(card, node_identities=None)

    assert result == [raw_node]


def test_resolve_with_friendly_name_matches_identity() -> None:
    """Friendly-name-only card resolves against the identities map."""
    bmbp_node = NodeId()
    smbp_node = NodeId()
    identities: Mapping[NodeId, NodeIdentity] = {
        bmbp_node: _identity("wc-bmbp"),
        smbp_node: _identity("wc-smbp"),
    }
    card = _drafter_card(eligible_friendly_names=["wc-bmbp"])

    result = resolve_drafter_eligible_nodes(card, identities)

    assert result == [bmbp_node]


def test_resolve_dedupes_overlap_between_raw_and_friendly() -> None:
    """A node listed by both NodeId and friendly name appears only once."""
    bmbp_node = NodeId()
    identities = {bmbp_node: _identity("wc-bmbp")}
    card = _drafter_card(
        eligible_nodes=[bmbp_node],
        eligible_friendly_names=["wc-bmbp"],
    )

    result = resolve_drafter_eligible_nodes(card, identities)

    assert result == [bmbp_node]


def test_resolve_drops_unmatched_friendly_name() -> None:
    """A friendly name with no live match is silently dropped."""
    bmbp_node = NodeId()
    identities = {bmbp_node: _identity("wc-bmbp")}
    card = _drafter_card(eligible_friendly_names=["wc-doesnt-exist"])

    result = resolve_drafter_eligible_nodes(card, identities)

    assert result == []


def test_resolve_with_none_identities_skips_friendly_resolution() -> None:
    """Passing identities=None falls back to raw-NodeIds-only behaviour."""
    raw_node = NodeId()
    card = _drafter_card(
        eligible_nodes=[raw_node],
        eligible_friendly_names=["wc-bmbp"],
    )

    result = resolve_drafter_eligible_nodes(card, node_identities=None)

    assert result == [raw_node]


def test_resolve_preserves_first_appearance_order() -> None:
    """Resolved list keeps insertion order; raw entries come before friendly."""
    raw_node = NodeId()
    friendly_node = NodeId()
    identities = {friendly_node: _identity("wc-friendly")}
    card = _drafter_card(
        eligible_nodes=[raw_node],
        eligible_friendly_names=["wc-friendly"],
    )

    result = resolve_drafter_eligible_nodes(card, identities)

    assert result == [raw_node, friendly_node]


def test_place_instance_resolves_friendly_name_end_to_end() -> None:
    """Friendly-name-only card produces an asymmetric drafter placement.

    Topology: one target node + one drafter node connected over a socket;
    drafter card lists the drafter by friendly name only. Placement
    resolves the friendly name and creates a drafter rank on the matched
    node.
    """
    target_node = NodeId()
    drafter_node = NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(drafter_node)
    topology.add_connection(
        Connection(
            source=target_node,
            sink=drafter_node,
            edge=create_socket_connection(ip=1),
        )
    )
    topology.add_connection(
        Connection(
            source=drafter_node,
            sink=target_node,
            edge=create_socket_connection(ip=2),
        )
    )

    identities = {
        target_node: _identity("wc-smbp"),
        drafter_node: _identity("wc-bmbp"),
    }

    card = _drafter_card(eligible_friendly_names=["wc-bmbp"])
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_node: create_node_memory(64_000_000_000),
            drafter_node: create_node_memory(48_000_000_000),
        },
        {
            target_node: create_node_network(),
            drafter_node: create_node_network(),
        },
        node_identities=identities,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert instance.drafter_placement is not None
    assert instance.drafter_placement.drafter_node_id == drafter_node


def test_place_instance_without_identities_falls_back_to_raw_only() -> None:
    """Friendly-name-only card with identities=None => no asymmetric drafter.

    This is the explicit "skip friendly-name resolution" path; the
    placement is left without an asymmetric drafter (legacy single-rank
    flow) rather than raising or guessing.
    """
    target_node = NodeId()
    topology = Topology()
    topology.add_node(target_node)

    card = _drafter_card(eligible_friendly_names=["wc-bmbp"])
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {target_node: create_node_memory(64_000_000_000)},
        {target_node: create_node_network()},
        node_identities=None,
    )

    instance = next(iter(placements.values()))
    assert instance.drafter_placement is None
