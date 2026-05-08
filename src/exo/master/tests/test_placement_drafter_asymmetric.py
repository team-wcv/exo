"""Tests for asymmetric drafter placement (Layer B).

When a model card declares ``drafter_eligible_nodes`` AND the cluster
has at least one such node alive, reachable from every target rank, and
with sufficient memory, placement appends a *drafter rank* to the
parent ``mx.distributed`` group on a separate node. Target ranks split
off into a target subgroup at runtime; the parent group is reserved for
``RemoteTransport`` send/recv between target rank 0 and the drafter
rank.

Coverage:
- Asymmetric placement is constructed when an eligible node is reachable
  with both backends (``MlxRing`` over socket, ``MlxJaccl`` over RDMA).
- Placement degrades loudly when no eligible node is alive, when every
  eligible node is already a target rank, or when the only eligible
  candidate has no reachable transport. The user's request still
  completes (placement returns *something*), and a
  ``DrafterPlacementDegraded`` event is emitted with the reason.
- Empty ``drafter_eligible_nodes`` preserves legacy behaviour.
- The drafter rank is always the LAST rank in the parent group.
"""

from collections.abc import Iterator

import pytest
from loguru import logger as loguru_logger

from exo.master.placement import place_instance
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_rdma_connection,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    DrafterPlacementDegradationReason,
    DrafterPlacementDegraded,
)
from exo.shared.types.memory import Memory
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import (
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.shards import Sharding


@pytest.fixture
def loguru_capture() -> Iterator[list[str]]:
    captured: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: captured.append(str(message)), level="ERROR"
    )
    try:
        yield captured
    finally:
        loguru_logger.remove(sink_id)


def _drafter_aware_card(
    *,
    storage_bytes: int,
    eligible_nodes: list[NodeId],
    family: str = "gemma",
    base_model: str = "Gemma 4 31B",
    model_id: str = "mlx-community/gemma-4-31b-it-8bit",
) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_bytes(storage_bytes),
        n_layers=60,
        hidden_size=5376,
        num_key_value_heads=16,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family=family,
        base_model=base_model,
        drafter_model_ids=[
            ModelId("mlx-community/gemma-4-e2b-it-8bit"),
            ModelId("mlx-community/gemma-4-e4b-it-8bit"),
        ],
        drafter_eligible_nodes=eligible_nodes,
    )


def _bidi_socket(topology: Topology, a: NodeId, b: NodeId, ip: int) -> None:
    topology.add_connection(
        Connection(source=a, sink=b, edge=create_socket_connection(ip))
    )
    topology.add_connection(
        Connection(source=b, sink=a, edge=create_socket_connection(ip + 1))
    )


def _bidi_rdma(topology: Topology, a: NodeId, b: NodeId, iface: int) -> None:
    topology.add_connection(
        Connection(source=a, sink=b, edge=create_rdma_connection(iface))
    )
    topology.add_connection(
        Connection(source=b, sink=a, edge=create_rdma_connection(iface + 1))
    )


def test_asymmetric_single_node_target_auto_upgrades_to_jaccl() -> None:
    """Single-node target + RDMA-reachable drafter => asymmetric jaccl.

    Single-rank target requires Pipeline sharding, but the parent group
    spans 2 ranks (1 target + 1 drafter) and ring backend lacks
    ``Group.split`` / ``send/recv``. Placement therefore auto-upgrades
    ``MlxRing`` -> ``MlxJaccl`` whenever asymmetric drafter placement
    will succeed, so the parent group can use jaccl for the drafter
    transport.
    """
    target_node, drafter_node = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(drafter_node)
    _bidi_socket(topology, target_node, drafter_node, ip=2)
    _bidi_rdma(topology, target_node, drafter_node, iface=4)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_node: create_node_memory(64_000_000_000),
            drafter_node: create_node_memory(32_000_000_000),
        },
        {
            target_node: create_node_network(),
            drafter_node: create_node_network(),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    assert len(placements) == 1
    assert not degradations
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance)
    assert instance.drafter_placement is not None
    placement = instance.drafter_placement
    assert placement.drafter_node_id == drafter_node
    assert placement.drafter_model_id == ModelId("mlx-community/gemma-4-e2b-it-8bit")
    assert placement.drafter_rank == 1  # target=1 rank, drafter is last (rank 1)
    # v3+ wire: drafter does not join mx.distributed -> parent_group_size
    # is the target-only rank count.
    assert instance.parent_group_size == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1


def test_asymmetric_ring_socket_only_places_drafter_over_socket() -> None:
    """Single-node ring target + socket-only drafter places drafter over TCP.

    v3+ wire decoupled the drafter from ``mx.distributed`` -- the wire
    runs over a plain TCP socket. RDMA is therefore no longer required
    for asymmetric placement; a socket-only path between target rank 0
    and the drafter node is sufficient. The instance still upgrades
    ``MlxRing -> MlxJaccl`` because the target ranks (1 here) are
    fine to leave on jaccl, but the drafter wire itself runs over TCP
    regardless of the target backend.
    """
    target_node, drafter_node = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(drafter_node)
    _bidi_socket(topology, target_node, drafter_node, ip=2)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_node: create_node_memory(64_000_000_000),
            drafter_node: create_node_memory(32_000_000_000),
        },
        {
            target_node: create_node_network(),
            drafter_node: create_node_network(),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert instance.drafter_placement is not None
    assert instance.drafter_placement.drafter_node_id == drafter_node
    # Target stays single-rank; drafter rides TCP regardless.
    assert instance.parent_group_size == 1
    assert not degradations


def test_asymmetric_jaccl_places_drafter_with_rdma_reachability() -> None:
    """Two-node target (RDMA cycle) + RDMA-reachable drafter => asymmetric jaccl.

    Single-node target gets downgraded MlxJaccl -> MlxRing by the legacy
    ``len(selected_cycle) == 1 -> InstanceMeta.MlxRing`` rewrite, so to
    exercise asymmetric jaccl we need the target to span 2 RDMA-connected
    nodes + a 3rd drafter node with RDMA edges to both.
    """
    target_a, target_b, drafter_node = NodeId(), NodeId(), NodeId()
    topology = Topology()
    for n in (target_a, target_b, drafter_node):
        topology.add_node(n)
    # Target cycle has bidirectional RDMA between target_a and target_b
    _bidi_rdma(topology, target_a, target_b, iface=10)
    _bidi_socket(topology, target_a, target_b, ip=12)
    # Drafter has bidirectional RDMA + socket to both target ranks.
    _bidi_rdma(topology, target_a, drafter_node, iface=20)
    _bidi_rdma(topology, target_b, drafter_node, iface=22)
    _bidi_socket(topology, target_a, drafter_node, ip=14)
    _bidi_socket(topology, target_b, drafter_node, ip=16)

    # Use a Qwen-family card so the test isn't subject to Gemma 4's
    # "no multi-node Pipeline" restriction. Tensor sharding works across
    # 2 RDMA-connected nodes when hidden_size is divisible by world_size.
    card = _drafter_aware_card(
        storage_bytes=40_000_000_000,
        eligible_nodes=[drafter_node],
        family="qwen",
        base_model="Qwen3 30B",
        model_id="mlx-community/Qwen3-30B-A3B-4bit",
    )
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=card,
        # min_nodes=2 forces multi-node target so the placement layer
        # keeps MlxJaccl instead of rewriting to MlxRing.
        min_nodes=2,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_a: create_node_memory(32_000_000_000),
            target_b: create_node_memory(32_000_000_000),
            drafter_node: create_node_memory(32_000_000_000),
        },
        {
            target_a: create_node_network(),
            target_b: create_node_network(),
            drafter_node: create_node_network(),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    assert len(placements) == 1
    assert not degradations, [(e.reason, e.detail) for e in degradations]
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance)
    assert instance.drafter_placement is not None
    placement = instance.drafter_placement
    assert placement.drafter_node_id == drafter_node
    assert placement.drafter_rank == 2  # logical telemetry index past target ranks
    # v3+ wire: drafter is on a TCP socket, not in mx.distributed.
    # parent_group_size and jaccl_devices cover only the 2 target ranks.
    assert instance.parent_group_size == 2
    assert len(instance.jaccl_devices) == 2
    assert len(instance.jaccl_devices[0]) == 2
    # Drafter node does not coordinate the target's mx.distributed group.
    assert drafter_node not in instance.jaccl_coordinators


def test_asymmetric_jaccl_socket_only_drafter_succeeds(
    loguru_capture: list[str],
) -> None:
    """Two-node jaccl target + socket-only drafter places successfully.

    v3+ wire: drafter IPC runs over a plain TCP socket independent of
    the target's ``mx.distributed`` group. So a socket-only path from
    target rank 0 to the drafter node is sufficient even when the
    target ranks themselves are coordinating over jaccl/RDMA. No
    degradation event should fire.
    """
    target_a, target_b, drafter_node = NodeId(), NodeId(), NodeId()
    topology = Topology()
    for n in (target_a, target_b, drafter_node):
        topology.add_node(n)
    # Target cycle has bidirectional RDMA; drafter only has socket edges.
    _bidi_rdma(topology, target_a, target_b, iface=30)
    _bidi_socket(topology, target_a, target_b, ip=32)
    _bidi_socket(topology, target_a, drafter_node, ip=34)
    _bidi_socket(topology, target_b, drafter_node, ip=36)

    card = _drafter_aware_card(
        storage_bytes=40_000_000_000,
        eligible_nodes=[drafter_node],
        family="qwen",
        base_model="Qwen3 30B",
        model_id="mlx-community/Qwen3-30B-A3B-4bit",
    )
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=card,
        min_nodes=2,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_a: create_node_memory(32_000_000_000),
            target_b: create_node_memory(32_000_000_000),
            drafter_node: create_node_memory(32_000_000_000),
        },
        {
            target_a: create_node_network(),
            target_b: create_node_network(),
            drafter_node: create_node_network(),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance)
    assert instance.drafter_placement is not None
    assert instance.drafter_placement.drafter_node_id == drafter_node
    # 2 target ranks + drafter on socket; mx.distributed is target-only.
    assert instance.parent_group_size == 2
    assert not degradations
    # No degradation log line either.
    joined = "\n".join(loguru_capture)
    assert "Drafter placement degraded" not in joined


def test_asymmetric_degrades_when_eligible_node_missing_from_topology(
    loguru_capture: list[str],
) -> None:
    """Eligible node id refers to a node not present in topology."""
    target_node = NodeId()
    missing_drafter_node = NodeId()  # Never added to topology.
    topology = Topology()
    topology.add_node(target_node)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[missing_drafter_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {target_node: create_node_memory(64_000_000_000)},
        {target_node: create_node_network()},
        on_drafter_placement_degraded=degradations.append,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert instance.drafter_placement is None
    assert len(degradations) == 1
    assert (
        degradations[0].reason
        == DrafterPlacementDegradationReason.NoEligibleNodeAvailable
    )
    assert degradations[0].fallback == "single_device_drafter"
    joined = "\n".join(loguru_capture).lower()
    assert "drafter placement degraded" in joined


def test_asymmetric_degrades_when_eligible_node_in_target_cycle(
    loguru_capture: list[str],
) -> None:
    """Listing the target node itself as eligible is a misconfig => degrade."""
    target_node = NodeId()
    topology = Topology()
    topology.add_node(target_node)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[target_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {target_node: create_node_memory(64_000_000_000)},
        {target_node: create_node_network()},
        on_drafter_placement_degraded=degradations.append,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert instance.drafter_placement is None
    assert len(degradations) == 1
    assert (
        degradations[0].reason
        == DrafterPlacementDegradationReason.AllEligibleNodesInTargetCycle
    )
    del loguru_capture  # captured but content irrelevant beyond emission


def test_asymmetric_degrades_when_drafter_node_lacks_memory() -> None:
    """Drafter node reachable but below memory floor (~6GB) => degrade.

    RDMA-reachable so jaccl auto-upgrade is viable, but memory check
    rejects the candidate. Single-node target therefore reverts to
    symmetric MlxRing without drafter.
    """
    target_node, drafter_node = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(drafter_node)
    _bidi_socket(topology, target_node, drafter_node, ip=8)
    _bidi_rdma(topology, target_node, drafter_node, iface=40)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_node: create_node_memory(64_000_000_000),
            drafter_node: create_node_memory(2_000_000_000),  # 2GB is below floor
        },
        {
            target_node: create_node_network(),
            drafter_node: create_node_network(),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert instance.drafter_placement is None
    assert len(degradations) == 1
    assert (
        degradations[0].reason
        == DrafterPlacementDegradationReason.InsufficientDrafterMemory
    )


def test_empty_drafter_eligible_nodes_preserves_legacy_behaviour() -> None:
    """No eligible list => no asymmetric attempt, no degradation events."""
    target_node = NodeId()
    topology = Topology()
    topology.add_node(target_node)

    card = ModelCard(
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
        drafter_eligible_nodes=[],
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {target_node: create_node_memory(64_000_000_000)},
        {target_node: create_node_network()},
        on_drafter_placement_degraded=degradations.append,
    )

    instance = next(iter(placements.values()))
    assert instance.drafter_placement is None
    assert not degradations  # no asymmetric attempt was made


def test_asymmetric_with_multiple_eligible_nodes_picks_first_reachable() -> None:
    """When multiple eligible nodes are listed, placement picks the first
    reachable (in card order). Earlier candidates that fail reachability
    are skipped silently (the search is best-effort, not first-fail).

    Single-node target auto-upgrades to jaccl, so the reachable drafter
    needs an RDMA edge (not just a socket edge); the unreachable drafter
    has no edges at all.
    """
    target_node = NodeId()
    unreachable_drafter = NodeId()
    reachable_drafter = NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(unreachable_drafter)
    topology.add_node(reachable_drafter)
    # Only reachable_drafter has socket + RDMA edges to target.
    _bidi_socket(topology, target_node, reachable_drafter, ip=20)
    _bidi_rdma(topology, target_node, reachable_drafter, iface=50)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000,
        eligible_nodes=[unreachable_drafter, reachable_drafter],
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_node: create_node_memory(64_000_000_000),
            unreachable_drafter: create_node_memory(32_000_000_000),
            reachable_drafter: create_node_memory(32_000_000_000),
        },
        {
            target_node: create_node_network(),
            unreachable_drafter: create_node_network(),
            reachable_drafter: create_node_network(),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    instance = next(iter(placements.values()))
    assert instance.drafter_placement is not None
    assert instance.drafter_placement.drafter_node_id == reachable_drafter
    assert not degradations  # successful placement, no degradation


def test_asymmetric_round_trip_serialization() -> None:
    """An asymmetric instance round-trips through pydantic serialisation.

    Single-node target auto-upgrades to ``MlxJaccl`` for the asymmetric
    parent group (ring lacks split + send/recv), so the round-trip is
    exercised on ``MlxJacclInstance`` here. RDMA edges to the drafter
    node make the auto-upgrade viable.
    """
    target_node, drafter_node = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(drafter_node)
    _bidi_socket(topology, target_node, drafter_node, ip=30)
    _bidi_rdma(topology, target_node, drafter_node, iface=60)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
    )
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
            drafter_node: create_node_memory(32_000_000_000),
        },
        {
            target_node: create_node_network(),
            drafter_node: create_node_network(),
        },
    )
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance)
    assert instance.drafter_placement is not None

    dumped = instance.model_dump()
    rehydrated = MlxJacclInstance.model_validate(dumped)
    assert rehydrated == instance
    assert rehydrated.drafter_placement is not None
    assert (
        rehydrated.drafter_placement.drafter_node_id
        == instance.drafter_placement.drafter_node_id
    )
