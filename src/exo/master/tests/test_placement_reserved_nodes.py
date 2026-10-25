"""Inference reservations must not consume or conceal usable placements."""

from exo.master.placement import place_instance
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_socket_connection,
)
from exo.master.tests.test_placement_drafter_asymmetric import (
    _drafter_aware_card,  # pyright: ignore[reportPrivateUsage]
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import NodeId
from exo.shared.types.events import DrafterPlacementDegraded
from exo.shared.types.memory import Memory
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding


def _connected(nodes: list[NodeId]) -> Topology:
    topology = Topology()
    for node in nodes:
        topology.add_node(node)
    for source in nodes:
        for index, sink in enumerate(nodes):
            if source != sink:
                topology.add_connection(
                    Connection(
                        source=source,
                        sink=sink,
                        edge=create_socket_connection(index + 2),
                    )
                )
    return topology


def test_reserved_singleton_does_not_hide_larger_usable_cycle() -> None:
    reserved, first, second = NodeId(), NodeId(), NodeId()
    nodes = [reserved, first, second]
    card = ModelCard(
        model_id=ModelId("test/reservation"),
        storage_size=Memory.from_bytes(40_000_000_000),
        n_layers=4,
        hidden_size=128,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    placements = place_instance(
        PlaceInstance(
            model_card=card,
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        _connected(nodes),
        {},
        {
            reserved: create_node_memory(80_000_000_000),
            first: create_node_memory(30_000_000_000),
            second: create_node_memory(30_000_000_000),
        },
        {node: create_node_network() for node in nodes},
        node_backends={
            reserved: [],
            first: [Backend.MlxMetal],
            second: [Backend.MlxMetal],
        },
    )
    instance = next(iter(placements.values()))
    assert {node for node, _, _ in instance.shard_assignments.shards} == {first, second}


def test_reserved_drafter_degrades_without_blocking_target() -> None:
    target, reserved = NodeId(), NodeId()
    nodes = [target, reserved]
    card = _drafter_aware_card(storage_bytes=20_000_000_000, eligible_nodes=[reserved])
    degraded: list[DrafterPlacementDegraded] = []
    placements = place_instance(
        PlaceInstance(
            model_card=card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        _connected(nodes),
        {},
        {node: create_node_memory(64_000_000_000) for node in nodes},
        {node: create_node_network() for node in nodes},
        node_backends={target: [Backend.MlxMetal], reserved: []},
        on_drafter_placement_degraded=degraded.append,
    )
    instance = next(iter(placements.values()))
    assert instance.drafter_placement is None
    assert {node for node, _, _ in instance.shard_assignments.shards} == {target}
    assert len(degraded) == 1
