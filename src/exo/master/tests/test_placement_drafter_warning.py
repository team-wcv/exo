"""Tests for the drafter-aware placement warning (item 10).

When a model card declares `drafter_model_ids`, the placement engine still
prefers single-node (via the existing smallest-cycle-first logic). When
single-node placement is impossible because no single node has enough RAM
for the requested quant, placement falls back to multi-node and emits a
clear warning so the operator knows speculative decoding has been silently
disabled and can re-place a smaller-quant variant.

The warning is gated on whether the asymmetric drafter codepath actually
selected a placement: when an asymmetric drafter rank IS attached to the
parent ``mx.distributed`` group (a separate node listed in
``drafter_eligible_nodes`` / ``drafter_eligible_friendly_names``),
speculative decoding is fully active across the multi-node target and the
warning must NOT fire. A premature unconditional warning would mislead
operators into thinking JACCL clusters can't run drafters at all.
"""

from collections.abc import Iterator

import pytest
from loguru import logger as loguru_logger

from exo.master.placement import place_instance
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
from exo.shared.types.profiling import NodeRdmaCtlStatus
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding


@pytest.fixture
def loguru_capture() -> Iterator[list[str]]:
    """Capture loguru WARNING+ messages into a list (caplog doesn't see loguru)."""
    captured: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: captured.append(str(message)), level="WARNING"
    )
    try:
        yield captured
    finally:
        loguru_logger.remove(sink_id)


def _drafter_aware_card(
    storage_bytes: int, eligible_nodes: list[NodeId] | None = None
) -> ModelCard:
    return ModelCard(
        model_id=ModelId("mlx-community/gemma-4-31b-it-8bit"),
        storage_size=Memory.from_bytes(storage_bytes),
        n_layers=60,
        hidden_size=5376,
        num_key_value_heads=16,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="gemma",
        base_model="Gemma 4 31B",
        drafter_model_ids=[
            ModelId("mlx-community/gemma-4-e2b-it-8bit"),
            ModelId("mlx-community/gemma-4-e4b-it-8bit"),
        ],
        drafter_eligible_nodes=eligible_nodes or [],
    )


def test_drafter_aware_card_placed_single_node_when_fits(
    loguru_capture: list[str],
) -> None:
    """When a single node has enough RAM, the model lands on that node and
    no warning is emitted -- speculative decoding is preserved."""
    big_node = NodeId()
    topology = Topology()
    topology.add_node(big_node)

    card = _drafter_aware_card(20_000_000_000)
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
        {big_node: create_node_memory(64_000_000_000)},
        {big_node: create_node_network()},
    )
    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert len(instance.shard_assignments.node_to_runner) == 1
    joined = "\n".join(loguru_capture).lower()
    assert "speculative decoding is single-device only" not in joined


def test_drafter_aware_card_warns_when_only_multi_node_fits(
    loguru_capture: list[str],
) -> None:
    """When no single node has enough RAM, placement falls back to multi-node
    and warns the operator that the drafter will be silently disabled."""
    node_a, node_b = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(2))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )

    # 20 GB target with hidden_size divisible by 2 nodes; only multi-node
    # fits (16 GB each). Use Tensor sharding because Gemma 4 doesn't allow
    # multi-node Pipeline.
    card = _drafter_aware_card(20_000_000_000)
    command = PlaceInstance(
        sharding=Sharding.Tensor,
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
            node_a: create_node_memory(16_000_000_000),
            node_b: create_node_memory(16_000_000_000),
        },
        {
            node_a: create_node_network(),
            node_b: create_node_network(),
        },
    )
    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert len(instance.shard_assignments.node_to_runner) == 2
    joined = "\n".join(loguru_capture).lower()
    assert "speculative decoding is single-device only" in joined
    assert "smaller quant" in joined


def test_drafter_aware_card_no_warning_when_asymmetric_drafter_placed(
    loguru_capture: list[str],
) -> None:
    """Multi-node target + asymmetric drafter rank => spec decoding active,
    no degradation warning fires.

    This is the regression guard for the bug where the placement engine
    emitted ``Speculative decoding is single-device only and will be
    disabled`` *before* the asymmetric drafter codepath ran, causing the
    warning to fire on every multi-node placement of a drafter-aware
    model -- even when an eligible drafter rank was about to be attached
    to the parent ``mx.distributed`` group.
    """
    target_a, target_b, drafter_node = NodeId(), NodeId(), NodeId()
    topology = Topology()
    for n in (target_a, target_b, drafter_node):
        topology.add_node(n)
    # Bidirectional sockets between every pair so MlxRing target + asym
    # drafter has a TCP path in both directions.
    for source, sink, ip in (
        (target_a, target_b, 30),
        (target_b, target_a, 31),
        (target_a, drafter_node, 32),
        (drafter_node, target_a, 33),
        (target_b, drafter_node, 34),
        (drafter_node, target_b, 35),
    ):
        topology.add_connection(
            Connection(source=source, sink=sink, edge=create_socket_connection(ip))
        )

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        # Force multi-node target so the misleading warning would have
        # fired pre-fix; after the fix the asym drafter rank gets
        # attached and the warning is suppressed.
        min_nodes=2,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_a: create_node_memory(16_000_000_000),
            target_b: create_node_memory(16_000_000_000),
            drafter_node: create_node_memory(32_000_000_000),
        },
        {
            target_a: create_node_network(),
            target_b: create_node_network(),
            drafter_node: create_node_network(),
        },
        node_rdma_ctl={
            target_a: NodeRdmaCtlStatus(enabled=False),
            target_b: NodeRdmaCtlStatus(enabled=False),
            drafter_node: NodeRdmaCtlStatus(enabled=False),
        },
    )
    assert len(placements) == 1
    instance = next(iter(placements.values()))
    # Target spans both nodes; the drafter rank is attached separately as
    # ``instance.drafter_placement`` (not part of ``shard_assignments``).
    assert len(instance.shard_assignments.node_to_runner) == 2
    assert instance.drafter_placement is not None, (
        "asymmetric drafter placement should have been chosen for this topology"
    )
    assert instance.drafter_placement.drafter_node_id == drafter_node
    joined = "\n".join(loguru_capture).lower()
    assert "speculative decoding is single-device only" not in joined, (
        f"misleading warning fired despite asymmetric drafter being placed: "
        f"{joined}"
    )
