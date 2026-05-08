from collections.abc import Callable, Mapping
from copy import deepcopy
from os import environ
from typing import Sequence

from loguru import logger

from exo.master.placement_utils import (
    Cycle,
    filter_cycles_by_memory,
    get_mlx_jaccl_coordinators,
    get_mlx_jaccl_devices_matrix,
    get_mlx_ring_hosts_by_node,
    get_shard_assignments,
    get_smallest_cycles,
)
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.topology import Topology
from exo.shared.types.commands import (
    CancelDownload,
    CreateInstance,
    DeleteInstance,
    DownloadCommand,
    PlaceInstance,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    DrafterPlacementDegradationReason,
    DrafterPlacementDegraded,
    Event,
    InstanceCreated,
    InstanceDeleted,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, NodeNetworkInfo
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.topology import RDMAConnection, SocketConnection
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.instances import (
    DrafterPlacement,
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import Sharding
from exo.utils.ports import random_ephemeral_port

ASYMMETRIC_TENSOR_AUTO_UPGRADE_ENV = "EXO_ENABLE_ASYMMETRIC_TP_AUTO_UPGRADE"


def _supports_asymmetric_tensor_parallel(model_card: ModelCard) -> bool:
    model_id = model_card.model_id.lower()
    base_model = model_card.base_model.lower()
    return (
        base_model.startswith("qwen3.5")
        or "qwen3.5" in model_id
        or "qwen-3.5" in model_id
    )


def _asymmetric_tensor_auto_upgrade_enabled() -> bool:
    return environ.get(ASYMMETRIC_TENSOR_AUTO_UPGRADE_ENV, "").lower() in {
        "1",
        "true",
        "yes",
    }


def add_instance_to_placements(
    command: CreateInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
) -> Mapping[InstanceId, Instance]:
    # TODO: validate against topology

    return {**current_instances, command.instance.instance_id: command.instance}


def _get_node_download_fraction(
    node_id: NodeId,
    model_id: ModelId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Return the download fraction (0.0-1.0) for a model on a given node."""
    for progress in download_status.get(node_id, []):
        if progress.shard_metadata.model_card.model_id != model_id:
            continue
        match progress:
            case DownloadCompleted():
                return 1.0
            case DownloadOngoing():
                total = progress.download_progress.total.in_bytes
                return (
                    progress.download_progress.downloaded.in_bytes / total
                    if total > 0
                    else 0.0
                )
            case DownloadPending():
                total = progress.total.in_bytes
                return progress.downloaded.in_bytes / total if total > 0 else 0.0
            case DownloadFailed():
                return 0.0
    return 0.0


def _cycle_download_score(
    cycle: Cycle,
    model_id: ModelId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Sum of download fractions across all nodes in a cycle."""
    return sum(
        _get_node_download_fraction(node_id, model_id, download_status)
        for node_id in cycle
    )


def place_instance(
    command: PlaceInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    required_nodes: set[NodeId] | None = None,
    allowed_nodes: set[NodeId] | None = None,
    allow_single_node_total_memory: bool = False,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]] | None = None,
    on_drafter_placement_degraded: (
        Callable[[DrafterPlacementDegraded], None] | None
    ) = None,
) -> dict[InstanceId, Instance]:
    sharding = command.sharding
    instance_meta = command.instance_meta
    cycles = topology.get_cycles()
    candidate_cycles = list(filter(lambda it: len(it) >= command.min_nodes, cycles))

    # Filter to cycles containing all required nodes (subset matching)
    if required_nodes:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if required_nodes.issubset(cycle.node_ids)
        ]
    if allowed_nodes is not None:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if set(cycle.node_ids).issubset(allowed_nodes)
        ]

    # Reserve drafter-eligible nodes for the drafter rank when possible, so
    # the placement layer doesn't accidentally pull a drafter-eligible node
    # into the target cycle and then degrade because no eligible host
    # remains. If filtering them out leaves zero cycles, fall back to the
    # unfiltered set -- the user gets target placement at the cost of the
    # asymmetric drafter, and `_select_drafter_placement` emits a
    # ``AllEligibleNodesInTargetCycle`` degradation downstream.
    eligible_drafter_set = set(command.model_card.drafter_eligible_nodes)
    if eligible_drafter_set and command.model_card.drafter_model_ids:
        cycles_excluding_drafters = [
            cycle
            for cycle in candidate_cycles
            if not (set(cycle.node_ids) & eligible_drafter_set)
        ]
        if cycles_excluding_drafters:
            candidate_cycles = cycles_excluding_drafters
    cycles_with_sufficient_memory = filter_cycles_by_memory(
        candidate_cycles,
        node_memory,
        command.model_card.storage_size,
        allow_single_node_total_memory=allow_single_node_total_memory,
    )
    if len(cycles_with_sufficient_memory) == 0:
        raise ValueError("No cycles found with sufficient memory")

    if (
        sharding == Sharding.AsymmetricTensor
        and not _supports_asymmetric_tensor_parallel(command.model_card)
    ):
        raise ValueError(
            f"Asymmetric tensor parallelism is not yet supported for "
            f"model '{command.model_card.model_id}'. Supported: Qwen3.5."
        )

    if sharding in (Sharding.Tensor, Sharding.AsymmetricTensor):
        if not command.model_card.supports_tensor:
            raise ValueError(
                f"Requested Tensor sharding but this model does not support tensor parallelism: {command.model_card.model_id}"
            )
        if sharding == Sharding.Tensor:
            # TODO: the condition here for tensor parallel is not correct, but it works good enough for now.
            # DeepSeek V4 is MQA (num_key_value_heads=1) but its sharding strategy
            # head-parallelises wq_b/wo_a and shards MoE experts instead of splitting
            # KV heads, so the kv-head divisibility check doesn't apply.
            is_deepseek_v4 = command.model_card.base_model.startswith("DeepSeek V4")
            kv_heads = command.model_card.num_key_value_heads
            cycles_with_sufficient_memory = [
                cycle
                for cycle in cycles_with_sufficient_memory
                if command.model_card.hidden_size % len(cycle) == 0
                and (is_deepseek_v4 or kv_heads is None or kv_heads % len(cycle) == 0)
            ]
            if not cycles_with_sufficient_memory:
                raise ValueError(
                    f"No tensor sharding found for model with "
                    f"hidden_size={command.model_card.hidden_size}"
                    f"{f', num_key_value_heads={kv_heads}' if kv_heads is not None else ''}"
                    f" across candidate cycles"
                )

            # Auto-upgrade to AsymmetricTensor when equal TP won't fit on
            # the smallest node but asymmetric split would.
            if (
                _asymmetric_tensor_auto_upgrade_enabled()
                and _supports_asymmetric_tensor_parallel(command.model_card)
            ):
                for cycle in cycles_with_sufficient_memory:
                    if len(cycle) != 2:
                        continue
                    equal_share = command.model_card.storage_size.in_bytes / len(cycle)
                    min_node_mem = min(
                        node_memory[nid].ram_available.in_bytes for nid in cycle
                    )
                    if equal_share > min_node_mem * 0.9:
                        # Equal split too tight; try asymmetric.
                        total_mem = sum(
                            node_memory[nid].ram_available.in_bytes for nid in cycle
                        )
                        if command.model_card.storage_size.in_bytes < total_mem * 0.85:
                            logger.info(
                                "Equal tensor split won't fit on smallest node "
                                f"({min_node_mem / 1e9:.0f}GB available, "
                                f"needs {equal_share / 1e9:.0f}GB). "
                                "Auto-upgrading to AsymmetricTensor."
                            )
                            sharding = Sharding.AsymmetricTensor
                        break
    if sharding == Sharding.AsymmetricTensor:
        cycles_with_sufficient_memory = [
            cycle for cycle in cycles_with_sufficient_memory if len(cycle) == 2
        ]
        cycles_with_sufficient_memory = [
            cycle
            for cycle in cycles_with_sufficient_memory
            if _asymmetric_tensor_rank_zero_is_socket_reachable(
                cycle=cycle,
                node_memory=node_memory,
                topology=topology,
            )
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                "Asymmetric tensor parallelism currently requires exactly 2 nodes "
                "with the largest-memory rank-0 node socket-reachable"
            )

    if sharding == Sharding.Pipeline and command.model_card.model_id == ModelId(
        "mlx-community/DeepSeek-V3.1-8bit"
    ):
        raise ValueError(
            "Pipeline parallelism is not supported for DeepSeek V3.1 (8-bit)"
        )
    if sharding == Sharding.Pipeline and command.model_card.base_model.startswith(
        "Gemma 4"
    ):
        cycles_with_sufficient_memory = [
            cycle for cycle in cycles_with_sufficient_memory if len(cycle) == 1
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                "Pipeline parallelism is not supported for Gemma 4; use tensor parallelism instead."
            )

    smallest_cycles = get_smallest_cycles(cycles_with_sufficient_memory)

    smallest_rdma_cycles = [
        cycle for cycle in smallest_cycles if topology.is_rdma_cycle(cycle)
    ]

    if instance_meta == InstanceMeta.MlxJaccl:
        if not smallest_rdma_cycles:
            raise ValueError(
                "Requested RDMA (MlxJaccl) but no RDMA-connected cycles available"
            )
        smallest_cycles = smallest_rdma_cycles

    resolved_download_status = download_status or {}

    selected_cycle = max(
        smallest_cycles,
        key=lambda cycle: (
            _cycle_download_score(
                cycle, command.model_card.model_id, resolved_download_status
            ),
            sum(
                (node_memory[node_id].ram_available for node_id in cycle),
                start=Memory(),
            ),
            any(topology.node_is_leaf(node_id) for node_id in cycle),
        ),
    )
    selected_cycle = _prefer_socket_reachable_rank_zero(selected_cycle, topology)
    if sharding == Sharding.AsymmetricTensor:
        selected_cycle = _order_asymmetric_tensor_cycle(
            cycle=selected_cycle,
            node_memory=node_memory,
            topology=topology,
        )

    # Single-node target cycle requires Pipeline sharding (PP=1). The
    # backend choice depends on whether an asymmetric drafter rank will
    # extend the parent ``mx.distributed`` group beyond size 1: ring lacks
    # ``Group.split`` / ``send/recv`` so an N+1=2 parent group cannot use
    # it; jaccl supports both. We therefore peek at drafter eligibility
    # before locking the backend, then re-run the full drafter selection
    # below with the (possibly upgraded) ``instance_meta``.
    if len(selected_cycle) == 1:
        sharding = Sharding.Pipeline
        will_attempt_asymmetric_drafter = (
            bool(command.model_card.drafter_eligible_nodes)
            and bool(command.model_card.drafter_model_ids)
            and any(
                node_id in topology.list_nodes()
                and node_id not in selected_cycle.node_ids
                for node_id in command.model_card.drafter_eligible_nodes
            )
        )
        if not will_attempt_asymmetric_drafter:
            instance_meta = InstanceMeta.MlxRing
        elif instance_meta == InstanceMeta.MlxRing:
            # User asked for ring but the model declares an asymmetric
            # drafter on a separate node. Auto-upgrade to jaccl since ring
            # cannot support the parent group's split + send/recv path.
            # If jaccl reachability fails downstream, drafter selection
            # emits a degradation event and target falls back to ring
            # symmetric (no drafter), restoring V1 ring behavior.
            instance_meta = InstanceMeta.MlxJaccl

    placement_node_memory = (
        _node_memory_with_total_capacity(selected_cycle, node_memory)
        if allow_single_node_total_memory and len(selected_cycle) == 1
        else node_memory
    )
    shard_assignments = get_shard_assignments(
        command.model_card, selected_cycle, sharding, placement_node_memory
    )

    instance_id = InstanceId()
    drafter_placement = _select_drafter_placement(
        command=command,
        selected_cycle=selected_cycle,
        instance_meta=instance_meta,
        topology=topology,
        node_memory=node_memory,
        instance_id=instance_id,
        on_drafter_placement_degraded=on_drafter_placement_degraded,
    )

    # If the auto-upgrade to MlxJaccl above didn't yield a drafter (e.g.
    # no RDMA path to the eligible node), revert to MlxRing for the
    # symmetric single-rank target. The degradation event was already
    # emitted by ``_select_drafter_placement``; the user's instance
    # still completes, just without speculative decoding.
    if (
        len(selected_cycle) == 1
        and instance_meta == InstanceMeta.MlxJaccl
        and drafter_placement is None
    ):
        instance_meta = InstanceMeta.MlxRing

    # Asymmetric placement extends the parent ``mx.distributed`` group to
    # include the drafter rank as the last rank. Subgraph + connectivity
    # tables (hosts_by_node / jaccl_devices) must cover both target nodes
    # and the drafter node; ``shard_assignments`` stays target-only because
    # the drafter has no target shard.
    if drafter_placement is not None:
        nodes_for_group = list(selected_cycle.node_ids) + [
            drafter_placement.drafter_node_id
        ]
    else:
        nodes_for_group = list(selected_cycle.node_ids)
    cycle_digraph: Topology = topology.get_subgraph_from_nodes(nodes_for_group)

    target_instances = dict(deepcopy(current_instances))

    match instance_meta:
        case InstanceMeta.MlxJaccl:
            # TODO(evan): shard assignments should contain information about ranks, this is ugly
            def get_device_rank(node_id: NodeId) -> int:
                runner_id = shard_assignments.node_to_runner[node_id]
                shard_metadata = shard_assignments.runner_to_shard.get(runner_id)
                assert shard_metadata is not None
                return shard_metadata.device_rank

            zero_node_ids = [
                node_id
                for node_id in selected_cycle.node_ids
                if get_device_rank(node_id) == 0
            ]
            assert len(zero_node_ids) == 1
            coordinator_node_id = zero_node_ids[0]

            mlx_jaccl_devices = get_mlx_jaccl_devices_matrix(
                nodes_for_group,
                cycle_digraph,
            )
            mlx_jaccl_coordinators = get_mlx_jaccl_coordinators(
                coordinator=coordinator_node_id,
                coordinator_port=random_ephemeral_port(),
                cycle_digraph=cycle_digraph,
                node_network=node_network,
            )
            target_instances[instance_id] = MlxJacclInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                jaccl_devices=mlx_jaccl_devices,
                jaccl_coordinators=mlx_jaccl_coordinators,
                drafter_placement=drafter_placement,
            )
        case InstanceMeta.MlxRing:
            ephemeral_port = random_ephemeral_port()
            hosts_by_node = get_mlx_ring_hosts_by_node(
                selected_cycle=Cycle(node_ids=nodes_for_group),
                cycle_digraph=cycle_digraph,
                ephemeral_port=ephemeral_port,
                node_network=node_network,
            )
            target_instances[instance_id] = MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                hosts_by_node=hosts_by_node,
                ephemeral_port=ephemeral_port,
                drafter_placement=drafter_placement,
            )

    # Multi-node placement WITHOUT an asymmetric drafter rank still loses
    # speculative decoding (mlx_lm doesn't run draft_model on TP/PP target
    # ranks today). Degrade-loud so operators see it without crawling logs;
    # the user's request still completes.
    if (
        len(selected_cycle) > 1
        and command.model_card.drafter_model_ids
        and drafter_placement is None
    ):
        logger.warning(
            f"Model {command.model_card.model_id} declares drafters "
            f"{list(command.model_card.drafter_model_ids)} but is being "
            f"placed across {len(selected_cycle)} nodes WITHOUT an asymmetric "
            "drafter rank. Speculative decoding is single-device only and "
            "will be disabled for this instance. To get the drafter speedup, "
            "either place a smaller quant on a single node OR list a separate "
            "drafter-eligible node in the model card's `drafter_eligible_nodes`."
        )

    return target_instances


def _select_drafter_placement(
    *,
    command: PlaceInstance,
    selected_cycle: Cycle,
    instance_meta: InstanceMeta,
    topology: Topology,
    node_memory: Mapping[NodeId, MemoryUsage],
    instance_id: InstanceId,
    on_drafter_placement_degraded: (Callable[[DrafterPlacementDegraded], None] | None),
) -> DrafterPlacement | None:
    """Pick a drafter-eligible node for asymmetric drafter placement.

    A drafter rank is appended to the parent ``mx.distributed`` group when
    *all* of the following hold:

      * The model card lists ``drafter_eligible_nodes``.
      * The card lists ``drafter_model_ids`` (otherwise there's nothing to
        run on the drafter rank).
      * At least one eligible node is alive in topology, NOT already a
        target rank, AND reachable from target rank 0 over the right
        transport (RDMA for ``MlxJaccl``; socket for ``MlxRing``).

    The fallback is loud-but-graceful: when none of the eligible nodes
    satisfies the constraints, the function emits a
    :class:`DrafterPlacementDegraded` event via
    ``on_drafter_placement_degraded`` and returns ``None``. The caller
    proceeds with the legacy symmetric topology, the user's request still
    completes, and the operator sees the degradation event surfaced in
    the dashboard / API stats so they know to fix the cluster (bring an
    eligible node online, free RAM, repair the network edge).

    The drafter is always assigned the **last rank** in the parent group
    (``len(selected_cycle)``). Target ranks split off into a subgroup at
    runtime via ``mx.distributed.Group.split``.
    """
    eligible_nodes = list(command.model_card.drafter_eligible_nodes)
    drafter_candidates = list(command.model_card.drafter_model_ids)
    if not eligible_nodes or not drafter_candidates:
        return None

    target_node_ids = list(selected_cycle.node_ids)
    fallback = _drafter_fallback(target_node_ids)

    alive_in_topology = set(topology.list_nodes())
    alive_eligible = [n for n in eligible_nodes if n in alive_in_topology]
    if not alive_eligible:
        _emit_drafter_degraded(
            on_drafter_placement_degraded,
            command=command,
            instance_id=instance_id,
            target_node_ids=target_node_ids,
            eligible_nodes=eligible_nodes,
            reason=DrafterPlacementDegradationReason.NoEligibleNodeAvailable,
            fallback=fallback,
            detail=(
                f"None of {eligible_nodes} are present in topology "
                f"(known nodes: {sorted(alive_in_topology)})"
            ),
        )
        return None

    not_in_target = [n for n in alive_eligible if n not in target_node_ids]
    if not not_in_target:
        _emit_drafter_degraded(
            on_drafter_placement_degraded,
            command=command,
            instance_id=instance_id,
            target_node_ids=target_node_ids,
            eligible_nodes=eligible_nodes,
            reason=DrafterPlacementDegradationReason.AllEligibleNodesInTargetCycle,
            fallback=fallback,
            detail=(
                f"All eligible nodes {alive_eligible} are already target "
                f"ranks ({target_node_ids}); no spare host available"
            ),
        )
        return None

    requires_rdma = instance_meta == InstanceMeta.MlxJaccl
    reachable: list[NodeId] = []
    for candidate in not_in_target:
        if _drafter_node_is_reachable(
            target_node_ids=target_node_ids,
            drafter_node=candidate,
            topology=topology,
            requires_rdma=requires_rdma,
        ):
            reachable.append(candidate)

    if not reachable:
        _emit_drafter_degraded(
            on_drafter_placement_degraded,
            command=command,
            instance_id=instance_id,
            target_node_ids=target_node_ids,
            eligible_nodes=eligible_nodes,
            reason=DrafterPlacementDegradationReason.NoReachablePathFromTargetRankZero,
            fallback=fallback,
            detail=(
                f"No {'RDMA' if requires_rdma else 'socket'} path from target "
                f"ranks {target_node_ids} to any of {not_in_target}"
            ),
        )
        return None

    drafter_node_id = reachable[0]
    if not _node_has_drafter_memory(
        drafter_node=drafter_node_id,
        node_memory=node_memory,
        target_card=command.model_card,
    ):
        _emit_drafter_degraded(
            on_drafter_placement_degraded,
            command=command,
            instance_id=instance_id,
            target_node_ids=target_node_ids,
            eligible_nodes=eligible_nodes,
            reason=DrafterPlacementDegradationReason.InsufficientDrafterMemory,
            fallback=fallback,
            detail=(
                f"Drafter node {drafter_node_id} has "
                f"{node_memory[drafter_node_id].ram_available.in_gb:.1f}GB "
                f"available; conservative drafter estimate is "
                f"{_DRAFTER_MEMORY_FLOOR.in_gb:.1f}GB"
            ),
        )
        return None

    drafter_model_id = drafter_candidates[0]
    drafter_runner_id = RunnerId()
    drafter_rank = len(selected_cycle)
    return DrafterPlacement(
        drafter_node_id=drafter_node_id,
        drafter_runner_id=drafter_runner_id,
        drafter_model_id=drafter_model_id,
        drafter_rank=drafter_rank,
    )


def _drafter_fallback(target_node_ids: list[NodeId]) -> str:
    """``single_device_drafter`` when target is single-node, else ``no_drafter``.

    Multi-node target with no asymmetric drafter rank can't host the
    drafter at all (mlx_lm spec decode is single-device); single-node
    target falls back to in-process drafter as before.
    """
    return "single_device_drafter" if len(target_node_ids) == 1 else "no_drafter"


def _emit_drafter_degraded(
    callback: Callable[[DrafterPlacementDegraded], None] | None,
    *,
    command: PlaceInstance,
    instance_id: InstanceId,
    target_node_ids: list[NodeId],
    eligible_nodes: list[NodeId],
    reason: DrafterPlacementDegradationReason,
    fallback: str,
    detail: str,
) -> None:
    logger.error(
        f"Drafter placement degraded for {command.model_card.model_id} "
        f"({reason.value}): {detail}; falling back to {fallback}"
    )
    if callback is None:
        return
    assert fallback in ("single_device_drafter", "no_drafter")
    callback(
        DrafterPlacementDegraded(
            model_id=command.model_card.model_id,
            instance_id=instance_id,
            target_node_ids=target_node_ids,
            eligible_nodes=eligible_nodes,
            reason=reason,
            fallback=fallback,
            detail=detail,
        )
    )


def _drafter_node_is_reachable(
    *,
    target_node_ids: list[NodeId],
    drafter_node: NodeId,
    topology: Topology,
    requires_rdma: bool,
) -> bool:
    """Drafter must reach every target rank (and be reached from each).

    For ``MlxJaccl``: ``get_mlx_jaccl_devices_matrix`` enforces all-to-all
    RDMA across the parent group. The drafter is part of that group, so
    it needs bidirectional RDMA edges to *every* target node, not just
    rank 0.

    For ``MlxRing``: target rank 0 (the rank that does send/recv with the
    drafter via ``RemoteTransport``) must reach the drafter directly. The
    ring backend additionally wires drafter <-> rank N-1 (drafter's left
    neighbor in the parent ring). Both edges have to exist as sockets.
    Requiring all target ranks to be socket-reachable from drafter is
    over-conservative for ring, but in practice every multi-node exo
    deployment already has all-pairs socket discovery, and the cost of
    requiring it is zero. Keeps the predicate simple and matches jaccl's
    behaviour.
    """
    edge_check: Callable[[object], bool]
    if requires_rdma:
        edge_check = lambda c: isinstance(c, RDMAConnection)  # noqa: E731
    else:
        edge_check = lambda c: isinstance(c, SocketConnection)  # noqa: E731
    for target in target_node_ids:
        forward = list(topology.get_all_connections_between(target, drafter_node))
        backward = list(topology.get_all_connections_between(drafter_node, target))
        if not any(edge_check(c) for c in forward):
            return False
        if not any(edge_check(c) for c in backward):
            return False
    return True


# Conservative floor for the drafter's wired-memory bump. The drafter
# weights are usually 1-5GB (e.g. gemma-4-e2b @ 8-bit ~ 2GB), but during
# load the runner may briefly hold the safetensors mmap + decompression
# buffers; bake in headroom so placement doesn't pick a node that will
# OOM at warmup. If the drafter on disk is larger than this floor the
# runner's own ``set_wired_limit_for_model`` will catch it; this is just
# a placement-time sanity check.
_DRAFTER_MEMORY_FLOOR = Memory.from_gb(6.0)


def _node_has_drafter_memory(
    *,
    drafter_node: NodeId,
    node_memory: Mapping[NodeId, MemoryUsage],
    target_card: ModelCard,
) -> bool:
    del target_card  # reserved for future per-drafter sizing
    if drafter_node not in node_memory:
        return False
    return node_memory[drafter_node].ram_available >= _DRAFTER_MEMORY_FLOOR


def _prefer_socket_reachable_rank_zero(cycle: Cycle, topology: Topology) -> Cycle:
    """Rotate multi-node placements so rank 0 is easiest for peers to reach.

    MLX ring and JACCL both make rank 0 the listener/coordinator. Discovery can
    produce RDMA-only edges in one direction and socket control-plane edges in
    another, so putting a node with advertised inbound socket edges at rank 0
    avoids assigning the listener role to a machine peers cannot dial.
    """
    if len(cycle) <= 1:
        return cycle

    inbound_socket_edges: dict[NodeId, int] = {node_id: 0 for node_id in cycle}
    for connection in topology.list_connections():
        if connection.sink not in inbound_socket_edges:
            continue
        if isinstance(connection.edge, SocketConnection):
            inbound_socket_edges[connection.sink] += 1

    best_index = max(
        range(len(cycle.node_ids)),
        key=lambda index: (inbound_socket_edges[cycle.node_ids[index]], -index),
    )
    if best_index == 0:
        return cycle
    return Cycle(node_ids=cycle.node_ids[best_index:] + cycle.node_ids[:best_index])


def _node_memory_with_total_capacity(
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> Mapping[NodeId, MemoryUsage]:
    return {
        node_id: (
            memory_usage.model_copy(update={"ram_available": memory_usage.ram_total})
            if node_id in cycle.node_ids
            else memory_usage
        )
        for node_id, memory_usage in node_memory.items()
    }


def _order_asymmetric_tensor_cycle(
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    topology: Topology,
) -> Cycle:
    """Order an asymmetric TP cycle with the largest reachable node as rank 0."""
    ordered_cycle = Cycle(
        node_ids=sorted(
            cycle.node_ids,
            key=lambda node_id: node_memory[node_id].ram_available.in_bytes,
            reverse=True,
        )
    )
    preferred_cycle = _prefer_socket_reachable_rank_zero(ordered_cycle, topology)
    if preferred_cycle.node_ids[0] != ordered_cycle.node_ids[0]:
        raise ValueError(
            "Asymmetric tensor parallelism requires the largest-memory rank-0 "
            "node to be socket-reachable"
        )
    return ordered_cycle


def _asymmetric_tensor_rank_zero_is_socket_reachable(
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    topology: Topology,
) -> bool:
    try:
        _order_asymmetric_tensor_cycle(
            cycle=cycle,
            node_memory=node_memory,
            topology=topology,
        )
    except ValueError:
        return False
    return True


def auto_place_prefill_siblings(
    *,
    decode_instance_id: InstanceId,
    decode_instance: Instance,
    model_card: ModelCard,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    download_status: Mapping[NodeId, Sequence[DownloadProgress]] | None = None,
) -> tuple[dict[InstanceId, Instance], list[InstanceId]]:
    """Place single-rank prefill-only siblings on each viable eligible node.

    Returns a tuple of ``(new_instances, new_prefill_instance_ids)`` where
    ``new_instances`` maps newly-created prefill ``InstanceId`` to its
    placement and ``new_prefill_instance_ids`` preserves placement order.
    Both are empty when ``model_card.prefill_eligible_nodes`` is empty,
    when no candidate is alive in topology, or when every candidate fails
    feasibility (insufficient RAM, no socket reachability, etc.) -- the
    decode instance still comes up; the caller emits no
    ``InstanceLinkCreated`` and the user's traffic prefills locally on
    the target rank.

    The recursive ``place_instance`` call is invoked with a sanitised
    model card (drafter and prefill eligibility cleared) and
    ``allowed_nodes={candidate}`` to force a single-node Pipeline / PP=1
    placement. We do NOT inherit drafter placement onto prefill siblings:
    the prefill role is a pure remote-prefill server (TCP-only via
    :class:`~exo.worker.disaggregated.server.PrefillServer`), so it
    needs the target weights but not the drafter pair.
    """
    eligible = list(dict.fromkeys(model_card.prefill_eligible_nodes))
    if not eligible:
        return {}, []

    decode_nodes: set[NodeId] = set(
        decode_instance.shard_assignments.node_to_runner.keys()
    )
    if decode_instance.drafter_placement is not None:
        decode_nodes.add(decode_instance.drafter_placement.drafter_node_id)

    alive = set(topology.list_nodes())

    candidates = [
        node_id
        for node_id in eligible
        if node_id in alive and node_id not in decode_nodes
    ]
    if not candidates:
        logger.warning(
            f"Auto-prefill placement skipped for decode {decode_instance_id}: "
            f"no eligible node alive AND outside the decode cycle. "
            f"eligible={eligible} decode_nodes={sorted(decode_nodes)} "
            f"alive={sorted(alive)}"
        )
        return {}, []

    # Sanitise the recursive card so the prefill-only sibling does not
    # itself recursively spawn drafters or further prefill siblings.
    prefill_card = model_card.model_copy(
        update={
            "drafter_eligible_nodes": [],
            "drafter_model_ids": [],
            "prefill_eligible_nodes": [],
        }
    )

    placed: dict[InstanceId, Instance] = {}
    placed_ids: list[InstanceId] = []
    accumulating_instances: dict[InstanceId, Instance] = dict(current_instances)

    for candidate_node in candidates:
        sub_command = PlaceInstance(
            model_card=prefill_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        )
        try:
            sub_placement = place_instance(
                sub_command,
                topology,
                accumulating_instances,
                node_memory,
                node_network,
                allowed_nodes={candidate_node},
                download_status=download_status,
            )
        except ValueError as err:
            logger.warning(
                f"Auto-prefill skip {candidate_node} for decode "
                f"{decode_instance_id}: {err}"
            )
            continue

        new_ids_this_round = [
            iid for iid in sub_placement if iid not in accumulating_instances
        ]
        if not new_ids_this_round:
            logger.warning(
                f"Auto-prefill on {candidate_node} returned no new "
                f"instance for decode {decode_instance_id}; skipping"
            )
            continue
        for iid in new_ids_this_round:
            placed[iid] = sub_placement[iid]
            placed_ids.append(iid)
            accumulating_instances[iid] = sub_placement[iid]

    return placed, placed_ids


def delete_instance(
    command: DeleteInstance,
    current_instances: Mapping[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    target_instances = dict(deepcopy(current_instances))
    if command.instance_id in target_instances:
        del target_instances[command.instance_id]
        return target_instances
    raise ValueError(f"Instance {command.instance_id} not found")


def get_transition_events(
    current_instances: Mapping[InstanceId, Instance],
    target_instances: Mapping[InstanceId, Instance],
    tasks: Mapping[TaskId, Task],
) -> Sequence[Event]:
    events: list[Event] = []

    # find instances to create
    for instance_id, instance in target_instances.items():
        if instance_id not in current_instances:
            events.append(
                InstanceCreated(
                    instance=instance,
                )
            )

    # find instances to delete
    for instance_id in current_instances:
        if instance_id not in target_instances:
            for task in tasks.values():
                if task.instance_id == instance_id and task.task_status in [
                    TaskStatus.Pending,
                    TaskStatus.Running,
                ]:
                    events.append(
                        TaskStatusUpdated(
                            task_status=TaskStatus.Cancelled,
                            task_id=task.task_id,
                        )
                    )

            events.append(
                InstanceDeleted(
                    instance_id=instance_id,
                )
            )

    return events


def cancel_unnecessary_downloads(
    instances: Mapping[InstanceId, Instance],
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> Sequence[DownloadCommand]:
    commands: list[DownloadCommand] = []
    currently_downloading = [
        (k, v.shard_metadata.model_card.model_id)
        for k, vs in download_status.items()
        for v in vs
        if isinstance(v, (DownloadOngoing))
    ]
    active_models = set(
        (
            node_id,
            instance.shard_assignments.runner_to_shard[runner_id].model_card.model_id,
        )
        for instance in instances.values()
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items()
    )
    for pair in currently_downloading:
        if pair not in active_models:
            commands.append(CancelDownload(target_node_id=pair[0], model_id=pair[1]))

    return commands
