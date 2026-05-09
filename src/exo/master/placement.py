import re
from collections.abc import Mapping
from copy import deepcopy
from os import environ
from typing import Literal, Sequence

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
    Event,
    InstanceCreated,
    InstanceDeleted,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
    NodeRdmaCtlStatus,
)
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.topology import SocketConnection
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)
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
    node_rdma_ctl: Mapping[NodeId, NodeRdmaCtlStatus] | None = None,
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
    rdma_ctl_status = node_rdma_ctl or {}

    def _all_rdma_ctl_enabled(cycle: Cycle) -> bool:
        return all(
            ((status := rdma_ctl_status.get(node_id)) is not None and status.enabled)
            for node_id in cycle
        )

    smallest_rdma_cycles = [
        cycle
        for cycle in smallest_cycles
        if topology.is_rdma_cycle(cycle) and _all_rdma_ctl_enabled(cycle)
    ]

    if instance_meta == InstanceMeta.MlxJaccl:
        if not smallest_rdma_cycles:
            raise ValueError(
                "Requested RDMA (MlxJaccl) but no RDMA-connected cycles available"
            )
        # Filter to cycles whose every node advertises a valid Thunderbolt
        # IPv4 peer path BEFORE the scoring/selection pass. Previously the
        # preflight only ran on the already-chosen cycle, so a single
        # unrepaired node could fail placement even when another RDMA cycle
        # of the same size was perfectly valid (e.g. mixed clusters where
        # only one node is still on 169.254-only paths). When no candidate
        # is eligible we deliberately fall back to the full RDMA pool so
        # the post-selection ``_validate_jaccl_thunderbolt_ipv4_paths``
        # check still surfaces the actionable, node-specific error message
        # (which lists the missing nodes) instead of a generic
        # "no candidates" failure here.
        #
        # Codex P2 (PR #11 round 4): the JACCL prefilter must NOT run on
        # singleton cycles. A ``MlxJaccl`` request with ``min_nodes=1``
        # gets downgraded to ``MlxRing`` further down (single-node
        # JACCL is meaningless because target ranks have no peers to
        # talk to over Thunderbolt RDMA), and that downgraded ring
        # placement does not require a TB-IPv4 path. Pre-fix, requiring
        # TB-IPv4 on length-1 candidates pushed the selector toward
        # nodes that happened to have TB metadata (lower memory /
        # download score in mixed clusters) instead of letting the
        # ring downgrade pick the actual best singleton.
        jaccl_eligible_rdma_cycles = [
            cycle
            for cycle in smallest_rdma_cycles
            if len(cycle) == 1
            or all(
                _node_has_or_lacks_known_jaccl_path(node_network, node_id)
                != "known_no_path"
                for node_id in cycle.node_ids
            )
        ]
        smallest_cycles = jaccl_eligible_rdma_cycles or smallest_rdma_cycles

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

    # Single-node: force Pipeline/Ring (Tensor and Jaccl require multi-node).
    # Has to run BEFORE the JACCL Thunderbolt IPv4 preflight: a singleton
    # cycle is RDMA-capable on its own (``is_rdma_cycle`` admits length-1
    # cycles), but a JACCL request with ``min_nodes=1`` should fall through
    # to MlxRing exactly as the comment promises -- the preflight is a
    # multi-node JACCL contract (target ranks talk to each other over
    # Thunderbolt RDMA), so running it on a singleton cycle would
    # incorrectly reject placements that the downgrade-to-ring branch
    # below would have happily satisfied.
    if len(selected_cycle) == 1:
        instance_meta = InstanceMeta.MlxRing
        sharding = Sharding.Pipeline

    # Three independent post-selection adjustments. They land in this
    # order so the JACCL preflight fails fast (raising a node-specific
    # error message) before we go through the work of computing the
    # singleton total-memory expansion or the drafter-multi-node warning.
    # The first two checks are mutually exclusive in practice -- the JACCL
    # preflight only fires when ``instance_meta == MlxJaccl`` (multi-node)
    # and the ``allow_single_node_total_memory`` expansion only fires for
    # singleton cycles, which were already downgraded to ``MlxRing`` by
    # the block above -- but we keep both unconditional so the invariant
    # is encoded in the code itself rather than in a comment about
    # ordering. The drafter-multi-node warning (item 10) is purely an
    # operator hint emitted when a drafter-aware model card ends up on
    # more than one node, since speculative decoding is single-device
    # only in mlx_lm and the drafter would otherwise be silently dropped.
    if instance_meta == InstanceMeta.MlxJaccl:
        _validate_jaccl_thunderbolt_ipv4_paths(selected_cycle, node_network)

    if (
        len(selected_cycle) > 1
        and command.model_card.drafter_model_ids
    ):
        logger.warning(
            f"Model {command.model_card.model_id} declares drafters "
            f"{list(command.model_card.drafter_model_ids)} but is being "
            f"placed across {len(selected_cycle)} nodes. Speculative "
            "decoding is single-device only and will be disabled for this "
            "instance. To get the drafter speedup, place a smaller quant "
            "(e.g. 4-bit) on the largest single node instead."
        )

    placement_node_memory = (
        _node_memory_with_total_capacity(selected_cycle, node_memory)
        if allow_single_node_total_memory and len(selected_cycle) == 1
        else node_memory
    )
    shard_assignments = get_shard_assignments(
        command.model_card, selected_cycle, sharding, placement_node_memory
    )

    cycle_digraph: Topology = topology.get_subgraph_from_nodes(selected_cycle.node_ids)

    instance_id = InstanceId()
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
                [node_id for node_id in selected_cycle],
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
            )
        case InstanceMeta.MlxRing:
            ephemeral_port = random_ephemeral_port()
            hosts_by_node = get_mlx_ring_hosts_by_node(
                selected_cycle=selected_cycle,
                cycle_digraph=cycle_digraph,
                ephemeral_port=ephemeral_port,
                node_network=node_network,
            )
            target_instances[instance_id] = MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                hosts_by_node=hosts_by_node,
                ephemeral_port=ephemeral_port,
            )

    return target_instances


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


def _validate_jaccl_thunderbolt_ipv4_paths(
    cycle: Cycle,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> None:
    """Reject the placement only when we have *positive evidence* that
    a node lacks a TB-IPv4 peer path.

    Codex P1 (PR #11 round 5): ``State.node_network`` is populated by
    a best-effort async watcher and starts empty on cold-boot, so
    ``node_network.get(node_id)`` returning ``None`` is not the same
    thing as ``the node has no Thunderbolt interface``. The original
    guard collapsed both into "missing" and rejected ``MlxJaccl``
    placements whenever the gatherer hadn't run yet (or failed
    transiently for a node), even on clusters with healthy RDMA
    topology. We now distinguish the two:

    * ``known_no_path`` -- the node has gathered network info and
      none of its interfaces satisfy the Thunderbolt IPv4 predicate.
      That is genuine misconfiguration; raise with the actionable
      ``bb rdma repair`` guidance.
    * ``unknown`` -- the node has no entry in ``node_network`` (yet).
      We let placement proceed because the topology-derived RDMA
      edge already attests that some real connection exists; the
      JACCL backend will surface a clearer per-link error if the
      address turns out to be unusable at bind time.
    """
    missing_nodes = [
        node_id
        for node_id in cycle.node_ids
        if _node_has_or_lacks_known_jaccl_path(node_network, node_id) == "known_no_path"
    ]
    if missing_nodes:
        raise ValueError(
            "Requested RDMA (MlxJaccl), but the selected nodes do not advertise "
            "MLX/JACCL Thunderbolt IPv4 peer paths. Run `bb rdma repair all` and "
            "`bb rdma jaccl-status all`, then retry. Missing nodes: "
            + ", ".join(str(node_id) for node_id in missing_nodes)
        )


def _node_has_or_lacks_known_jaccl_path(
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_id: NodeId,
) -> Literal["has_path", "known_no_path", "unknown"]:
    """Three-valued JACCL preflight verdict for a single node.

    Returns ``"unknown"`` when:

    * ``node_id`` has no entry in ``node_network`` at all (the
      best-effort gatherer hasn't reported yet on this node), OR
    * the entry exists but **interface typing is missing** for every
      interface (e.g. the ``networksetup -listallhardwareports``
      parse failed on the gatherer side, so we have IP addresses
      but no ``interface_type`` field to classify them as
      thunderbolt vs ethernet vs wifi).

    Returns ``"has_path"`` when at least one Thunderbolt-style
    interface advertises a routable IPv4. Returns ``"known_no_path"``
    when typing IS available (at least one interface has a non-None,
    non-``"unknown"`` ``interface_type``) but no qualifying interface
    exists -- that's positive evidence of misconfiguration and we
    surface the actionable ``bb rdma repair`` error.

    Codex P1 (PR #11 round-(N+2)): pre-fix this helper collapsed
    "interfaces present but typing unavailable" into ``known_no_path``
    and rejected placement, even though we had no positive evidence
    that the node actually lacked a Thunderbolt path. With this
    refinement, the gatherer's partial-success/parse-failure case is
    treated as ``unknown`` and placement proceeds; the JACCL backend
    will surface a clearer per-link error if the IP turns out to be
    unusable at bind time.
    """
    info = node_network.get(node_id)
    if info is None:
        return "unknown"
    if _has_jaccl_thunderbolt_ipv4(info):
        return "has_path"
    if _interface_typing_is_missing(info):
        return "unknown"
    return "known_no_path"


# Match the exact set of macOS interface names that can plausibly be
# a Thunderbolt link or bridge:
#
# * ``en2`` ... ``en9`` and ``en10`` ... ``en9999`` -- ``en0`` and
#   ``en1`` are reserved for Wi-Fi/primary NIC by Apple convention
#   (also encoded in
#   :func:`exo.utils.info_gatherer.system_info._get_interface_types_from_networksetup`,
#   which classifies any other ``en\\d+`` as ``"maybe_ethernet"``
#   because Apple Silicon Thunderbolt bridges always live on
#   ``en2``/``en3``/``en4``). Excluding ``en0``/``en1`` prevents the
#   permissive fallback from firing on a Wi-Fi-only node whose
#   primary ``en0`` happened to land in ``"unknown"`` typing
#   (e.g. due to a transient ``networksetup`` parse failure).
# * ``bridge0`` ... ``bridge99`` -- ``bridge0`` is the canonical
#   macOS Thunderbolt Bridge service device, but
#   :func:`exo.utils.info_gatherer.info_gatherer._get_bridge_services`
#   and :func:`_find_thunderbolt_bridge` enumerate **arbitrary**
#   ``bridge\\d+`` devices and intersect their member set with the
#   Thunderbolt hardware-port device list -- a user with multiple
#   bridges (or a system that already had ``bridge0`` claimed by
#   another service) can have a real Thunderbolt Bridge exposed as
#   ``bridge1``/``bridge2``/etc. Codex P1 (PR #11 round-(N+15),
#   placement.py:567) called out that hard-coding ``bridge0`` here
#   rejects those legitimate configurations. We accept
#   ``bridge[0-9]{1,2}`` (i.e. ``bridge0``..``bridge99``); macOS
#   Internet Sharing reserves ``bridge100``+ for NAT/Parallels/
#   VirtualBox VM stacks (see ``man 8 bridge``), so excluding the
#   3-digit range still keeps VM-stack bridges out of the
#   permissive fallback.
_THUNDERBOLT_CANDIDATE_INTERFACE_NAME = re.compile(
    r"^(en[2-9]|en[1-9]\d+|bridge[0-9]{1,2})$"
)


def _is_plausible_thunderbolt_candidate(
    interface: NetworkInterfaceInfo,
) -> bool:
    """Return whether an ``"unknown"``-typed interface could plausibly
    be a Thunderbolt bridge whose hardware-port line wasn't classified.

    The heuristic limits the permissive ``unknown``-typing fallback to
    interfaces whose names exactly match the Apple/macOS Thunderbolt
    naming convention (see :data:`_THUNDERBOLT_CANDIDATE_INTERFACE_NAME`)
    AND that advertise a routable IPv4
    (:func:`_is_routable_jaccl_ipv4` filters loopback / link-local /
    unset addresses).

    Tunnel/VPN adapters (``utun*``, ``tun*``, ``tap*``, ``wg*``,
    ``gif*``, ``stf*``, ``ipsec*``), Apple Wireless Direct Link
    (``awdl*`` / ``llw*``), packet-capture (``pktap*``), loopback
    (``lo*``), Internet-Sharing/VM-stack bridges
    (``bridge100``, ``bridge101``, ...), and the Wi-Fi/primary
    leaves (``en0``, ``en1``) all fail the name check, so a
    Wi-Fi-only node that happens to have a Tailscale ``utun3``
    link or a Parallels ``bridge100`` with a routable IPv4 no
    longer slips through the JACCL preflight.

    Codex history:

    Round-(N+13) introduced the helper with regex ``^en\\d+$`` --
    too narrow because ``info_gatherer`` explicitly models the
    macOS Thunderbolt Bridge as ``bridge0`` and that device does
    not appear in ``networksetup -listallhardwareports``.

    Round-(N+14) widened to ``^(en|bridge)\\d+$`` to admit
    ``bridge0``. Codex flagged (P1, PR #11 round-(N+14),
    placement.py:548) that this re-admitted ``bridge100``
    (Parallels Desktop), ``bridge101`` (Parallels), arbitrary
    ``bridge\\d+`` from VirtualBox/VMware, AND ``en0``/``en1``
    (Wi-Fi/primary), so the Wi-Fi-only-on-VPN attack surface
    re-opened with VM-stack bridges as the new bypass vector.

    Round-(N+15) narrowed to ``^(en[2-9]|en[1-9]\\d+|bridge0)$``
    (excludes ``en0``/``en1`` and rejects every non-``bridge0``
    bridge). Codex flagged (P1, PR #11 round-(N+15),
    placement.py:567) that the gatherer's
    :func:`exo.utils.info_gatherer.info_gatherer._find_thunderbolt_bridge`
    operates on **arbitrary** ``bridgeX`` devices -- a user with
    multiple bridge services (or one whose ``bridge0`` is already
    claimed by another stack) can have a real Thunderbolt Bridge
    exposed as ``bridge1``/``bridge2``/etc., so hard-coding
    ``bridge0`` rejected legitimate TB-only configurations.

    Round-(N+16) (this commit) widens the bridge half to
    ``bridge[0-9]{1,2}`` (i.e. ``bridge0``..``bridge99``) so the
    real-Thunderbolt indices below the macOS Internet-Sharing
    reservation (``bridge100``+) are accepted, while the VM-stack
    bridges in the 3-digit range remain excluded.
    """
    if not _THUNDERBOLT_CANDIDATE_INTERFACE_NAME.match(interface.name):
        return False
    return _is_routable_jaccl_ipv4(interface.ip_address)


def _interface_typing_is_missing(network_info: NodeNetworkInfo) -> bool:
    """Heuristic for "the gatherer couldn't classify this node's
    interfaces" vs "the gatherer reports a node with no TB interfaces".

    Returns ``True`` when:

    * ``network_info`` has no interfaces at all (gatherer reported
      nothing), OR
    * **every** interface has ``interface_type == "unknown"`` (the
      gatherer's parse of ``networksetup -listallhardwareports``
      failed across the board), OR
    * **some** interface has ``interface_type == "unknown"`` AND
      passes :func:`_is_plausible_thunderbolt_candidate` (interface
      name matches ``en\\d+`` AND has a routable IPv4) -- this
      narrows the permissive fallback to genuine TB-bridge
      candidates rather than VPN/tunnel adapters with routable IPs.

    Returns ``False`` when typing IS available for every routable
    candidate -- the node has positive evidence of bad config and
    placement should reject with the actionable
    ``bb rdma repair`` error.

    Codex history:

    Round-(N+2) introduced the helper using ``all(...)`` --
    correctly handles total parse failure but rejects mixed-typing
    nodes (Wi-Fi typed plus unparsed TB bridge).

    Round-(N+11) widened to ``any(interface.interface_type ==
    "unknown" ...)`` to admit the partial-typing case. That was
    too permissive: ``get_network_interfaces`` assigns ``"unknown"``
    to interfaces not present in ``networksetup`` output (loopback,
    tunnel, etc.) so virtually every node had at least one
    unknown interface and the JACCL preflight reverted to
    permissive behavior on misconfigured clusters too -- the user
    only saw the runtime JACCL failure later.

    Round-(N+12) coupled the unknown check with routable-IPv4
    candidacy. That filtered out loopback and link-local interfaces
    but VPN/tunnel adapters (``utun*`` from Tailscale/Wireguard)
    are typed as ``"unknown"`` AND have routable ``10.x``/``100.x``
    IPv4s, so the permissive branch still fired on Wi-Fi-only nodes
    with VPNs and bypassed the preflight (Codex P1 PR #11
    round-(N+12) follow-up at placement.py:597).

    Round-(N+13) (this commit) further restricts the permissive
    fallback to the Apple ``en\\d+`` naming convention via
    :func:`_is_plausible_thunderbolt_candidate`. ``utun*`` /
    ``wg*`` / ``tun*`` / ``awdl*`` / ``lo*`` no longer satisfy the
    plausibility check, so a Wi-Fi-only node with a Tailscale tunnel
    correctly resolves to ``known_no_path`` (and the actionable
    ``bb rdma repair`` error). The legitimate Thunderbolt-bridge
    case -- ``en3`` with a routable IPv4 whose hardware-port line
    failed to parse -- still defers to ``unknown``.
    """
    if not network_info.interfaces:
        return True
    if all(
        interface.interface_type == "unknown" for interface in network_info.interfaces
    ):
        return True
    return any(
        interface.interface_type == "unknown"
        and _is_plausible_thunderbolt_candidate(interface)
        for interface in network_info.interfaces
    )


def _has_jaccl_thunderbolt_ipv4(network_info: NodeNetworkInfo | None) -> bool:
    """Return whether the node advertises at least one Thunderbolt-style
    routable IPv4 interface usable as a JACCL peer path.

    Why ``maybe_ethernet`` is accepted alongside ``thunderbolt``:
    :func:`exo.utils.info_gatherer.system_info._get_interface_types_from_networksetup`
    reclassifies any ``en*`` adapter that isn't ``en0`` / ``en1`` to
    ``"maybe_ethernet"`` regardless of what ``networksetup
    -listallhardwareports`` reports the hardware port as. On every
    cluster machine we ship, the Thunderbolt bridge sits on ``en2`` /
    ``en3`` / ``en4``, so its interface_type ends up as
    ``"maybe_ethernet"`` even though the underlying hardware is in
    fact a Thunderbolt link. Restricting the preflight to
    ``interface_type == "thunderbolt"`` rejected those (correctly
    repaired) bridges as missing, breaking placement on real
    deployments. The upstream guard ``instance_meta ==
    InstanceMeta.MlxJaccl`` already requires an RDMA-connected cycle
    (libp2p only forms RDMA edges over Thunderbolt on Apple Silicon),
    so accepting ``maybe_ethernet`` here cannot let a true LAN
    ethernet sneak past -- nodes without TB hardware would have been
    filtered upstream by the missing RDMA edge.
    """
    if network_info is None:
        return False
    return any(
        interface.interface_type in ("thunderbolt", "maybe_ethernet")
        and _is_routable_jaccl_ipv4(interface.ip_address)
        for interface in network_info.interfaces
    )


def _is_routable_jaccl_ipv4(ip_address: str) -> bool:
    """Return True iff ``ip_address`` is a syntactically-valid, unicast
    IPv4 address that's plausibly usable as a JACCL peer path.

    A valid IPv4 here is *exactly* four numeric octets in 0..255
    separated by dots, and the first octet must fall in the unicast
    range (1..223). We deliberately do not use ``ipaddress.IPv4Address``
    because that class accepts a few legacy alternate forms (e.g.
    integer-only ``"3232235521"``) that we don't want to allow as
    Thunderbolt peer paths -- the upstream gatherer always reports
    dotted-quad form, so anything else is malformed interface data
    we'd rather reject fast than parse leniently.

    Octet validation matters because malformed strings like
    ``"999.1.1.1"`` or ``"1..2.3"`` would otherwise satisfy the
    preflight (they have four split components on the dot delimiter)
    and let an ``MlxJaccl`` placement reach the runtime layer, where
    it'd fail with a far less actionable error when the JACCL backend
    tries to resolve unusable peer addresses.

    Non-unicast ranges rejected (in addition to the loopback /
    link-local / all-zero prefixes already filtered):

    - ``224.0.0.0/4`` (multicast 224..239) -- a peer path can never
      be a multicast group;
    - ``240.0.0.0/4`` (reserved / experimental 240..254) -- not
      assigned for general use, including the misconfiguration
      target ranges some Thunderbolt utilities default to;
    - ``255.255.255.255`` (limited broadcast) -- specifically
      called out by the codex review because the previous rule
      accepted it as a "valid IPv4" even though it cannot be a
      peer path.

    The unicast cap at 223 covers all three above (Class D starts at
    224, Class E at 240, broadcast falls inside Class E).
    """
    if ":" in ip_address:
        return False
    if ip_address.startswith(("0.", "127.", "169.254.")):
        return False
    octets = ip_address.split(".")
    if len(octets) != 4:
        return False
    parsed: list[int] = []
    for octet in octets:
        # Reject empty fields ("1..2.3"), non-digit characters, leading
        # whitespace, signs, etc. We don't allow leading zeros either
        # ("01.2.3.4"), since networksetup never emits them and they
        # historically trigger octal-style parsing in some libc tools.
        #
        # Codex P3 (PR #11 round 4): ``str.isdigit()`` returns True for
        # Unicode digit characters (e.g. superscript digits like
        # ``"\u00b2"``) that ``int()`` then rejects with
        # ``ValueError``. The earlier guard let those through to
        # ``int(octet)``, so a malformed network string from a
        # corrupted info-gatherer payload would raise instead of
        # cleanly returning False, aborting placement instead of
        # surfacing the routine "no eligible cycle" path. Restrict to
        # the ASCII 0-9 range explicitly.
        if not octet.isascii() or not octet.isdigit():
            return False
        if len(octet) > 1 and octet.startswith("0"):
            return False
        # Codex P2 (PR #11 round-(N+8), placement.py): even after the
        # ASCII-digit guard, ``int(octet)`` can still raise
        # ``ValueError`` because CPython enforces a string-conversion
        # digit limit (``sys.set_int_max_str_digits``, default 4300).
        # A pathological ``node_network`` payload such as
        # ``"9" * 4301 + ".1.1.1"`` would reach this line and abort
        # the placement preflight instead of returning False. The
        # contract for this helper is "never raise on malformed
        # network payloads", so cap octet length at 3 (any IPv4 octet
        # in the range 0..255 fits in three ASCII digits) before
        # attempting conversion.
        if len(octet) > 3:
            return False
        value = int(octet)
        if value < 0 or value > 255:
            return False
        parsed.append(value)
    # First octet in unicast range only (1..223). 0.* is already
    # caught above by the prefix block, but we re-check the full
    # range here for clarity and because the unicast bound rejects
    # multicast (224..239), reserved/experimental (240..254), and
    # broadcast (255). The directed-broadcast case (e.g.
    # ``192.168.10.255``) on a /24 is not generally distinguishable
    # without subnet info -- we accept it as syntactically unicast
    # and let the JACCL backend reject it on actual bind.
    return 1 <= parsed[0] <= 223


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
