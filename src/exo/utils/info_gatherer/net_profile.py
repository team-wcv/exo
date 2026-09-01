import ipaddress
import os
from collections import defaultdict
from collections.abc import AsyncGenerator, Mapping

import anyio
import httpx
from anyio import create_task_group
from loguru import logger

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NodeNetworkInfo
from exo.utils.channels import Sender, channel

REACHABILITY_ATTEMPTS = 3
REACHABILITY_ALLOWED_IPS_ENV = "EXO_REACHABILITY_ALLOWED_IPS"
REACHABILITY_ALLOWED_CIDRS_ENV = "EXO_REACHABILITY_ALLOWED_CIDRS"
_IGNORED_INTERFACE_PREFIXES = ("lo", "utun", "awdl", "llw")
_TAILSCALE_IPV4 = ipaddress.ip_network("100.64.0.0/10")
_TAILSCALE_IPV6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")


def _parse_allowed_cidrs() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    raw = os.getenv(REACHABILITY_ALLOWED_CIDRS_ENV, "")
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        try:
            networks.append(ipaddress.ip_network(value, strict=False))
        except ValueError:
            logger.warning(
                f"Ignoring invalid {REACHABILITY_ALLOWED_CIDRS_ENV} entry {value!r}"
            )
    return tuple(networks)


def _parse_allowed_ips() -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    raw = os.getenv(REACHABILITY_ALLOWED_IPS_ENV, "")
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        try:
            ips.append(ipaddress.ip_address(value.split("%", 1)[0]))
        except ValueError:
            logger.warning(
                f"Ignoring invalid {REACHABILITY_ALLOWED_IPS_ENV} entry {value!r}"
            )
    return tuple(ips)


def _probeable_interface(name: str, ip_address: str) -> bool:
    if name.startswith(_IGNORED_INTERFACE_PREFIXES):
        return False

    try:
        ip = ipaddress.ip_address(ip_address.split("%", 1)[0])
    except ValueError:
        return False

    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        return False
    if ip in _TAILSCALE_IPV4 or ip in _TAILSCALE_IPV6:
        return False

    allowed_ips = _parse_allowed_ips()
    if allowed_ips and ip not in allowed_ips:
        return False

    allowed_cidrs = _parse_allowed_cidrs()
    return not allowed_cidrs or any(ip in network for network in allowed_cidrs)


async def check_reachability(
    target_ip: str,
    expected_node_id: NodeId,
    out: dict[NodeId, set[str]],
    client: httpx.AsyncClient,
    api_port: int,
) -> None:
    """Check if a node is reachable at the given IP and verify its identity."""
    if ":" in target_ip:
        # TODO: use real IpAddress types
        url = f"http://[{target_ip}]:{api_port}/node_id"
    else:
        url = f"http://{target_ip}:{api_port}/node_id"

    remote_node_id = None
    last_error = None

    for _ in range(REACHABILITY_ATTEMPTS):
        try:
            r = await client.get(url)
            if r.status_code != 200:
                await anyio.sleep(1)
                continue

            body = r.text.strip().strip('"')
            if not body:
                await anyio.sleep(1)
                continue

            remote_node_id = NodeId(body)
            break

        # expected failure cases
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
        ):
            await anyio.sleep(1)

        # other failures should be logged on last attempt
        except httpx.HTTPError as e:
            last_error = e
            await anyio.sleep(1)

    if last_error is not None:
        logger.warning(
            f"connect error {type(last_error).__name__} from {target_ip} after {REACHABILITY_ATTEMPTS} attempts; treating as down"
        )

    if remote_node_id is None:
        return

    if remote_node_id != expected_node_id:
        logger.debug(
            f"Discovered node with unexpected node_id; "
            f"ip={target_ip}, expected_node_id={expected_node_id}, "
            f"remote_node_id={remote_node_id}"
        )
        return

    if remote_node_id not in out:
        out[remote_node_id] = set()
    out[remote_node_id].add(target_ip)


async def check_reachable(
    topology: Topology,
    self_node_id: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    api_port: int,
) -> AsyncGenerator[tuple[str, NodeId], None]:
    """Yield (ip, node_id) pairs as reachability probes complete."""

    send, recv = channel[tuple[str, NodeId]]()

    # these are intentionally httpx's defaults so we can tune them later
    timeout = httpx.Timeout(timeout=5.0)
    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=5,
    )

    async def _probe(
        target_ip: str,
        expected_node_id: NodeId,
        client: httpx.AsyncClient,
        send: Sender[tuple[str, NodeId]],
    ) -> None:
        async with send:
            out: defaultdict[NodeId, set[str]] = defaultdict(set)
            await check_reachability(target_ip, expected_node_id, out, client, api_port)
            if expected_node_id in out:
                await send.send((target_ip, expected_node_id))

    async with (
        httpx.AsyncClient(timeout=timeout, limits=limits, verify=False) as client,
        create_task_group() as tg,
    ):
        for node_id in topology.list_nodes():
            if node_id not in node_network:
                continue
            if node_id == self_node_id:
                continue
            for iface in node_network[node_id].interfaces:
                if not _probeable_interface(iface.name, iface.ip_address):
                    continue
                tg.start_soon(_probe, iface.ip_address, node_id, client, send.clone())
        send.close()

        with recv:
            async for item in recv:
                yield item
