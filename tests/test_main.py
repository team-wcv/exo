import sys

import pytest

from exo.main import Args, _zenoh_bootstrap_endpoints


def test_namespace_defaults_to_app_environment(monkeypatch) -> None:
    monkeypatch.setenv("EXO_ZENOH_NAMESPACE", "private-app-cluster")
    monkeypatch.setattr(sys, "argv", ["exo"])

    assert Args.parse().namespace == "private-app-cluster"


def test_legacy_bootstrap_peers_are_accepted(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr("exo.main.logger.warning", warnings.append)

    endpoints = _zenoh_bootstrap_endpoints(
        ["/ip4/192.168.0.2/tcp/52418/p2p/legacy-peer-id"]
    )

    assert endpoints == ["tcp/192.168.0.2:52418"]
    assert len(warnings) == 1
    assert "explicit Zenoh TCP endpoint" in warnings[0]


def test_empty_bootstrap_peers_do_not_warn(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr("exo.main.logger.warning", warnings.append)

    endpoints = _zenoh_bootstrap_endpoints([])

    assert endpoints == []
    assert warnings == []


def test_ipv6_bootstrap_peer_is_bracketed() -> None:
    assert _zenoh_bootstrap_endpoints(["/ip6/2001:db8::1/tcp/52418"]) == [
        "tcp/[2001:db8::1]:52418"
    ]


@pytest.mark.parametrize(
    "peer",
    [
        "/dns4/twin.local/tcp/52418",
        "/ip4/192.168.0.2/udp/52418",
        "/ip4/not-an-address/tcp/52418",
        "/ip4/192.168.0.2/tcp/not-a-port",
        "/ip4/192.168.0.2/tcp/70000",
    ],
)
def test_invalid_legacy_bootstrap_peer_is_rejected(peer: str) -> None:
    with pytest.raises(ValueError):
        _zenoh_bootstrap_endpoints([peer])
