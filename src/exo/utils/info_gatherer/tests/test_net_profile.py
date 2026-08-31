# pyright: reportPrivateUsage=false

from pytest import MonkeyPatch

from exo.utils.info_gatherer.net_profile import _probeable_interface


def test_probeable_interface_rejects_non_lan_paths(monkeypatch: MonkeyPatch):
    monkeypatch.delenv("EXO_REACHABILITY_ALLOWED_CIDRS", raising=False)

    assert _probeable_interface("en11", "192.168.1.63")
    assert not _probeable_interface("lo0", "127.0.0.1")
    assert not _probeable_interface("en5", "169.254.108.157")
    assert not _probeable_interface("utun0", "100.80.253.10")
    assert not _probeable_interface("utun0", "fd7a:115c:a1e0::9938:fd0a")
    assert not _probeable_interface("awdl0", "fe80::1406:6dff:fed0:b945%awdl0")


def test_probeable_interface_honors_allowed_cidrs(monkeypatch: MonkeyPatch):
    monkeypatch.delenv("EXO_REACHABILITY_ALLOWED_IPS", raising=False)
    monkeypatch.setenv("EXO_REACHABILITY_ALLOWED_CIDRS", "192.168.1.0/24")

    assert _probeable_interface("en11", "192.168.1.63")
    assert not _probeable_interface("en2", "192.168.0.2")
    assert not _probeable_interface("bridge100", "192.168.2.1")


def test_probeable_interface_honors_allowed_ips(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("EXO_REACHABILITY_ALLOWED_IPS", "192.168.1.63,192.168.1.120")
    monkeypatch.setenv("EXO_REACHABILITY_ALLOWED_CIDRS", "192.168.1.0/24")

    assert _probeable_interface("en11", "192.168.1.63")
    assert _probeable_interface("en0", "192.168.1.120")
    assert not _probeable_interface("en0", "192.168.1.38")
    assert not _probeable_interface("en16", "192.168.1.224")
