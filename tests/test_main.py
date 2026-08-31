from exo.main import _handle_legacy_bootstrap_peers


def test_legacy_bootstrap_peers_are_accepted(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr("exo.main.logger.warning", warnings.append)

    _handle_legacy_bootstrap_peers(["/ip4/192.168.0.2/tcp/52418"])

    assert len(warnings) == 1
    assert "ignored by Zenoh" in warnings[0]


def test_empty_bootstrap_peers_do_not_warn(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr("exo.main.logger.warning", warnings.append)

    _handle_legacy_bootstrap_peers([])

    assert warnings == []
