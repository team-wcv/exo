import pytest

from exo.main import Node


@pytest.mark.asyncio
async def test_invalid_diagnostic_snapshot_interval_falls_back(monkeypatch):
    monkeypatch.setenv("EXO_DIAGNOSTIC_SNAPSHOT_SECONDS", "15s")
    snapshots = 0

    async def stop_after_first_sleep(_seconds):
        raise RuntimeError("stop")

    def count_snapshot(_self):
        nonlocal snapshots
        snapshots += 1

    monkeypatch.setattr("exo.main.anyio.sleep", stop_after_first_sleep)
    monkeypatch.setattr(Node, "_log_diagnostic_snapshot", count_snapshot)

    with pytest.raises(RuntimeError, match="stop"):
        await Node._diagnostic_snapshot_loop(None)

    assert snapshots == 0


@pytest.mark.asyncio
async def test_non_positive_diagnostic_snapshot_interval_disables(monkeypatch):
    monkeypatch.setenv("EXO_DIAGNOSTIC_SNAPSHOT_SECONDS", "0")

    async def fail_sleep(_seconds):
        raise AssertionError("diagnostic loop should not sleep when disabled")

    monkeypatch.setattr("exo.main.anyio.sleep", fail_sleep)

    await Node._diagnostic_snapshot_loop(None)
