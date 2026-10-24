"""Reserved nodes stay observable without advertising inference capacity."""

import pytest

import exo.utils.info_gatherer.info_gatherer as gatherer
from exo.shared.types.backends import Backend


@pytest.mark.anyio
async def test_reserved_node_advertises_no_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gatherer, "EXO_INFERENCE_DISABLED", True)
    assert (await gatherer.NodeBackends.gather()).backends == []


@pytest.mark.anyio
async def test_normal_node_keeps_cpu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gatherer, "EXO_INFERENCE_DISABLED", False)
    assert Backend.MlxCpu in (await gatherer.NodeBackends.gather()).backends
