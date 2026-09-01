import socket
from typing import cast

from pytest import MonkeyPatch

from exo.worker.engines.mlx.disaggregated.client import (
    _connect_prefill,  # pyright: ignore[reportPrivateUsage]
)


def test_connect_prefill_separates_connect_and_transfer_timeouts(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, int], float]] = []
    transfer_timeouts: list[float] = []

    class FakeSocket:
        def settimeout(self, value: float) -> None:
            transfer_timeouts.append(value)

    fake_socket = cast(socket.socket, cast(object, FakeSocket()))

    def fake_create_connection(
        address: tuple[str, int],
        timeout: float,
    ) -> socket.socket:
        calls.append((address, timeout))
        return fake_socket

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)

    assert _connect_prefill("10.77.44.2", 33141, 300) is fake_socket
    assert calls == [(("10.77.44.2", 33141), 5)]
    assert transfer_timeouts == [300]


def test_connect_prefill_preserves_short_caller_timeout(
    monkeypatch: MonkeyPatch,
) -> None:
    connect_timeouts: list[float] = []

    class FakeSocket:
        def settimeout(self, _value: float) -> None:
            pass

    fake_socket = cast(socket.socket, cast(object, FakeSocket()))

    def fake_create_connection(
        _address: tuple[str, int],
        timeout: float,
    ) -> socket.socket:
        connect_timeouts.append(timeout)
        return fake_socket

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)

    _connect_prefill("127.0.0.1", 1, 0.25)
    assert connect_timeouts == [0.25]
