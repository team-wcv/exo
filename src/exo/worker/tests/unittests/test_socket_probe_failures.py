from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.topology import Connection, SocketConnection
from exo.utils.channels import channel
from exo.worker.main import (
    DEFAULT_SOCKET_PROBE_FAILURE_THRESHOLD,
    EXO_SOCKET_PROBE_FAILURE_THRESHOLD_ENV,
    Worker,
    _socket_probe_failure_threshold,
    _socket_probe_key,
)


def test_socket_probe_failure_threshold_env(monkeypatch):
    monkeypatch.delenv(EXO_SOCKET_PROBE_FAILURE_THRESHOLD_ENV, raising=False)
    assert _socket_probe_failure_threshold() == DEFAULT_SOCKET_PROBE_FAILURE_THRESHOLD

    monkeypatch.setenv(EXO_SOCKET_PROBE_FAILURE_THRESHOLD_ENV, "5")
    assert _socket_probe_failure_threshold() == 5

    monkeypatch.setenv(EXO_SOCKET_PROBE_FAILURE_THRESHOLD_ENV, "0")
    assert _socket_probe_failure_threshold() == 1

    monkeypatch.setenv(EXO_SOCKET_PROBE_FAILURE_THRESHOLD_ENV, "not-an-int")
    assert _socket_probe_failure_threshold() == DEFAULT_SOCKET_PROBE_FAILURE_THRESHOLD


def test_socket_probe_failures_are_tracked_per_socket_edge():
    event_tx, event_rx = channel()
    command_tx, command_rx = channel()
    download_tx, download_rx = channel()

    worker = Worker(
        node_id=NodeId("SOURCE"),
        event_receiver=event_rx,
        event_sender=event_tx,
        command_sender=command_tx,
        download_command_sender=download_tx,
        api_port=52415,
        peer_download_port=52416,
    )

    first = Connection(
        source=NodeId("SOURCE"),
        sink=NodeId("SINK"),
        edge=SocketConnection(
            sink_multiaddr=Multiaddr(address="/ip4/192.168.1.120/tcp/52415")
        ),
    )
    second = Connection(
        source=NodeId("SOURCE"),
        sink=NodeId("SINK"),
        edge=SocketConnection(
            sink_multiaddr=Multiaddr(address="/ip4/192.168.1.121/tcp/52415")
        ),
    )

    worker._socket_probe_failures[_socket_probe_key(first)] += 1

    assert worker._socket_probe_failures[_socket_probe_key(first)] == 1
    assert worker._socket_probe_failures[_socket_probe_key(second)] == 0

    event_tx.close()
    command_tx.close()
    download_tx.close()
