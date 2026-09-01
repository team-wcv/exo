from exo_rs import FromSwarm

from exo.routing.connection_message import ConnectionMessage
from exo.shared.types.common import NodeId


def test_connection_message_preserves_peer_identity() -> None:
    update = FromSwarm.Connection(connected=True, peer_id="1" + "a" * 31)

    message = ConnectionMessage.from_update(update)

    assert message.connected is True
    assert message.peer_id == NodeId("1" + "a" * 31)
