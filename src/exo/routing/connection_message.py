from exo_rs import FromSwarm

from exo.shared.types.common import NodeId
from exo.utils.pydantic_ext import FrozenModel

"""Serialisable types for Connection Updates/Messages"""


class ConnectionMessage(FrozenModel):
    connected: bool
    peer_id: NodeId | None = None

    @classmethod
    def from_update(cls, update: FromSwarm.Connection) -> "ConnectionMessage":
        return cls(connected=update.connected, peer_id=NodeId(update.peer_id))
