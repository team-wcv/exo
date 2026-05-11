from collections.abc import Sequence
from typing import NamedTuple

from pydantic import model_validator

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import Id, NodeId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class RunnerId(Id):
    pass


class RunnerError(Exception):
    pass


class BaseRunnerStatus(TaggedModel):
    def is_running(self):
        return isinstance(self, RunnerRunning)


class RunnerIdle(BaseRunnerStatus):
    pass


class RunnerConnecting(BaseRunnerStatus):
    pass


class RunnerConnected(BaseRunnerStatus):
    pass


class RunnerLoading(BaseRunnerStatus):
    layers_loaded: int = 0
    total_layers: int = 0


class RunnerLoaded(BaseRunnerStatus):
    pass


class RunnerWarmingUp(BaseRunnerStatus):
    pass


class RunnerReady(BaseRunnerStatus):
    prefill_server_port: int | None = None


class RunnerRunning(BaseRunnerStatus):
    pass


class RunnerShuttingDown(BaseRunnerStatus):
    pass


class RunnerShutdown(BaseRunnerStatus):
    pass


class RunnerFailed(BaseRunnerStatus):
    error_message: str | None = None


RunnerStatus = (
    RunnerIdle
    | RunnerConnecting
    | RunnerConnected
    | RunnerLoading
    | RunnerLoaded
    | RunnerWarmingUp
    | RunnerReady
    | RunnerRunning
    | RunnerShuttingDown
    | RunnerShutdown
    | RunnerFailed
)


class ShardWithId(NamedTuple):
    node_id: NodeId
    runner_id: RunnerId
    shard: ShardMetadata


class ShardAssignments(FrozenModel):
    model_id: ModelId
    shards: Sequence[ShardWithId]
    # this node needs to be connected to the API node for the stream to be considered ready
    # (this is a device rank)
    primary_output_node: int

    @model_validator(mode="after")
    def validate_runners_exist(self) -> "ShardAssignments":
        if not self.shards[self.primary_output_node].shard.is_primary_output():
            raise ValueError("primary output node does not correspond to primary shard")

        return self

    @property
    def runner_to_shard(self) -> dict[RunnerId, ShardMetadata]:
        """Backwards-compatible mapping derived from ``self.shards``.

        The canonical storage is now ``shards: Sequence[ShardWithId]``
        per the upstream PR #2058 refactor. This property preserves the
        legacy dict accessor so existing fork callers (tests, drafter
        bookkeeping, asymmetric-TP helpers) keep working without
        touching every call site at once. New code SHOULD iterate
        ``shards`` directly. Treat this dict as read-only; mutating it
        does not write through to the underlying sequence.
        """
        return {rid: shard for _, rid, shard in self.shards}

    @property
    def node_to_runner(self) -> dict[NodeId, RunnerId]:
        """Backwards-compatible mapping derived from ``self.shards``.

        See ``runner_to_shard`` for the rationale. When multiple shards
        share a node (theoretically possible under the new sequence
        format but not exercised by current placements) the last shard
        wins, matching the dict-collapse behaviour of the legacy
        mapping that this property replaces.
        """
        return {nid: rid for nid, rid, _ in self.shards}
