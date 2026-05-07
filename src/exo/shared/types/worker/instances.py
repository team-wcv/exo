from enum import Enum
from typing import final

from pydantic import Field, model_validator

from exo.shared.models.model_cards import ModelTask
from exo.shared.types.common import Host, Id, ModelId, NodeId
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"


@final
class DrafterPlacement(FrozenModel):
    """Locator for an asymmetric drafter rank inside an :class:`Instance`.

    The drafter runs on a separate node from the target ranks but
    participates in the same parent ``mx.distributed`` group, so the
    target rank that owns sampling decisions (``target_subgroup`` rank
    0) can reach it via point-to-point ``send/recv``. Target ranks split
    off into ``target_subgroup`` for their own collectives; the drafter
    drops into a size-1 subgroup of its own. The parent group is
    reserved for target<->drafter point-to-point IPC.

    Convention: ``drafter_rank`` is the **last** rank in the parent
    group (``parent_world_size - 1``). Placement enforces this so the
    runtime never has to guess where the drafter lives.

    Fields:
        drafter_node_id:    Where the drafter runner lives.
        drafter_runner_id:  Identifies the drafter runner; the bootstrap
                            checks ``bound_runner_id == drafter_runner_id``
                            to switch into drafter-only loading mode and
                            enter the drafter serve loop instead of the
                            normal generation engine.
        drafter_model_id:   Which drafter weights to load. Must be one
                            of the entries in the target's
                            ``ModelCard.drafter_model_ids`` list
                            (placement enforces this invariant).
        drafter_rank:       Rank of the drafter inside the parent
                            ``mx.distributed`` group. Conventionally
                            ``parent_world_size - 1``.
    """

    drafter_node_id: NodeId
    drafter_runner_id: RunnerId
    drafter_model_id: ModelId
    drafter_rank: int = Field(ge=0)


class BaseInstance(TaggedModel):
    instance_id: InstanceId
    shard_assignments: ShardAssignments
    # When set, this instance places the drafter on a separate node from
    # the target ranks and routes drafter/verify IPC over the parent
    # ``mx.distributed`` group. ``None`` (the default) preserves legacy
    # symmetric placement: every rank in ``shard_assignments`` runs a
    # target shard, and any drafter declared on the model card is loaded
    # in-process alongside the target on the single-device cycle.
    drafter_placement: DrafterPlacement | None = None

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)

    @property
    def parent_group_size(self) -> int:
        """Size of the parent ``mx.distributed`` group.

        Equals ``len(shard_assignments.runner_to_shard)`` for symmetric
        placement (every rank is a target shard), or that count + 1 when
        an asymmetric drafter rank is appended.
        """
        target_world_size = len(self.shard_assignments.runner_to_shard)
        if self.drafter_placement is not None:
            return target_world_size + 1
        return target_world_size

    def is_drafter_runner(self, runner_id: RunnerId) -> bool:
        return (
            self.drafter_placement is not None
            and self.drafter_placement.drafter_runner_id == runner_id
        )


class MlxRingInstance(BaseInstance):
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int


class MlxJacclInstance(BaseInstance):
    jaccl_devices: list[list[str | None]]
    jaccl_coordinators: dict[NodeId, str]


# TODO: Single node instance
Instance = MlxRingInstance | MlxJacclInstance


class BoundInstance(FrozenModel):
    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @property
    def is_drafter_rank(self) -> bool:
        """``True`` when this runner serves the drafter, not a target shard.

        Callers that read ``bound_shard``, ``is_image_model``, or any
        target-shard-derived property MUST branch on this first; those
        properties raise on a drafter-rank bound instance because the
        drafter has no target shard.
        """
        return self.instance.is_drafter_runner(self.bound_runner_id)

    @property
    def bound_shard(self) -> ShardMetadata:
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None, (
            "bound_shard is only defined for target ranks; "
            "check `is_drafter_rank` before reading it"
        )
        return shard

    @property
    def is_image_model(self) -> bool:
        if self.is_drafter_rank:
            return False
        return (
            ModelTask.TextToImage in self.bound_shard.model_card.tasks
            or ModelTask.ImageToImage in self.bound_shard.model_card.tasks
        )

    @model_validator(mode="after")
    def validate_runner_known(self) -> "BoundInstance":
        if self.bound_runner_id in self.instance.shard_assignments.runner_to_shard:
            return self
        if self.instance.is_drafter_runner(self.bound_runner_id):
            placement = self.instance.drafter_placement
            assert placement is not None  # type narrowed by is_drafter_runner
            assert self.bound_node_id == placement.drafter_node_id, (
                f"Drafter runner {self.bound_runner_id} bound to node "
                f"{self.bound_node_id}, but DrafterPlacement points to "
                f"{placement.drafter_node_id}"
            )
            return self
        raise AssertionError(
            f"bound_runner_id {self.bound_runner_id} is neither a target rank "
            f"in shard_assignments nor the drafter rank declared by "
            f"instance.drafter_placement"
        )
