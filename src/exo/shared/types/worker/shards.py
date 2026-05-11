from enum import Enum
from typing import TypeAlias, final

from pydantic import Field

from exo.shared.models.model_cards import ModelCard
from exo.utils.pydantic_ext import TaggedModel


class Sharding(str, Enum):
    Tensor = "Tensor"
    AsymmetricTensor = "AsymmetricTensor"
    Pipeline = "Pipeline"


class BaseShardMetadata(TaggedModel):
    """
    Defines a specific shard of the model that is ready to be run on a device.
    Replaces previous `Shard` object.

    Layers are represented as a half-open interval [start_layer, end_layer),
    where start_layer is inclusive and end_layer is exclusive.
    """

    model_card: ModelCard
    device_rank: int
    world_size: int

    start_layer: int = Field(ge=0)
    end_layer: int = Field(ge=0)
    n_layers: int = Field(ge=0)

    @property
    def is_first_layer(self) -> bool:
        return self.start_layer == 0

    @property
    def is_last_layer(self) -> bool:
        return self.end_layer == self.n_layers

    def __hash__(self) -> int:
        return hash(
            (
                self.model_card.model_id,
                self.start_layer,
                self.end_layer,
                self.n_layers,
                self.device_rank,
                self.world_size,
            )
        )

    def is_primary_output(self) -> bool:
        return self.device_rank == self.world_size - 1


@final
class PipelineShardMetadata(BaseShardMetadata):
    pass


@final
class CfgShardMetadata(BaseShardMetadata):
    # example
    # world_size 6
    # rank prank crank
    #  0     0     0
    #  1     1     0
    #  2     2     0
    #  3     2     1
    #  4     1     1
    #  5     0     1

    @property
    def cfg_rank(self) -> int:
        # 0 = positive branch, 1 = negative branch
        return 0 if self.device_rank < self.world_size // 2 else 1

    @property
    def cfg_world_size(self) -> int:
        return 2

    @property
    def pipeline_rank(self) -> int:
        return (
            self.device_rank
            if self.cfg_rank == 0
            else (self.world_size - self.device_rank - 1)
        )

    @property
    def pipeline_world_size(self) -> int:
        return self.world_size // 2

    def is_primary_output(self) -> bool:
        assert self.pipeline_world_size == self.world_size // 2
        assert self.world_size % 2 == 0
        return self.device_rank == (self.world_size // 2) - 1


@final
class TensorShardMetadata(BaseShardMetadata):
    pass


@final
class AsymmetricTensorShardMetadata(BaseShardMetadata):
    """
    Asymmetric tensor parallelism shard metadata.

    Unlike standard tensor parallelism which splits weights 50/50 (or equally
    across N nodes), asymmetric TP splits weights proportionally to each node's
    available memory. This enables heterogeneous clusters (e.g. 128GB + 48GB)
    to run models using tensor parallelism where equal splits wouldn't fit.

    Each node holds a different fraction of each weight tensor, but ALL nodes
    compute every layer simultaneously. The all_sum reduction still works
    correctly because (x_a @ W_a^T) + (x_b @ W_b^T) = x @ W^T regardless
    of how W is partitioned.
    """

    ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Split point for rank 0, shared across all ranks. "
        "e.g. 0.75 means rank 0 gets the first 75% and rank 1 gets the last 25%. "
        "Every rank stores the same value so all workers agree on the split.",
    )


ShardMetadata: TypeAlias = (
    PipelineShardMetadata
    | CfgShardMetadata
    | TensorShardMetadata
    | AsymmetricTensorShardMetadata
)
