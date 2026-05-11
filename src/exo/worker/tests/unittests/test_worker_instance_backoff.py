# pyright: reportPrivateUsage=false

from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardWithId
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.main import Worker


def _make_instance(instance_id: InstanceId) -> MlxRingInstance:
    # Single-rank placeholder: PR #2058 (upstream) made
    # ``ShardAssignments`` reject empty shard sequences, so this
    # backoff-reconcile fixture now carries one minimal shard. The
    # backoff path under test only reads ``instance_id``, not the
    # shard contents.
    node_id = NodeId("node-1")
    runner_id = RunnerId()
    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1000),
            n_layers=1,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"),
            shards=[ShardWithId(node_id, runner_id, shard)],
            primary_output_node=0,
        ),
        hosts_by_node={node_id: []},
        ephemeral_port=1,
    )


def test_worker_reconciles_instance_backoff_from_state() -> None:
    live_instance_id = InstanceId("inst-live")
    deleted_instance_id = InstanceId("inst-deleted")
    worker = object.__new__(Worker)
    worker.state = State(instances={live_instance_id: _make_instance(live_instance_id)})
    worker._instance_backoff = KeyedBackoff[InstanceId]()
    worker._instance_backoff.record_attempt(live_instance_id)
    worker._instance_backoff.record_attempt(deleted_instance_id)

    worker._reconcile_instance_backoff_once()

    assert worker._instance_backoff.attempts(live_instance_id) == 1
    assert worker._instance_backoff.attempts(deleted_instance_id) == 0
