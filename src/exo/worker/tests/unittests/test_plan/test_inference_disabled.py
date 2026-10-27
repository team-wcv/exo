"""External inference reservations must not launch competing EXO runners."""

from typing import cast

import pytest

import exo.worker.plan as plan_mod
from exo.shared.types.tasks import Shutdown
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import RunnerReady, RunnerShuttingDown
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.runner.supervisor import RunnerSupervisor
from exo.worker.tests.constants import INSTANCE_1_ID, MODEL_A_ID, NODE_A, RUNNER_1_ID
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


@pytest.mark.parametrize("existing", [False, True])
def test_reserved_node_does_not_create_and_drains_existing_runner(
    monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    monkeypatch.setattr(plan_mod, "EXO_INFERENCE_DISABLED", True)
    instance = get_mlx_ring_instance(
        INSTANCE_1_ID,
        MODEL_A_ID,
        {NODE_A: RUNNER_1_ID},
        {RUNNER_1_ID: get_pipeline_shard_metadata(MODEL_A_ID, 0)},
    )
    runner = FakeRunnerSupervisor(
        BoundInstance(
            instance=instance, bound_node_id=NODE_A, bound_runner_id=RUNNER_1_ID
        ),
        RunnerReady(),
    )
    result = plan_mod.plan(
        NODE_A,
        {RUNNER_1_ID: cast(RunnerSupervisor, cast(object, runner))} if existing else {},
        {},
        {INSTANCE_1_ID: instance},
        {RUNNER_1_ID: RunnerReady()} if existing else {},
        {},
        {},
        {},
        KeyedBackoff(),
        KeyedBackoff(),
    )
    if existing:
        assert isinstance(result, Shutdown)
        assert result.instance_id == INSTANCE_1_ID
        assert result.runner_id == RUNNER_1_ID
    else:
        assert result is None


def test_reserved_node_does_not_repeat_in_progress_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plan_mod, "EXO_INFERENCE_DISABLED", True)
    instance = get_mlx_ring_instance(
        INSTANCE_1_ID,
        MODEL_A_ID,
        {NODE_A: RUNNER_1_ID},
        {RUNNER_1_ID: get_pipeline_shard_metadata(MODEL_A_ID, 0)},
    )
    runner = FakeRunnerSupervisor(
        BoundInstance(
            instance=instance, bound_node_id=NODE_A, bound_runner_id=RUNNER_1_ID
        ),
        RunnerShuttingDown(),
    )
    assert (
        plan_mod.plan(
            NODE_A,
            {RUNNER_1_ID: cast(RunnerSupervisor, cast(object, runner))},
            {},
            {INSTANCE_1_ID: instance},
            {RUNNER_1_ID: RunnerShuttingDown()},
            {},
            {},
            {},
            KeyedBackoff(),
            KeyedBackoff(),
        )
        is None
    )
