# pyright: reportPrivateUsage=false

from datetime import datetime, timedelta, timezone
from types import MethodType

import anyio
import pytest
from fastapi import HTTPException

from exo.api.cat_mode import CatModeController, CatModeRequest, Readiness
from exo.api.main import API
from exo.api.tests.test_agent_endpoints import _api_with_instances, _instance
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.commands import (
    ForwarderCommand,
    TaskCancelled,
    TaskFinished,
    TextGeneration,
)
from exo.shared.types.common import ModelId, NodeId, SystemId
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerReady,
    RunnerRunning,
)
from exo.utils.channels import channel

MODEL = ModelId("test/cat")
INSTANCE = InstanceId("cat-instance")


async def test_expiry_cancels_generation_and_releases_controller():
    entered = anyio.Event()
    cancelled = anyio.Event()

    async def generate(_model: ModelId, _instance: InstanceId, _number: int):
        entered.set()
        try:
            await anyio.sleep_forever()
        finally:
            cancelled.set()

    controller: CatModeController | None = None
    async with anyio.create_task_group() as group:
        controller = CatModeController(
            enabled=True,
            generate=generate,
            readiness=lambda _: "ready",
            spawn=group.start_soon,
            duration=0.05,
        )
        controller.start(MODEL, INSTANCE)
        await entered.wait()
        with pytest.raises(ValueError):
            controller.start(MODEL, INSTANCE)
    assert controller is not None
    assert cancelled.is_set()
    assert not controller.status.active
    assert controller.status.state == "finished"


@pytest.mark.parametrize("immediate", [True, False])
async def test_stop_and_restart_race(immediate: bool):
    entered = anyio.Event()

    async def generate(_model: ModelId, _instance: InstanceId, _number: int):
        entered.set()
        await anyio.sleep_forever()

    controller: CatModeController | None = None
    async with anyio.create_task_group() as group:
        controller = CatModeController(
            enabled=True,
            generate=generate,
            readiness=lambda _: "ready",
            spawn=group.start_soon,
        )
        controller.start(MODEL, INSTANCE)
        if not immediate:
            await entered.wait()
        controller.stop()
        with pytest.raises(ValueError):
            controller.start(MODEL, INSTANCE)
    assert controller is not None
    assert controller.status.state == "stopped"
    assert not controller.status.active
    assert entered.is_set() is not immediate


async def test_busy_waits_then_unavailable_stops_without_generating():
    readiness: Readiness = "busy"
    calls = 0

    async def generate(_model: ModelId, _instance: InstanceId, _number: int):
        nonlocal calls
        calls += 1

    controller: CatModeController | None = None
    async with anyio.create_task_group() as group:
        controller = CatModeController(
            enabled=True,
            generate=generate,
            readiness=lambda _: readiness,
            spawn=group.start_soon,
            pause=0.001,
        )
        controller.start(MODEL, INSTANCE)
        await anyio.sleep(0.01)
        assert calls == 0
        readiness = "ready"
        with anyio.fail_after(1):
            while calls == 0:
                await anyio.sleep(0.001)
        readiness = "unavailable"
    assert controller is not None
    assert controller.status.state == "error"
    assert controller.status.completed_requests == calls


async def test_generation_failure_is_contained():
    async def generate(_model: ModelId, _instance: InstanceId, _number: int):
        raise RuntimeError("runner unavailable")

    controller: CatModeController | None = None
    async with anyio.create_task_group() as group:
        controller = CatModeController(
            enabled=True,
            generate=generate,
            readiness=lambda _: "ready",
            spawn=group.start_soon,
        )
        controller.start(MODEL, INSTANCE)
    assert controller is not None
    assert controller.status.error == "runner unavailable"
    assert controller is not None
    assert controller.status.state == "error"


def ready_api() -> API:
    api = _api_with_instances({INSTANCE: _instance(MODEL, INSTANCE)})
    api.paused = False
    api.state.topology.add_node(NodeId("node-one"))
    api.state = api.state.model_copy(
        update={
            "runners": {
                RunnerId(f"runner-{INSTANCE}"): RunnerReady(),
                RunnerId("old"): RunnerFailed(
                    error_message="historical", diagnostics=[]
                ),
            },
            "last_seen": {NodeId("node-one"): datetime.now(timezone.utc)},
        }
    )
    return api


def test_only_active_instance_runners_and_fresh_nodes_are_eligible():
    api = ready_api()
    assert api._cat_mode_readiness(INSTANCE) == "ready"
    api.state = api.state.model_copy(
        update={"runners": {RunnerId(f"runner-{INSTANCE}"): RunnerRunning()}}
    )
    assert api._cat_mode_readiness(INSTANCE) == "busy"
    api.state = api.state.model_copy(
        update={
            "last_seen": {
                NodeId("node-one"): datetime.now(timezone.utc) - timedelta(seconds=60)
            }
        }
    )
    assert api._cat_mode_readiness(INSTANCE) == "unavailable"


async def test_disabled_missing_and_duplicate_start():
    api = ready_api()

    async def generate(_model: ModelId, _instance: InstanceId, _number: int):
        await anyio.sleep_forever()

    async with anyio.create_task_group() as group:
        api._cat_mode = CatModeController(
            enabled=False,
            generate=generate,
            readiness=api._cat_mode_readiness,
            spawn=group.start_soon,
        )
        with pytest.raises(HTTPException) as disabled:
            await api.start_cat_mode(CatModeRequest(model_id=MODEL))
        assert disabled.value.status_code == 404
        api._cat_mode.status.enabled = True
        with pytest.raises(HTTPException) as missing:
            await api.start_cat_mode(CatModeRequest(model_id=ModelId("missing")))
        assert missing.value.status_code == 409
        await api.start_cat_mode(CatModeRequest(model_id=MODEL))
        with pytest.raises(HTTPException) as duplicate:
            await api.start_cat_mode(CatModeRequest(model_id=MODEL))
        assert duplicate.value.status_code == 409
        await api.stop_cat_mode()


@pytest.mark.parametrize("complete", [True, False])
async def test_generation_pins_route_and_cleans_up_on_cancellation(complete: bool):
    api = ready_api()
    api._system_id = SystemId()
    api.command_sender, commands = channel[ForwarderCommand]()
    seen: list[TextGeneration] = []

    async def send(self: API, command: TextGeneration):
        seen.append(command)
        sender, receiver = channel[TokenChunk]()
        self._text_generation_queues[command.command_id] = sender  # type: ignore[assignment]
        self._text_generation_receivers[command.command_id] = receiver  # type: ignore[assignment]
        if complete:
            sender.send_nowait(
                TokenChunk(
                    model=MODEL,
                    text="purr",
                    token_id=1,
                    usage=None,
                    finish_reason="stop",
                )
            )

    api._send = MethodType(send, api)
    with anyio.move_on_after(0.02):
        await api._cat_mode_generate(MODEL, INSTANCE, 1)
    assert seen[0].target_instance_id == INSTANCE
    assert seen[0].task_params.enable_thinking is False
    assert seen[0].task_params.max_output_tokens == 512
    if not complete:
        assert isinstance((await commands.receive()).command, TaskCancelled)
    assert isinstance((await commands.receive()).command, TaskFinished)
    assert not api._text_generation_queues
    assert not api._text_generation_receivers


def test_prefill_only_and_orphaned_instances_cannot_run_cat_mode():
    from exo.shared.types.instance_link import InstanceLink, InstanceLinkId

    api = ready_api()
    link_id = InstanceLinkId("prefill-link")
    api.state = api.state.model_copy(
        update={
            "instance_links": {
                link_id: InstanceLink(
                    link_id=link_id,
                    prefill_instances=[INSTANCE],
                    decode_instances=[InstanceId("other")],
                )
            }
        }
    )
    assert api._cat_mode_readiness(INSTANCE) == "unavailable"
    api.state = api.state.model_copy(
        update={
            "instance_links": {},
            "orphaned_prefill_instances": frozenset({INSTANCE}),
        }
    )
    assert api._cat_mode_readiness(INSTANCE) == "unavailable"


async def test_api_shutdown_cancels_controller():
    entered = anyio.Event()

    async def generate(_model: ModelId, _instance: InstanceId, _number: int):
        entered.set()
        await anyio.sleep_forever()

    controller: CatModeController | None = None
    async with anyio.create_task_group() as group:
        controller = CatModeController(
            enabled=True,
            generate=generate,
            readiness=lambda _: "ready",
            spawn=group.start_soon,
        )
        controller.start(MODEL, INSTANCE)
        await entered.wait()
        group.cancel_scope.cancel()
    assert controller is not None
    assert controller.status.state == "stopped"
    assert controller.status.active is False
