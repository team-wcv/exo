# pyright: reportPrivateUsage=false

from types import MethodType
from typing import Any

import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.commands import TextGeneration
from exo.shared.types.common import Host, ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


class _RequestStub:
    base_url = "http://testserver/"


def _model_card(model_id: ModelId) -> ModelCard:
    return ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def _instance(model_id: ModelId, instance_id: InstanceId) -> MlxRingInstance:
    node_id = NodeId("node-one")
    runner_id = RunnerId(f"runner-{instance_id}")
    shard = PipelineShardMetadata(
        model_card=_model_card(model_id),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=model_id,
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={node_id: [Host(ip="127.0.0.1", port=1)]},
        ephemeral_port=1,
    )


def _api_with_instances(
    instances: dict[InstanceId, MlxRingInstance],
) -> API:
    api = API.__new__(API)
    api.state = State(instances=instances)

    async def _noop_notify(_: API, __: ModelId) -> None:
        return None

    api._trigger_notify_user_to_download_model = MethodType(_noop_notify, api)
    return api


def _text_params(model_id: ModelId) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=model_id,
        input=[
            InputMessage(role="user", content=InputMessageContent("hello")),
        ],
    )


def test_provider_list_includes_default_model_and_instance_endpoints() -> None:
    model_id = ModelId("mlx-community/Test-Model-4bit")
    instance_id = InstanceId("instance-one")
    api = _api_with_instances({instance_id: _instance(model_id, instance_id)})

    providers = api.get_agent_endpoints(_RequestStub()).data  # type: ignore[arg-type]

    assert providers[0].name == "default"
    assert providers[0].openai_base_url == "http://testserver/v1"
    assert providers[0].claude_base_url == "http://testserver"
    assert any(
        provider.kind == "model"
        and provider.model_id == model_id
        and provider.target_instance_id is None
        and provider.claude_base_url is None
        for provider in providers
    )
    assert any(
        provider.kind == "instance"
        and provider.model_id == model_id
        and provider.target_instance_id == instance_id
        and provider.openai_base_url
        == "http://testserver/agents/inst-instance-one/v1"
        and provider.claude_base_url is None
        for provider in providers
    )


@pytest.mark.asyncio
async def test_agent_chat_dispatches_with_target_instance_id() -> None:
    model_id = ModelId("mlx-community/Test-Model-4bit")
    instance_id = InstanceId("instance-one")
    api = _api_with_instances({instance_id: _instance(model_id, instance_id)})
    captured: dict[str, Any] = {}

    async def _capture_send(
        self: API,
        task_params: TextGenerationTaskParams,
        target_instance_id: InstanceId | None = None,
    ) -> TextGeneration:
        captured["model"] = task_params.model
        captured["target_instance_id"] = target_instance_id
        return TextGeneration(
            task_params=task_params, target_instance_id=target_instance_id
        )

    api._send_text_generation_with_images = MethodType(_capture_send, api)
    route = api._resolve_text_generation_route(
        ModelId("ignored-request-model"), f"inst-{instance_id}", _RequestStub()  # type: ignore[arg-type]
    )

    command = await api._send_routed_text_generation(
        _text_params(ModelId("ignored-request-model")), route
    )

    assert captured == {"model": model_id, "target_instance_id": instance_id}
    assert command.task_params.model == model_id
    assert command.target_instance_id == instance_id


@pytest.mark.asyncio
async def test_model_endpoint_dispatches_without_target_instance_id() -> None:
    model_id = ModelId("mlx-community/Test-Model-4bit")
    instance_id = InstanceId("instance-one")
    api = _api_with_instances({instance_id: _instance(model_id, instance_id)})
    captured: dict[str, Any] = {}
    endpoint = api._model_endpoint_name(model_id)

    async def _capture_send(
        self: API,
        task_params: TextGenerationTaskParams,
        target_instance_id: InstanceId | None = None,
    ) -> TextGeneration:
        captured["model"] = task_params.model
        captured["target_instance_id"] = target_instance_id
        return TextGeneration(
            task_params=task_params, target_instance_id=target_instance_id
        )

    api._send_text_generation_with_images = MethodType(_capture_send, api)
    route = api._resolve_text_generation_route(
        ModelId("ignored-request-model"), endpoint, _RequestStub()  # type: ignore[arg-type]
    )

    command = await api._send_routed_text_generation(
        _text_params(ModelId("ignored-request-model")), route
    )

    assert captured == {"model": model_id, "target_instance_id": None}
    assert command.task_params.model == model_id
    assert command.target_instance_id is None


def test_unknown_agent_endpoint_returns_404_before_dispatch() -> None:
    api = _api_with_instances({})

    with pytest.raises(HTTPException) as exception_info:
        api._resolve_text_generation_route(
            ModelId("model"), "inst-deleted", _RequestStub()  # type: ignore[arg-type]
        )

    assert exception_info.value.status_code == 404
