# pyright: reportUnusedFunction=false, reportAny=false, reportPrivateUsage=false
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect
from starlette.types import Message

from exo.api.keepalive import with_sse_keepalive
from exo.api.main import API
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.commands import (
    ForwarderCommand,
    TaskCancelled,
    TaskFinished,
    TextGeneration,
)
from exo.shared.types.common import ModelId, SystemId
from exo.shared.types.text_generation import TextGenerationTaskParams


def _streaming_api() -> API:
    api = object.__new__(API)
    api.paused = False
    api._system_id = SystemId()
    api._text_generation_queues = {}
    api._text_generation_receivers = {}
    api.command_sender = AsyncMock()
    return api


@pytest.mark.asyncio
async def test_immediate_error_is_buffered_before_stream_subscription() -> None:
    api = _streaming_api()
    command = TextGeneration(
        task_params=TextGenerationTaskParams(model=ModelId("lost-decoder"), input=[])
    )
    error = ErrorChunk(
        model=ModelId("lost-decoder"), error_message="No decode instance"
    )

    async def immediate_reply(forwarded: ForwarderCommand) -> None:
        if isinstance(forwarded.command, TextGeneration):
            await api._text_generation_queues[command.command_id].send(error)

    api.command_sender.send = AsyncMock(side_effect=immediate_reply)
    await api._send(command)
    assert [chunk async for chunk in api._token_chunk_stream(command.command_id)] == [
        error
    ]
    assert not api._text_generation_queues
    assert not api._text_generation_receivers


@pytest.mark.asyncio
async def test_failed_command_send_cleans_registered_stream() -> None:
    api = _streaming_api()
    api.command_sender.send = AsyncMock(side_effect=RuntimeError("router unavailable"))
    command = TextGeneration(
        task_params=TextGenerationTaskParams(model=ModelId("lost-decoder"), input=[])
    )
    with pytest.raises(RuntimeError, match="router unavailable"):
        await api._send(command)
    assert not api._text_generation_queues
    assert not api._text_generation_receivers


@pytest.mark.asyncio
async def test_disconnect_after_initial_keepalive_cleans_unstarted_stream() -> None:
    api = _streaming_api()
    sent: list[ForwarderCommand] = []

    async def record_send(command: ForwarderCommand) -> None:
        sent.append(command)

    api.command_sender.send = AsyncMock(side_effect=record_send)
    command = TextGeneration(
        task_params=TextGenerationTaskParams(model=ModelId("test-model"), input=[])
    )
    await api._send(command)

    async def body():
        async for _chunk in api._token_chunk_stream(command.command_id):
            yield "data: token\n\n"

    response = api._text_response(
        command.command_id, with_sse_keepalive(body()), media_type="text/event-stream"
    )

    async def send(message: Message) -> None:
        if message["type"] == "http.response.body":
            assert message["body"] == b": keep-alive\n\n"
            raise OSError("client disconnected")

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    with pytest.raises(ClientDisconnect):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert not api._text_generation_queues
    assert not api._text_generation_receivers
    assert isinstance(sent[-2].command, TaskCancelled)
    assert isinstance(sent[-1].command, TaskFinished)


def test_http_exception_handler_formats_openai_style() -> None:
    """Test that HTTPException is converted to OpenAI-style error format."""

    app = FastAPI()

    # Setup exception handler
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()

    # Add test routes that raise HTTPException
    @app.get("/test-error")
    async def _test_error() -> None:
        raise HTTPException(status_code=500, detail="Test error message")

    @app.get("/test-not-found")
    async def _test_not_found() -> None:
        raise HTTPException(status_code=404, detail="Resource not found")

    client = TestClient(app)

    # Test 500 error
    response = client.get("/test-error")
    assert response.status_code == 500
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Test error message"
    assert data["error"]["type"] == "Internal Server Error"
    assert data["error"]["code"] == 500

    # Test 404 error
    response = client.get("/test-not-found")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Resource not found"
    assert data["error"]["type"] == "Not Found"
    assert data["error"]["code"] == 404
