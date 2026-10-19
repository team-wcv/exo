from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping
from typing import Final, override

import anyio
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

_DONE: Final = object()


class FinalizingStreamingResponse(StreamingResponse):
    """Release resources even when disconnect happens before stream iteration."""

    def __init__(
        self,
        content: AsyncIterable[str],
        *,
        finalize: Callable[[], Awaitable[None]],
        media_type: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(content, media_type=media_type, headers=headers)
        self._finalize = finalize

    @override
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self._finalize()


async def with_sse_keepalive(
    generator: AsyncIterator[str],
    keepalive_message: str = ": keep-alive\n\n",
    interval: float = 10.0,
) -> AsyncIterator[str]:
    yield keepalive_message
    send, recv = anyio.create_memory_object_stream[str | object]()

    async def _consume() -> None:
        async for item in generator:
            await send.send(item)
        await send.send(_DONE)

    async with anyio.create_task_group() as tg:
        tg.start_soon(_consume)
        while True:
            item: str | object | None = None
            with anyio.move_on_after(interval):
                item = await recv.receive()
            if item is None:
                yield keepalive_message
            elif item is _DONE:
                break
            else:
                assert isinstance(item, str)
                yield item
