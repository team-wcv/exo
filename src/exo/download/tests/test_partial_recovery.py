"""Recovery tests for resumable downloads and peer-streaming metadata."""

from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from unittest.mock import AsyncMock, patch

import aiofiles
import aiofiles.os as aios
import pytest

from exo.download.download_utils import (
    _download_file,  # pyright: ignore[reportPrivateUsage]
)
from exo.shared.types.common import ModelId


class _FakeResponseContent:
    def __init__(
        self,
        chunks: list[bytes],
        on_first_read: Callable[[], None] | None = None,
    ) -> None:
        self._chunks = [*chunks, b""]
        self._on_first_read = on_first_read

    async def read(self, _size: int) -> bytes:
        if self._on_first_read is not None:
            callback, self._on_first_read = self._on_first_read, None
            callback()
        return self._chunks.pop(0)


class _FakeResponse:
    def __init__(
        self,
        status: int,
        chunks: list[bytes],
        on_first_read: Callable[[], None] | None = None,
    ) -> None:
        self.status = status
        self.headers: dict[str, str] = {}
        self.content = _FakeResponseContent(chunks, on_first_read)

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None


class _FakeSession:
    def __init__(
        self,
        responses: list[_FakeResponse],
        on_request: Callable[[int], None] | None = None,
    ) -> None:
        self._responses = responses
        self._on_request = on_request
        self.requests: list[tuple[str, dict[str, str]]] = []

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def get(self, url: str, *, headers: dict[str, str]) -> _FakeResponse:
        self.requests.append((url, headers))
        if self._on_request is not None:
            self._on_request(len(self.requests))
        return self._responses.pop(0)


async def _write_stale_state(partial: Path, payload: bytes = b"old") -> Path:
    await aios.makedirs(partial.parent, exist_ok=True)
    async with aiofiles.open(partial, "wb") as f:
        await f.write(payload)
    metadata = Path(f"{partial}.meta")
    async with aiofiles.open(metadata, "w") as f:
        await f.write('{"safe_bytes":999}')
    return metadata


async def _run_download(
    tmp_path: Path,
    session: _FakeSession,
    *,
    remote_size: int = 7,
    remote_hash: str = "expected",
    progress: Callable[[int, int, bool], None] | None = None,
) -> Path:
    with (
        patch(
            "exo.download.download_utils.file_meta",
            new_callable=AsyncMock,
            return_value=(remote_size, remote_hash),
        ),
        patch(
            "exo.download.download_utils.create_http_session",
            return_value=session,
        ),
        patch(
            "exo.download.download_utils.calc_hash",
            new_callable=AsyncMock,
            return_value=remote_hash,
        ),
    ):
        return await _download_file(
            ModelId("test-org/test-model"),
            "main",
            "model.bin",
            tmp_path,
            progress or (lambda _current, _total, _done: None),
        )


@pytest.mark.asyncio
async def test_oversized_partial_discards_payload_and_metadata(tmp_path: Path) -> None:
    partial = tmp_path / "model.bin.partial"
    metadata = await _write_stale_state(partial, b"oversized")
    progress: list[tuple[int, int, bool]] = []
    session = _FakeSession([_FakeResponse(200, [b"newdata"])])

    def record_progress(current: int, total: int, done: bool) -> None:
        progress.append((current, total, done))

    result = await _run_download(
        tmp_path,
        session,
        progress=record_progress,
    )

    assert result.read_bytes() == b"newdata"
    assert not partial.exists()
    assert not metadata.exists()
    assert "Range" not in session.requests[0][1]
    assert progress[0] == (0, 7, False)


@pytest.mark.asyncio
async def test_http_416_retries_once_without_stale_peer_state(tmp_path: Path) -> None:
    partial = tmp_path / "model.bin.partial"
    metadata = await _write_stale_state(partial)
    state_at_second_request: list[tuple[bool, bool]] = []

    def observe_request(number: int) -> None:
        if number == 2:
            state_at_second_request.append((partial.exists(), metadata.exists()))

    session = _FakeSession(
        [_FakeResponse(416, []), _FakeResponse(200, [b"newdata"])],
        on_request=observe_request,
    )

    result = await _run_download(tmp_path, session)

    assert result.read_bytes() == b"newdata"
    assert session.requests[0][1]["Range"] == "bytes=3-"
    assert "Range" not in session.requests[1][1]
    assert state_at_second_request == [(False, False)]


@pytest.mark.asyncio
async def test_http_416_clean_retry_is_bounded(tmp_path: Path) -> None:
    partial = tmp_path / "model.bin.partial"
    metadata = await _write_stale_state(partial)
    session = _FakeSession([_FakeResponse(416, []), _FakeResponse(416, [])])

    with pytest.raises(AssertionError, match="416"):
        await _run_download(tmp_path, session)

    assert len(session.requests) == 2
    assert not partial.exists()
    assert not metadata.exists()


@pytest.mark.asyncio
async def test_full_response_to_range_replaces_partial_before_read(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "model.bin.partial"
    metadata = await _write_stale_state(partial)
    state_at_read: list[tuple[bool, bool]] = []
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                [b"newdata"],
                on_first_read=lambda: state_at_read.append(
                    (partial.exists(), metadata.exists())
                ),
            )
        ]
    )

    result = await _run_download(tmp_path, session)

    assert result.read_bytes() == b"newdata"
    assert session.requests[0][1]["Range"] == "bytes=3-"
    assert state_at_read == [(True, False)]


@pytest.mark.asyncio
async def test_hash_failure_removes_partial_and_peer_metadata(tmp_path: Path) -> None:
    partial = tmp_path / "model.bin.partial"
    metadata = await _write_stale_state(partial)
    session = _FakeSession([_FakeResponse(416, []), _FakeResponse(200, [b"newdata"])])

    with (
        patch(
            "exo.download.download_utils.file_meta",
            new_callable=AsyncMock,
            return_value=(7, "expected"),
        ),
        patch(
            "exo.download.download_utils.create_http_session",
            return_value=session,
        ),
        patch(
            "exo.download.download_utils.calc_hash",
            new_callable=AsyncMock,
            return_value="wrong",
        ),
        pytest.raises(Exception, match="remote hash is expected"),
    ):
        await _download_file(
            ModelId("test-org/test-model"), "main", "model.bin", tmp_path
        )

    assert not partial.exists()
    assert not metadata.exists()
