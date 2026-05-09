"""Tests for peer-to-peer model downloading."""
# pyright: reportPrivateUsage=false

import json
from collections.abc import AsyncIterator, Generator, Iterable
from pathlib import Path
from typing import Callable, cast

import aiofiles
import aiofiles.os as aios
import aiohttp
import pytest

from exo.download.peer_download import download_file_from_peer, get_peer_file_status
from exo.download.peer_file_server import PeerFileServer
from exo.download.peer_shard_downloader import PeerAwareShardDownloader
from exo.download.shard_downloader import NoopShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.commands import PeerEndpoint
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata


@pytest.fixture
async def temp_models_dir(tmp_path: Path) -> AsyncIterator[Path]:
    """Set up a temporary models directory for testing."""
    models_dir = tmp_path / "models"
    await aios.makedirs(models_dir, exist_ok=True)
    yield models_dir


@pytest.fixture
async def peer_server(temp_models_dir: Path) -> AsyncIterator[PeerFileServer]:
    """Start a PeerFileServer on a random port for testing."""
    server = PeerFileServer(host="127.0.0.1", port=0, models_dirs=[temp_models_dir])
    # Use port 0 to let OS assign a free port
    from aiohttp import web

    server._runner = web.AppRunner(server._app)
    await server._runner.setup()
    site = web.TCPSite(server._runner, "127.0.0.1", 0)
    await site.start()
    # Get the actual port assigned
    server.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    yield server
    await server.shutdown()


def _make_shard(model_id: ModelId) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_mb(100),
            n_layers=28,
            hidden_size=1024,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=28,
        n_layers=28,
    )


class TestPeerFileServer:
    """Tests for the HTTP file server that serves model files to peers."""

    async def test_health_check(self, peer_server: PeerFileServer) -> None:
        """Health endpoint should return ok."""
        import aiohttp

        async with (
            aiohttp.ClientSession() as session,
            session.get(f"http://127.0.0.1:{peer_server.port}/health") as r,
        ):
            assert r.status == 200
            data = cast(dict[str, object], await r.json())
            assert data["status"] == "ok"

    async def test_status_empty_model(self, peer_server: PeerFileServer) -> None:
        """Status for non-existent model should return empty file list."""
        files = await get_peer_file_status(
            "127.0.0.1", peer_server.port, "nonexistent--model"
        )
        assert files is not None
        assert len(files) == 0

    async def test_status_with_complete_file(
        self, peer_server: PeerFileServer, temp_models_dir: Path
    ) -> None:
        """Status should report complete files correctly."""
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)

        # Create a complete test file
        async with aiofiles.open(model_dir / "config.json", "wb") as f:
            await f.write(b'{"test": true}')

        files = await get_peer_file_status("127.0.0.1", peer_server.port, "test--model")
        assert files is not None
        assert len(files) == 1
        assert files[0].path == "config.json"
        assert files[0].complete is True
        assert files[0].safe_bytes == 14

    async def test_status_with_partial_file(
        self, peer_server: PeerFileServer, temp_models_dir: Path
    ) -> None:
        """Status should report partial files with safe byte count."""
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)

        # Create a partial file with metadata
        partial_data = b"x" * 1024
        async with aiofiles.open(model_dir / "weights.safetensors.partial", "wb") as f:
            await f.write(partial_data)

        meta = {"safe_bytes": 1024, "total": 4096, "etag": "abc123"}
        async with aiofiles.open(
            model_dir / "weights.safetensors.partial.meta", "w"
        ) as f:
            await f.write(json.dumps(meta))

        files = await get_peer_file_status("127.0.0.1", peer_server.port, "test--model")
        assert files is not None
        assert len(files) == 1
        assert files[0].path == "weights.safetensors"
        assert files[0].complete is False
        assert files[0].safe_bytes == 1024
        assert files[0].size == 4096

    async def test_status_includes_nested_files(
        self, peer_server: PeerFileServer, temp_models_dir: Path
    ) -> None:
        """Status should report nested complete and partial files."""
        model_dir = temp_models_dir / "test--model"
        nested_dir = model_dir / "snapshots" / "abc123"
        await aios.makedirs(nested_dir, exist_ok=True)

        async with aiofiles.open(nested_dir / "config.json", "wb") as f:
            await f.write(b"{}")
        async with aiofiles.open(nested_dir / "model.safetensors.partial", "wb") as f:
            await f.write(b"x" * 512)
        async with aiofiles.open(
            nested_dir / "model.safetensors.partial.meta", "w"
        ) as f:
            await f.write(json.dumps({"safe_bytes": 512, "total": 2048}))

        files = await get_peer_file_status("127.0.0.1", peer_server.port, "test--model")
        assert files is not None
        by_path = {file.path: file for file in files}
        assert by_path["snapshots/abc123/config.json"].complete is True
        assert by_path["snapshots/abc123/model.safetensors"].complete is False
        assert by_path["snapshots/abc123/model.safetensors"].safe_bytes == 512

    async def test_serve_complete_file(
        self, peer_server: PeerFileServer, temp_models_dir: Path
    ) -> None:
        """Should serve a complete file with correct headers."""
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)

        content = b"hello world test content"
        async with aiofiles.open(model_dir / "config.json", "wb") as f:
            await f.write(content)

        import aiohttp

        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"http://127.0.0.1:{peer_server.port}/files/test--model/config.json"
            ) as r,
        ):
            assert r.status == 200
            assert r.headers["X-Exo-Complete"] == "true"
            body = await r.read()
            assert body == content

    async def test_serve_nested_file(
        self, peer_server: PeerFileServer, temp_models_dir: Path
    ) -> None:
        """Should serve a complete nested file with correct headers."""
        model_dir = temp_models_dir / "test--model"
        nested_dir = model_dir / "snapshots" / "abc123"
        await aios.makedirs(nested_dir, exist_ok=True)

        content = b"nested content"
        async with aiofiles.open(nested_dir / "config.json", "wb") as f:
            await f.write(content)

        import aiohttp

        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"http://127.0.0.1:{peer_server.port}/files/test--model/"
                "snapshots/abc123/config.json"
            ) as r,
        ):
            assert r.status == 200
            body = await r.read()
            assert body == content

    async def test_rejects_path_traversal(
        self, peer_server: PeerFileServer, temp_models_dir: Path
    ) -> None:
        """Should not serve files outside the requested model directory."""
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)

        outside_file = temp_models_dir / "outside.txt"
        async with aiofiles.open(outside_file, "wb") as f:
            await f.write(b"outside")

        import aiohttp

        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"http://127.0.0.1:{peer_server.port}/files/test--model/"
                "%2E%2E/outside.txt"
            ) as r,
        ):
            assert r.status == 404
            assert await r.text() != "outside"

    async def test_serve_with_range_request(
        self, peer_server: PeerFileServer, temp_models_dir: Path
    ) -> None:
        """Should support Range requests for resume."""
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)

        content = b"0123456789abcdef"
        async with aiofiles.open(model_dir / "weights.bin", "wb") as f:
            await f.write(content)

        import aiohttp

        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"http://127.0.0.1:{peer_server.port}/files/test--model/weights.bin",
                headers={"Range": "bytes=8-"},
            ) as r,
        ):
            assert r.status == 206
            body = await r.read()
            assert body == b"89abcdef"

    async def test_file_not_found(self, peer_server: PeerFileServer) -> None:
        """Should return 404 for missing files."""
        import aiohttp

        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"http://127.0.0.1:{peer_server.port}/files/test--model/missing.bin"
            ) as r,
        ):
            assert r.status == 404


class TestPeerFileServerMultipleDirectories:
    """The peer file server must look for the model in *every* configured
    models directory. Otherwise a node that lands a model in a non-default
    writable directory (custom path, low-disk fallback, or read-only mount)
    would silently fail to advertise it to peers and force them back onto
    HuggingFace -- defeating the whole peer download path.
    """

    async def test_serves_model_from_secondary_writable_dir(
        self, tmp_path: Path
    ) -> None:
        primary = tmp_path / "primary"
        secondary = tmp_path / "secondary"
        await aios.makedirs(primary, exist_ok=True)
        await aios.makedirs(secondary, exist_ok=True)

        model_dir = secondary / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)
        async with aiofiles.open(model_dir / "config.json", "wb") as f:
            await f.write(b'{"hello":"world"}')

        server = PeerFileServer(
            host="127.0.0.1", port=0, models_dirs=[primary, secondary]
        )

        from aiohttp import web

        server._runner = web.AppRunner(server._app)
        await server._runner.setup()
        site = web.TCPSite(server._runner, "127.0.0.1", 0)
        await site.start()
        port_int: int = cast(int, site._server.sockets[0].getsockname()[1])  # type: ignore[union-attr]
        server.port = port_int
        try:
            files = await get_peer_file_status("127.0.0.1", port_int, "test--model")
            assert files is not None
            assert {f.path for f in files} == {"config.json"}
        finally:
            await server.shutdown()

    async def test_serves_model_from_read_only_mount(self, tmp_path: Path) -> None:
        writable = tmp_path / "writable"
        read_only = tmp_path / "ro_mount"
        await aios.makedirs(writable, exist_ok=True)
        await aios.makedirs(read_only / "ro--model", exist_ok=True)
        async with aiofiles.open(read_only / "ro--model" / "config.json", "wb") as f:
            await f.write(b"{}")

        server = PeerFileServer(
            host="127.0.0.1", port=0, models_dirs=[writable, read_only]
        )

        from aiohttp import web

        server._runner = web.AppRunner(server._app)
        await server._runner.setup()
        site = web.TCPSite(server._runner, "127.0.0.1", 0)
        await site.start()
        port_int: int = cast(int, site._server.sockets[0].getsockname()[1])  # type: ignore[union-attr]
        server.port = port_int
        try:
            files = await get_peer_file_status("127.0.0.1", port_int, "ro--model")
            assert files is not None
            assert {f.path for f in files} == {"config.json"}
        finally:
            await server.shutdown()

    async def test_constructor_rejects_empty_directory_list(self) -> None:
        with pytest.raises(ValueError, match="at least one models directory"):
            PeerFileServer(host="127.0.0.1", port=0, models_dirs=[])

    async def test_status_unions_partial_in_first_root_with_complete_in_second(
        self, tmp_path: Path
    ) -> None:
        """Codex P2 (PR #16 round-(N+9), peer_file_server.py:201): if
        an earlier root has a stale/incomplete model directory and a
        later root has a complete copy, ``/status`` must surface the
        complete file -- otherwise peers see the file as missing and
        fall back to HuggingFace despite the local node having a
        canonical copy on a different mount.
        """
        from aiohttp import web

        first = tmp_path / "first"
        second = tmp_path / "second"
        await aios.makedirs(first / "test--model", exist_ok=True)
        await aios.makedirs(second / "test--model", exist_ok=True)

        # First root has only a partial of weights.bin (incomplete).
        partial_path = first / "test--model" / "weights.bin.partial"
        canonical = b"the canonical model weights"
        async with aiofiles.open(partial_path, "wb") as f:
            await f.write(canonical[: len(canonical) // 2])
        # Companion meta marking 50% safe.
        meta_path = first / "test--model" / "weights.bin.partial.meta"
        async with aiofiles.open(meta_path, "w") as f:
            await f.write(
                json.dumps(
                    {
                        "total": len(canonical),
                        "safe_bytes": len(canonical) // 2,
                    }
                )
            )

        # Second root has the full canonical file (complete).
        async with aiofiles.open(second / "test--model" / "weights.bin", "wb") as f:
            await f.write(canonical)

        server = PeerFileServer(
            host="127.0.0.1", port=0, models_dirs=[first, second]
        )
        server._runner = web.AppRunner(server._app)
        await server._runner.setup()
        site = web.TCPSite(server._runner, "127.0.0.1", 0)
        await site.start()
        port_int: int = cast(int, site._server.sockets[0].getsockname()[1])  # type: ignore[union-attr]
        server.port = port_int
        try:
            files = await get_peer_file_status("127.0.0.1", port_int, "test--model")
            assert files is not None
            file_map = {f.path: f for f in files}
            assert "weights.bin" in file_map, (
                "complete copy in the second root must surface in /status; "
                "got files={file_map.keys()}"
            )
            assert file_map["weights.bin"].complete is True, (
                "complete copy in the second root must dominate the "
                "partial in the first root; otherwise peers will fall "
                "back to HuggingFace"
            )
            assert file_map["weights.bin"].size == len(canonical)
        finally:
            await server.shutdown()

    async def test_files_serves_complete_copy_when_first_root_has_only_partial(
        self, tmp_path: Path
    ) -> None:
        """End-to-end: ``/files/<path>`` must select the root holding
        the complete file even when an earlier root has only a
        partial. Pre-fix the server returned 404 (or served the
        smaller partial via the partial-bytes path) when a complete
        file lived in a later root, forcing peers to fall back to
        HuggingFace.
        """
        from aiohttp import web

        first = tmp_path / "first"
        second = tmp_path / "second"
        await aios.makedirs(first / "test--model", exist_ok=True)
        await aios.makedirs(second / "test--model", exist_ok=True)

        canonical = b"complete-canonical-bytes"
        # First root has partial (with valid meta).
        partial_path = first / "test--model" / "weights.bin.partial"
        async with aiofiles.open(partial_path, "wb") as f:
            await f.write(canonical[: len(canonical) // 2])
        meta_path = first / "test--model" / "weights.bin.partial.meta"
        async with aiofiles.open(meta_path, "w") as f:
            await f.write(
                json.dumps(
                    {
                        "total": len(canonical),
                        "safe_bytes": len(canonical) // 2,
                    }
                )
            )
        # Second root has the complete file.
        async with aiofiles.open(second / "test--model" / "weights.bin", "wb") as f:
            await f.write(canonical)

        server = PeerFileServer(
            host="127.0.0.1", port=0, models_dirs=[first, second]
        )
        server._runner = web.AppRunner(server._app)
        await server._runner.setup()
        site = web.TCPSite(server._runner, "127.0.0.1", 0)
        await site.start()
        port_int: int = cast(int, site._server.sockets[0].getsockname()[1])  # type: ignore[union-attr]
        server.port = port_int
        try:
            url = f"http://127.0.0.1:{port_int}/files/test--model/weights.bin"
            async with (
                aiohttp.ClientSession() as session,
                session.get(url) as r,
            ):
                assert r.status == 200, (
                    f"expected 200 from /files when complete copy exists in "
                    f"a later root; got {r.status}"
                )
                body = await r.read()
            assert body == canonical, (
                f"expected canonical bytes from later root; got "
                f"{len(body)} bytes (expected {len(canonical)})"
            )
            # Sanity: X-Exo-Complete header should mark this as a
            # complete serving (not a partial-bytes fragment).
            assert r.headers.get("X-Exo-Complete") == "true"
        finally:
            await server.shutdown()


class TestPeerDownloadClient:
    """Tests for downloading files from a peer server."""

    async def test_download_complete_file(
        self, peer_server: PeerFileServer, temp_models_dir: Path, tmp_path: Path
    ) -> None:
        """Should download a complete file from peer."""
        # Set up source file on the peer server
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)

        content = b"model weights data " * 100
        async with aiofiles.open(model_dir / "weights.bin", "wb") as f:
            await f.write(content)

        # Download to a different directory
        download_dir = tmp_path / "downloads" / "test--model"
        await aios.makedirs(download_dir, exist_ok=True)

        progress_calls: list[tuple[int, int, bool]] = []

        result = await download_file_from_peer(
            "127.0.0.1",
            peer_server.port,
            "test--model",
            "weights.bin",
            download_dir,
            len(content),
            on_progress=lambda c, t, r: progress_calls.append((c, t, r)),
        )

        assert result is not None
        assert result == download_dir / "weights.bin"
        async with aiofiles.open(result, "rb") as f:
            downloaded = await f.read()
        assert downloaded == content
        # Should have progress calls including final
        assert len(progress_calls) > 0
        assert progress_calls[-1][2] is True  # is_renamed

    async def test_download_returns_none_on_missing(
        self, peer_server: PeerFileServer, tmp_path: Path
    ) -> None:
        """Should return None when file doesn't exist on peer."""
        download_dir = tmp_path / "downloads" / "test--model"
        await aios.makedirs(download_dir, exist_ok=True)

        result = await download_file_from_peer(
            "127.0.0.1",
            peer_server.port,
            "test--model",
            "nonexistent.bin",
            download_dir,
            1000,
        )
        assert result is None

    async def test_download_returns_none_on_unreachable_peer(
        self, tmp_path: Path
    ) -> None:
        """Should return None when peer is unreachable."""
        download_dir = tmp_path / "downloads" / "test--model"
        await aios.makedirs(download_dir, exist_ok=True)

        result = await download_file_from_peer(
            "127.0.0.1",
            19999,  # Nobody listening
            "test--model",
            "weights.bin",
            download_dir,
            1000,
        )
        assert result is None

    async def test_oversized_stale_partial_is_discarded_and_retransferred(
        self, peer_server: PeerFileServer, temp_models_dir: Path, tmp_path: Path
    ) -> None:
        """Codex P1 (PR #16 round 5): a stale ``.partial`` larger than
        ``expected_size`` left over from a previous run must be
        rejected, NOT silently renamed as the successful download.

        Pre-fix the resume loop ran ``while n_read < expected_size``,
        so an oversized partial skipped the loop entirely and the
        final ``rename`` accepted bad bytes. In offline mode (where
        hash verification is intentionally skipped) this would
        permanently poison the model cache without any warning.
        Post-fix the oversized partial is discarded and the file is
        re-fetched from the peer.
        """
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)
        canonical = b"the canonical model weights"
        async with aiofiles.open(model_dir / "weights.bin", "wb") as f:
            await f.write(canonical)

        download_dir = tmp_path / "downloads" / "test--model"
        await aios.makedirs(download_dir, exist_ok=True)
        # Stale partial from a "previous run" -- bigger than the
        # canonical file and full of junk bytes. Pre-fix, this would
        # be the file that ended up renamed as ``weights.bin``.
        stale_partial = download_dir / "weights.bin.partial"
        stale_bytes = b"\xde\xad\xbe\xef" * (len(canonical) * 2)
        async with aiofiles.open(stale_partial, "wb") as f:
            await f.write(stale_bytes)
        assert (await aios.stat(stale_partial)).st_size > len(canonical)

        result = await download_file_from_peer(
            "127.0.0.1",
            peer_server.port,
            "test--model",
            "weights.bin",
            download_dir,
            len(canonical),
        )

        assert result is not None
        assert result == download_dir / "weights.bin"
        async with aiofiles.open(result, "rb") as f:
            downloaded = await f.read()
        assert downloaded == canonical, (
            "stale oversized partial must NOT be accepted as the "
            "downloaded file; the fix must redownload from the peer"
        )
        assert not stale_partial.exists()

    async def test_resume_with_200_response_discards_partial_and_restarts(
        self, tmp_path: Path
    ) -> None:
        """Codex P1 (PR #16 round-(N+3), peer_download.py:162): when
        the client resumes a download (``n_read > 0``) it sends a
        ``Range`` header, but a non-compliant server is permitted to
        ignore it and return full content with HTTP 200 instead of
        206. Pre-fix the client appended the full body to the
        partial, pushing ``n_read`` past ``expected_size`` and
        renaming the oversized file as the "successful" download.
        In offline mode hash verification is intentionally skipped,
        so the bad bytes silently poisoned the model cache.

        We stand up a tiny aiohttp server that returns full content
        with 200 even when ``Range`` is set, prime a partial file,
        and assert the client discards the partial, restarts from
        zero, and lands the canonical bytes (matching ``expected_size``).
        """
        from aiohttp import web

        canonical = b"the canonical model weights"

        async def handler(request: web.Request) -> web.Response:
            # Always return full content with HTTP 200, ignoring any
            # ``Range`` header. This simulates the non-compliant
            # peer server the codex finding flagged.
            del request
            return web.Response(body=canonical, status=200)

        app = web.Application()
        # Path must match the client's URL template:
        # ``http://host:port/files/<model_id>/<file_path>``
        _ = app.router.add_get("/files/test/weights.bin", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            # Mirror the ``peer_server`` fixture: ``aiohttp.web.TCPSite``
            # surfaces the kernel-assigned port through its private
            # ``_server.sockets`` attribute. The module-level
            # ``reportPrivateUsage=false`` and ``type: ignore`` here
            # match the existing fixture's access pattern.
            port: int = cast(
                int,
                site._server.sockets[0].getsockname()[1],  # type: ignore[union-attr]
            )

            download_dir = tmp_path / "downloads" / "test"
            await aios.makedirs(download_dir, exist_ok=True)
            # Prime a stale partial with bogus content to force the
            # resume codepath (Range header) on the first attempt.
            partial_path = download_dir / "weights.bin.partial"
            stale_prefix = b"\xff" * (len(canonical) // 2)
            async with aiofiles.open(partial_path, "wb") as f:
                await f.write(stale_prefix)
            assert (await aios.stat(partial_path)).st_size > 0

            result = await download_file_from_peer(
                "127.0.0.1",
                port,
                "test",
                "weights.bin",
                download_dir,
                len(canonical),
            )

            assert result is not None, (
                "the client should ultimately succeed by discarding the "
                "stale partial and restarting from zero on the second "
                "request"
            )
            assert result == download_dir / "weights.bin"
            async with aiofiles.open(result, "rb") as f:
                downloaded = await f.read()
            assert downloaded == canonical, (
                "200-on-resume must trigger a partial restart; the final "
                "file must be the canonical bytes, not a duplicate-prefix "
                "concatenation"
            )
            assert not partial_path.exists(), (
                "successful download must remove the partial path"
            )
        finally:
            await runner.cleanup()

    async def test_oversized_peer_response_is_rejected_and_restarted(
        self, tmp_path: Path
    ) -> None:
        """Codex P1 (PR #16 round-(N+8), peer_download.py:187): the
        download loop used to keep appending bytes until EOF and only
        check ``n_read < expected_size`` afterwards. A non-compliant
        peer that serves *more* bytes than the advertised
        ``expected_size`` would push ``n_read`` past it, the file
        would be renamed as a successful download, and -- because
        offline mode skips hash verification -- silently poison the
        model cache.

        We stand up a tiny aiohttp server that always returns
        ``len(canonical) + 8`` bytes regardless of how much was
        requested. Pre-fix this would land a corrupt file in the
        cache. Post-fix the client must discard each oversized
        response and never end up with a final file containing extra
        bytes."""
        from aiohttp import web

        canonical = b"the canonical model weights"
        # The payload the bad peer always serves: the canonical
        # bytes plus extra trailing bytes the peer claimed wouldn't
        # exist. This is the attack/bug the fix guards against.
        oversized_payload = canonical + b"POISONED"
        request_count = 0
        max_requests = 4  # keep test fast: client retries a few times

        async def handler(request: web.Request) -> web.Response:
            nonlocal request_count
            request_count += 1
            del request
            if request_count > max_requests:
                # Surface a definitive failure if the client keeps
                # hammering the bad peer; that means the fix
                # regressed and we'd otherwise hang.
                return web.Response(body=b"", status=500)
            return web.Response(body=oversized_payload, status=200)

        app = web.Application()
        _ = app.router.add_get("/files/test/weights.bin", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            port: int = cast(
                int,
                site._server.sockets[0].getsockname()[1],  # type: ignore[union-attr]
            )

            download_dir = tmp_path / "downloads" / "test"
            await aios.makedirs(download_dir, exist_ok=True)

            result = await download_file_from_peer(
                "127.0.0.1",
                port,
                "test",
                "weights.bin",
                download_dir,
                len(canonical),
            )

            # The bad peer never serves a well-bounded response, so
            # the client cannot complete. The contract is "no
            # corrupt data lands in the cache". We tolerate either
            # outcome:
            #   1. ``result is None`` (client gave up after retries); or
            #   2. ``result == canonical`` (a future improvement
            #      where we keep the canonical-prefix bytes after
            #      stripping the over-supply).
            # The forbidden outcome is the final file containing
            # the trailing "POISONED" bytes.
            partial_path = download_dir / "weights.bin.partial"
            target_path = download_dir / "weights.bin"

            if result is not None:
                async with aiofiles.open(result, "rb") as f:
                    downloaded = await f.read()
                assert downloaded == canonical, (
                    "if the client claims success, the final file MUST "
                    "be exactly the canonical bytes; oversized peer "
                    "responses must never land trailing junk in the "
                    f"cache. got len={len(downloaded)} bytes: {downloaded!r}"
                )
            # In the giving-up branch, neither file should remain
            # poisoned. The partial is removed every time we detect
            # over-supply, and we never rename to ``target_path``
            # without a clean-budgeted final write.
            if target_path.exists():
                async with aiofiles.open(target_path, "rb") as f:
                    final = await f.read()
                assert final == canonical, (
                    f"target path was renamed but contains "
                    f"{len(final)} bytes (expected {len(canonical)}); "
                    "oversized response made it into the cache"
                )
            if partial_path.exists():
                size = (await aios.stat(partial_path)).st_size
                assert size <= len(canonical), (
                    f"partial path retains {size} bytes after "
                    f"oversized response (expected <= {len(canonical)}); "
                    "over-supply must be discarded, not preserved"
                )
        finally:
            await runner.cleanup()

    async def test_skip_already_complete(
        self, peer_server: PeerFileServer, temp_models_dir: Path, tmp_path: Path
    ) -> None:
        """Should skip download if file already exists locally with correct size."""
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)

        content = b"existing content"
        # File already exists in target
        download_dir = tmp_path / "downloads" / "test--model"
        await aios.makedirs(download_dir, exist_ok=True)
        async with aiofiles.open(download_dir / "config.json", "wb") as f:
            await f.write(content)

        result = await download_file_from_peer(
            "127.0.0.1",
            peer_server.port,
            "test--model",
            "config.json",
            download_dir,
            len(content),
        )

        assert result is not None
        assert result == download_dir / "config.json"


class TestPeerAwareShardDownloader:
    """Tests for peer selection handoff into peer-aware downloads."""

    def test_peers_are_queued_per_shard(self) -> None:
        """Concurrent downloads should not overwrite each other's peer list."""
        downloader = PeerAwareShardDownloader(NoopShardDownloader())
        shard_a = _make_shard(ModelId("test-org/model-a"))
        shard_b = _make_shard(ModelId("test-org/model-b"))
        peer_a = PeerEndpoint(
            node_id=NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            ip="10.0.0.1",
            port=52415,
        )
        peer_b = PeerEndpoint(
            node_id=NodeId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
            ip="10.0.0.2",
            port=52415,
        )

        downloader.set_available_peers(shard_a, [peer_a])
        downloader.set_available_peers(shard_b, [peer_b])

        assert downloader._pop_available_peers(shard_b) == [peer_b]
        assert downloader._pop_available_peers(shard_a) == [peer_a]
        assert downloader._pop_available_peers(shard_a) == []

    def test_peers_for_same_shard_are_not_overwritten(self) -> None:
        """Repeated commands for one shard should be consumed FIFO."""
        downloader = PeerAwareShardDownloader(NoopShardDownloader())
        shard = _make_shard(ModelId("test-org/model-a"))
        peer_a = PeerEndpoint(
            node_id=NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            ip="10.0.0.1",
            port=52415,
        )
        peer_b = PeerEndpoint(
            node_id=NodeId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
            ip="10.0.0.2",
            port=52415,
        )

        downloader.set_available_peers(shard, [peer_a])
        downloader.set_available_peers(shard, [peer_b])

        assert downloader._pop_available_peers(shard) == [peer_a]
        assert downloader._pop_available_peers(shard) == [peer_b]
        assert downloader._pop_available_peers(shard) == []


class TestPeerSelectionRespectsOfflineAndIgnorePatterns:
    """Codex P1s on PR #16 round 2: peer selection must mirror
    ``download_shard``'s logic exactly (``ignore_patterns`` for
    ``original/*`` / ``metal/*``) and must propagate the coordinator's
    offline mode into ``fetch_file_list_with_cache`` so a cold offline
    node can still complete a peer download without reaching out to
    HuggingFace for the initial file list.
    """

    def test_offline_flag_defaults_to_false(self) -> None:
        downloader = PeerAwareShardDownloader(NoopShardDownloader())
        assert downloader._offline is False

    def test_offline_flag_propagates(self) -> None:
        downloader = PeerAwareShardDownloader(NoopShardDownloader(), offline=True)
        assert downloader._offline is True

    async def test_try_peer_download_passes_offline_to_fetch_file_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_try_peer_download`` must thread ``self._offline`` into
        ``fetch_file_list_with_cache`` instead of always passing
        ``skip_internet=False``. We capture the kwargs by patching
        the import binding inside ``peer_shard_downloader``.
        """
        from exo.download import peer_shard_downloader as psd
        from exo.download.peer_download import PeerFileInfo
        from exo.shared.types.worker.downloads import FileListEntry

        captured: dict[str, object] = {}

        async def fake_fetch(*args: object, **kwargs: object) -> list[FileListEntry]:
            captured["args"] = args
            captured["kwargs"] = kwargs
            # Empty list -> no required files -> ``failed`` short-
            # circuit -> we get out cleanly with the call kwargs
            # captured.
            return []

        async def fake_peer_status(
            peer_host: str,
            peer_port: int,
            model_id_normalized: str,
            timeout: float = 5.0,
        ) -> list[PeerFileInfo] | None:
            return [
                PeerFileInfo(
                    path="model-00001-of-00002.safetensors",
                    size=10,
                    complete=True,
                    safe_bytes=10,
                )
            ]

        async def fake_resolve_dir(model_id: ModelId) -> Path:
            return Path("/tmp/fake-model")

        async def fake_resolve_allow(shard: ShardMetadata) -> list[str]:
            return ["*.safetensors"]

        monkeypatch.setattr(psd, "fetch_file_list_with_cache", fake_fetch)
        monkeypatch.setattr(psd, "get_peer_file_status", fake_peer_status)
        monkeypatch.setattr(psd, "resolve_model_dir", fake_resolve_dir)
        monkeypatch.setattr(psd, "resolve_allow_patterns", fake_resolve_allow)

        downloader = PeerAwareShardDownloader(NoopShardDownloader(), offline=True)
        shard = _make_shard(ModelId("test-org/model-a"))

        result = await downloader._try_peer_download(
            shard,
            peer_ip="10.0.0.1",
            peer_port=52415,
            model_id_normalized="test-org/model-a",
        )
        # Empty file list short-circuits to ``failed`` path and returns
        # None, but that's beside the point -- we just need the kwargs.
        assert result is None
        assert captured["kwargs"] == {
            "recursive": True,
            "skip_internet": True,
        }, f"skip_internet must reflect downloader.offline (got {captured['kwargs']!r})"

    async def test_try_peer_download_filters_ignore_patterns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Files under ``original/*`` and ``metal/*`` are excluded by
        ``download_shard``; the peer path must skip them too. Pre-fix
        the peer path filtered only ``allow_patterns``, leaving these
        in the required-files list. The peer doesn't have them
        locally (HF never downloads them), the strict
        ``peer_info missing => fail`` check fired, and every download
        fell back to HuggingFace.
        """
        from exo.download import peer_shard_downloader as psd
        from exo.download.peer_download import PeerFileInfo
        from exo.shared.types.worker.downloads import FileListEntry

        served = [
            FileListEntry(
                type="file",
                path="model-00001-of-00002.safetensors",
                size=100,
            ),
            FileListEntry(type="file", path="config.json", size=10),
            # These two should NOT show up on the peer's required-files
            # list once the fix lands. Pre-fix they did, the peer didn't
            # have them, and the whole transfer fell back to HF.
            FileListEntry(type="file", path="original/consolidated.00.pth", size=999),
            FileListEntry(type="file", path="metal/dist.bin", size=999),
        ]

        async def fake_fetch(*_args: object, **_kwargs: object) -> list[FileListEntry]:
            return served

        # The peer reports ONLY the canonical files, exactly the shape
        # production peers are in (HF never downloaded ``original/*`` or
        # ``metal/*`` for them either).
        peer_paths = ("model-00001-of-00002.safetensors", "config.json")

        async def fake_peer_status(
            peer_host: str,
            peer_port: int,
            model_id_normalized: str,
            timeout: float = 5.0,
        ) -> list[PeerFileInfo] | None:
            return [
                PeerFileInfo(path=p, size=100, complete=True, safe_bytes=100)
                for p in peer_paths
            ]

        async def fake_resolve_dir(model_id: ModelId) -> Path:
            return Path("/tmp/fake-model")

        async def fake_resolve_allow(shard: ShardMetadata) -> list[str]:
            # Match the production allow set permissively; the legacy
            # bug was that ``allow_patterns`` admitted ``original/*`` /
            # ``metal/*`` whenever the repo allow-list was loose.
            return ["*"]

        async def fake_download(
            peer_ip: str,
            peer_port: int,
            model_id_normalized: str,
            file_path: str,
            target_dir: Path,
            expected_size: int,
            on_progress: object = None,
        ) -> Path | None:
            return None

        captured_kwargs: list[object] = []
        real_filter = psd.filter_repo_objects

        def recording_filter(
            items: Iterable[FileListEntry],
            *,
            allow_patterns: list[str] | str | None = None,
            ignore_patterns: list[str] | str | None = None,
            key: Callable[[FileListEntry], str] | None = None,
        ) -> Generator[FileListEntry, None, None]:
            captured_kwargs.append(ignore_patterns)
            yield from real_filter(
                items,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                key=key,
            )

        monkeypatch.setattr(psd, "fetch_file_list_with_cache", fake_fetch)
        monkeypatch.setattr(psd, "get_peer_file_status", fake_peer_status)
        monkeypatch.setattr(psd, "resolve_model_dir", fake_resolve_dir)
        monkeypatch.setattr(psd, "resolve_allow_patterns", fake_resolve_allow)
        monkeypatch.setattr(psd, "download_file_from_peer", fake_download)
        monkeypatch.setattr(psd, "filter_repo_objects", recording_filter)

        downloader = PeerAwareShardDownloader(NoopShardDownloader())
        shard = _make_shard(ModelId("test-org/model-a"))

        await downloader._try_peer_download(
            shard,
            peer_ip="10.0.0.1",
            peer_port=52415,
            model_id_normalized="test-org/model-a",
        )

        assert captured_kwargs == [["original/*", "metal/*"]], (
            "peer download must apply the same ``ignore_patterns`` set "
            "as ``download_shard`` (download_utils.py:983) so peers "
            "that don't have ``original/*`` / ``metal/*`` aren't "
            "incorrectly judged incomplete; got "
            f"{captured_kwargs!r}"
        )


class TestPeerDownloadIntegrityCheckRespectsOfflineMode:
    """Codex P1 on PR #16 round 3: ``_try_peer_download`` was calling
    ``file_meta(...)`` against HuggingFace for every file, even when the
    coordinator was started with ``--offline`` / ``EXO_OFFLINE=true``.
    Any failure to reach HF (the entire point of offline mode) was
    treated as an integrity-check failure, the peer-fetched bytes were
    deleted, and the cold node was left with no path to complete model
    sync. The fix: when the downloader is in offline mode, trust the
    LAN peer's bytes and skip the HF metadata call entirely.
    """

    async def test_offline_mode_skips_file_meta_and_keeps_peer_bytes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from exo.download import peer_shard_downloader as psd
        from exo.download.peer_download import PeerFileInfo
        from exo.shared.types.worker.downloads import FileListEntry

        async def fake_fetch(*_args: object, **_kwargs: object) -> list[FileListEntry]:
            return [
                FileListEntry(
                    type="file",
                    path="model.safetensors",
                    size=10,
                ),
            ]

        async def fake_peer_status(
            peer_host: str,
            peer_port: int,
            model_id_normalized: str,
            timeout: float = 5.0,
        ) -> list[PeerFileInfo] | None:
            return [
                PeerFileInfo(
                    path="model.safetensors",
                    size=10,
                    complete=True,
                    safe_bytes=10,
                )
            ]

        async def fake_resolve_dir(model_id: ModelId) -> Path:
            return tmp_path

        async def fake_resolve_allow(shard: ShardMetadata) -> list[str]:
            return ["*"]

        target_path = tmp_path / "model.safetensors"

        async def fake_download(
            peer_ip: str,
            peer_port: int,
            model_id_normalized: str,
            file_path: str,
            target_dir: Path,
            expected_size: int,
            on_progress: object = None,
        ) -> Path | None:
            async with aiofiles.open(target_path, "wb") as f:
                await f.write(b"0123456789")
            return target_path

        async def file_meta_should_not_be_called(
            *_args: object, **_kwargs: object
        ) -> tuple[int, str]:
            raise AssertionError(
                "file_meta must not be called in offline mode -- the "
                "operator opted into trusting LAN peers"
            )

        monkeypatch.setattr(psd, "fetch_file_list_with_cache", fake_fetch)
        monkeypatch.setattr(psd, "get_peer_file_status", fake_peer_status)
        monkeypatch.setattr(psd, "resolve_model_dir", fake_resolve_dir)
        monkeypatch.setattr(psd, "resolve_allow_patterns", fake_resolve_allow)
        monkeypatch.setattr(psd, "download_file_from_peer", fake_download)
        monkeypatch.setattr(psd, "file_meta", file_meta_should_not_be_called)

        downloader = PeerAwareShardDownloader(NoopShardDownloader(), offline=True)
        shard = _make_shard(ModelId("test-org/model-a"))

        result = await downloader._try_peer_download(
            shard,
            peer_ip="10.0.0.1",
            peer_port=52415,
            model_id_normalized="test-org/model-a",
        )
        assert result is not None, (
            "offline peer download must succeed without consulting HF; "
            "got None which means the integrity check fired and the "
            "peer bytes were discarded"
        )
        assert await aios.path.exists(target_path), (
            "peer-downloaded file must be retained when offline mode "
            "skips the HF integrity check"
        )

    async def test_online_mode_still_calls_file_meta(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from exo.download import peer_shard_downloader as psd
        from exo.download.peer_download import PeerFileInfo
        from exo.shared.types.worker.downloads import FileListEntry

        async def fake_fetch(*_args: object, **_kwargs: object) -> list[FileListEntry]:
            return [
                FileListEntry(
                    type="file",
                    path="model.safetensors",
                    size=10,
                ),
            ]

        async def fake_peer_status(
            peer_host: str,
            peer_port: int,
            model_id_normalized: str,
            timeout: float = 5.0,
        ) -> list[PeerFileInfo] | None:
            return [
                PeerFileInfo(
                    path="model.safetensors",
                    size=10,
                    complete=True,
                    safe_bytes=10,
                )
            ]

        async def fake_resolve_dir(model_id: ModelId) -> Path:
            return tmp_path

        async def fake_resolve_allow(shard: ShardMetadata) -> list[str]:
            return ["*"]

        target_path = tmp_path / "model.safetensors"

        async def fake_download(
            peer_ip: str,
            peer_port: int,
            model_id_normalized: str,
            file_path: str,
            target_dir: Path,
            expected_size: int,
            on_progress: object = None,
        ) -> Path | None:
            async with aiofiles.open(target_path, "wb") as f:
                await f.write(b"0123456789")
            return target_path

        meta_calls: list[tuple[object, ...]] = []

        async def recording_meta(*args: object, **_kwargs: object) -> tuple[int, str]:
            meta_calls.append(args)
            # Return mismatched etag -> downloader will discard.
            return (10, "deadbeef" * 5)

        monkeypatch.setattr(psd, "fetch_file_list_with_cache", fake_fetch)
        monkeypatch.setattr(psd, "get_peer_file_status", fake_peer_status)
        monkeypatch.setattr(psd, "resolve_model_dir", fake_resolve_dir)
        monkeypatch.setattr(psd, "resolve_allow_patterns", fake_resolve_allow)
        monkeypatch.setattr(psd, "download_file_from_peer", fake_download)
        monkeypatch.setattr(psd, "file_meta", recording_meta)

        downloader = PeerAwareShardDownloader(NoopShardDownloader(), offline=False)
        shard = _make_shard(ModelId("test-org/model-a"))

        await downloader._try_peer_download(
            shard,
            peer_ip="10.0.0.1",
            peer_port=52415,
            model_id_normalized="test-org/model-a",
        )

        assert len(meta_calls) == 1, (
            "online mode must continue calling file_meta to validate "
            "peer-downloaded bytes against HF's authoritative hash; "
            f"got meta_calls={meta_calls!r}"
        )
