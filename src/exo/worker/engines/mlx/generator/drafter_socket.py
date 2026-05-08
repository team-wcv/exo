"""Direct TCP socket transport for the asymmetric drafter wire.

The original drafter wire (:mod:`remote_drafter`) carries small uint32
arrays via ``mx.distributed.send/recv`` over the parent
``mx.distributed.Group``. That design forces the drafter rank to be a
member of the parent group, which in turn requires
``mx.distributed.Group.split`` so target ranks can run TP/PP collectives
without dragging the drafter in. JACCL and ring backends do not
implement ``split`` on Apple Silicon, so the V1 asymmetric path was
limited to a single target rank.

This module breaks that coupling. The drafter rank no longer joins
``mx.distributed`` at all. Instead, target rank 0 binds a TCP server
socket at instance bootstrap time, the drafter dials it, and the same
wire frames flow over that connection. The target's
``mx.distributed.Group`` therefore contains only target ranks and is
free to do whatever TP/PP work it needs without ``Group.split``.

Wire frames are length-implicit (every op type has a known fixed shape;
``OP_PREFILL`` carries a variable-length token array whose length is
announced in the preceding command frame's ``num_forwards`` slot). Each
uint32 is serialised little-endian, matching mlx_lm's on-device layout
for ``mx.uint32``.

Threading model: both the target rank's ``RemoteTransport`` and the
drafter rank's serve loop run wire ops serially on a single thread (the
target uses a single-worker ``ThreadPoolExecutor``; the drafter loops
synchronously). Concurrency is multiplexed via session ids, not via
multiple sockets, so a single TCP connection per asymmetric instance is
sufficient and avoids mid-flight reordering.
"""

from __future__ import annotations

import socket
import struct
import time
from typing import Final

_HEADER_FORMAT: Final[str] = "<I"
"""Length prefix for variable-length payloads.

Used only for OP_PREFILL's prompt-token tail. Fixed-shape frames don't
need a header because both sides know the shape statically."""


def send_uint32_frame(sock: socket.socket, values: list[int]) -> None:
    """Send a fixed-length uint32 frame over ``sock``.

    Caller must guarantee both peers know the frame length statically;
    no length prefix is sent. Suitable for command/ack/drafts frames.
    """
    if not all(0 <= v <= 0xFFFFFFFF for v in values):
        raise ValueError(f"frame contains non-uint32 values: {values}")
    payload = struct.pack(f"<{len(values)}I", *values)
    sock.sendall(payload)


def recv_uint32_frame(sock: socket.socket, count: int) -> list[int]:
    """Receive ``count`` uint32 ints over ``sock`` (no length prefix).

    Blocks until ``count * 4`` bytes have been received, raising
    :class:`ConnectionError` if the peer closes mid-frame.
    """
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
    needed = count * 4
    buf = bytearray(needed)
    view = memoryview(buf)
    received = 0
    while received < needed:
        chunk = sock.recv_into(view[received:], needed - received)
        if chunk == 0:
            raise ConnectionError(
                f"drafter wire closed mid-frame "
                f"(received {received}/{needed} bytes)"
            )
        received += chunk
    unpacked = struct.unpack(f"<{count}I", bytes(buf))
    return list(unpacked)


def send_variable_uint32_payload(sock: socket.socket, values: list[int]) -> None:
    """Send a length-prefixed uint32 payload (4-byte header + values).

    Used for OP_PREFILL's prompt-token tail when the size isn't carried
    in the preceding command frame's slot.
    """
    if not all(0 <= v <= 0xFFFFFFFF for v in values):
        raise ValueError("variable payload contains non-uint32 values")
    header = struct.pack(_HEADER_FORMAT, len(values))
    sock.sendall(header)
    if values:
        sock.sendall(struct.pack(f"<{len(values)}I", *values))


def bind_target_listener(host: str, port: int, *, backlog: int = 1) -> socket.socket:
    """Open and listen on ``(host, port)`` for the drafter's incoming dial.

    Bound with ``SO_REUSEADDR`` so a previous instance teardown that
    left the port in TIME_WAIT does not block reclaim. Caller is
    responsible for ``accept()`` and ``close()``.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(backlog)
    return listener


def accept_drafter(
    listener: socket.socket,
    *,
    timeout_seconds: float = 60.0,
) -> socket.socket:
    """Block on ``listener.accept`` for the drafter's incoming connection.

    The drafter dials soon after target rank 0 reaches its
    ``ConnectToGroup`` step, so a generous default timeout (60s) covers
    drafter-side weight loading and warmup without spinning. ``TCP_NODELAY``
    is set on the accepted socket because every wire op is a small
    request/reply round trip; Nagle would add ~40ms of latency per op
    while batching tiny frames.
    """
    listener.settimeout(timeout_seconds)
    try:
        accepted = listener.accept()
    finally:
        listener.settimeout(None)
    conn: socket.socket = accepted[0]
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return conn


def dial_target(
    host: str,
    port: int,
    *,
    total_timeout_seconds: float = 120.0,
    initial_backoff_seconds: float = 0.5,
) -> socket.socket:
    """Dial ``(host, port)`` with exponential backoff until connected.

    Used by the drafter rank to reach target rank 0's listener. Target
    rank 0 binds inside its ``ConnectToGroup`` step, which races with
    the drafter rank's bootstrap; the drafter therefore retries until
    the listener is up or the deadline expires. Backoff caps at 5s
    between attempts so we don't sleep through a transient binding
    hiccup.
    """
    deadline = time.monotonic() + total_timeout_seconds
    backoff = initial_backoff_seconds
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            conn = socket.create_connection(
                (host, port), timeout=min(10.0, total_timeout_seconds)
            )
            conn.settimeout(None)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return conn
        except (ConnectionRefusedError, OSError, TimeoutError) as exc:
            last_error = exc
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 5.0)
    raise ConnectionError(
        f"drafter could not reach target rank 0 at {host}:{port} "
        f"within {total_timeout_seconds:.0f}s "
        f"(last error: {last_error!r})"
    )


__all__ = [
    "accept_drafter",
    "bind_target_listener",
    "dial_target",
    "recv_uint32_frame",
    "send_uint32_frame",
    "send_variable_uint32_payload",
]
