"""Regression coverage for shutdown with a dead multiprocessing consumer."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("endpoint", ["send", "recv"])
def test_aborted_channel_does_not_join_dead_consumer_at_exit(
    tmp_path: Path, endpoint: str
) -> None:
    # Exercise interpreter finalization, not just abort() returning. The original
    # failure printed "Shutdown complete" and then blocked in an atexit join.
    script = tmp_path / "dead_consumer.py"
    script.write_text(
        """
import multiprocessing as mp
import time
from exo.utils.channels import mp_channel

def consumer(recv, ready):
    ready.set()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    send, recv = mp_channel[bytes]()
    ready = mp.Event()
    child = mp.Process(target=consumer, args=(recv, ready))
    child.start()
    assert ready.wait(3)
    child.join(3)
    assert child.exitcode == 0
    send.send(b'x' * (2 * 1024 * 1024))
    time.sleep(0.1)
    ENDPOINT.abort()
    ENDPOINT.abort()
    print('shutdown-returned', flush=True)
""".replace("ENDPOINT", endpoint)
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[3])},
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert result.returncode == 0, result.stderr
    assert "shutdown-returned" in result.stdout
