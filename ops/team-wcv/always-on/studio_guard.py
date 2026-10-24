#!/usr/bin/env python3
"""Keep the dedicated Studio model available without restarting other services."""

import argparse
import fcntl
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def command(lms, *args):
    """Run the supported CLI without leaking prompts into service logs."""
    result = subprocess.run(
        [lms, *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode:
        raise RuntimeError(f"lms {' '.join(args[:2])} failed: {result.stderr[-1500:]}")
    return result.stdout


def loaded(lms, identifier):
    """Check resident instances; catalog availability alone is insufficient."""
    payload = json.loads(command(lms, "ps", "--json"))
    if not isinstance(payload, list):
        raise RuntimeError("Expected lms ps to return a list of loaded models")
    return any(model.get("identifier") == identifier for model in payload)


def reconcile(config):
    """Start only the missing daemon, listener, or dedicated model instance."""
    lms = str(Path(config["lms"]).expanduser())
    command(lms, "daemon", "up")
    try:
        with urllib.request.urlopen(config["health_url"], timeout=5) as response:
            if response.status != 200:
                raise RuntimeError("Studio model server returned a non-success status")
    except (urllib.error.URLError, TimeoutError):
        command(
            lms, "server", "start", "--port", str(config["port"]), "--bind", "0.0.0.0"
        )
    if loaded(lms, config["identifier"]):
        return "ready"
    command(
        lms,
        "load",
        config["model_key"],
        "--identifier",
        config["identifier"],
        "--context-length",
        str(config["context_length"]),
        "--gpu",
        "max",
        "--parallel",
        str(config["parallel"]),
        "--yes",
    )
    if not loaded(lms, config["identifier"]):
        raise RuntimeError(
            "Studio model load returned but its identifier is not resident"
        )
    return "loaded"


def main():
    """Serialize reconciliation and back off failed loads to avoid restart loops."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    lock = Path(config["lock_file"]).expanduser()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        cooldown = lock.with_suffix(".retry-after")
        while True:
            try:
                retry_after = float(cooldown.read_text()) if cooldown.exists() else 0
            except (OSError, ValueError):
                retry_after = 0
            if retry_after <= time.time():
                try:
                    result = reconcile(config)
                    print(f"Studio service {result}", flush=True)
                    cooldown.unlink(missing_ok=True)
                except Exception as error:
                    cooldown.write_text(str(time.time() + 300))
                    print(f"Studio service retry scheduled: {error}", flush=True)
                    if not args.watch:
                        raise
            if not args.watch:
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
