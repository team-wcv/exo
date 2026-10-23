"""Repair idle Mac workers that remain isolated from the configured LAN master."""

import argparse
import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class Record:
    pid: int = 0
    index: int = -1
    isolated_since: float | None = None
    last_restart: float | None = None


def decision(
    previous: Record,
    *,
    pid: int,
    index: int,
    master_fresh: bool,
    master_in_history: bool,
    idle: bool,
    master_reachable: bool,
    now: float,
    grace: float = 120,
    cooldown: float = 600,
) -> tuple[str, Record]:
    record = Record(pid, index, previous.isolated_since, previous.last_restart)
    progressing = master_in_history and previous.pid == pid and index > previous.index
    if master_fresh or progressing or not idle or not master_reachable:
        record.isolated_since = None
        status = (
            "healthy"
            if master_fresh
            else "catching_up"
            if progressing
            else "busy"
            if not idle
            else "master_unreachable"
        )
        return status, record
    if (
        previous.pid != pid
        or record.isolated_since is None
        or now < record.isolated_since
    ):
        record.isolated_since = now
    if now - record.isolated_since < grace:
        return "waiting", record
    if record.last_restart is not None and now - record.last_restart < cooldown:
        return "cooldown", record
    return "restart", record


def fresh(state: dict, node_id: str, now: float) -> bool:
    stamp = state.get("lastSeen", {}).get(node_id)
    if not isinstance(stamp, str):
        return False
    age = now - datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    return -5 <= age <= 45


def idle_worker(state: dict, node_id: str) -> bool:
    for download in state.get("downloads", {}).get(node_id, []):
        if "DownloadPending" in download or "DownloadOngoing" in download:
            return False
    for tagged in state.get("instances", {}).values():
        if not isinstance(tagged, dict) or len(tagged) != 1:
            return False
        instance = next(iter(tagged.values()))
        shards = instance.get("shardAssignments", {}).get("shards", [])
        if any(shard[0] == node_id for shard in shards):
            return False
        drafter = instance.get("drafterPlacement") or {}
        if drafter.get("drafterNodeId") == node_id:
            return False
    return True


def fetch(url: str):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=3) as response:
        return json.load(response)


def save(path: Path, record: Record) -> None:
    if path.exists():
        shutil.copy2(path, path.with_suffix(".previous.json"))
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(asdict(record), stream)
        temporary = Path(stream.name)
    temporary.replace(path)


def check(args: argparse.Namespace) -> None:
    state_path = Path(args.state_file).expanduser()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = (
            Record(**json.loads(state_path.read_text()))
            if state_path.exists()
            else Record()
        )
        service = f"gui/{os.getuid()}/com.teamwcv.exo.bigbrain"
        output = subprocess.check_output(["launchctl", "print", service], text=True)
        match = re.search(r"^\s*pid = (\d+)", output, re.MULTILINE)
        if not match:
            raise RuntimeError(
                "EXO launchd service is not running; leaving recovery to launchd"
            )
        pid = int(match.group(1))
        try:
            local_id = fetch(args.local_url + "/node_id")
            if local_id != args.node_id:
                raise ValueError(
                    "Local API node identity differs from configured worker"
                )
            local = fetch(args.local_url + "/state")
            now = time.time()
            master_fresh = fresh(local, args.master_id, now)
            reachable = False
            master_id = fetch(args.master_url + "/node_id")
            master = fetch(args.master_url + "/state")
            if master_id != args.master_id:
                raise ValueError("LAN master identity differs from configured master")
            if fresh(master, args.master_id, now):
                host = urlparse(args.master_url).hostname
                if host is None:
                    raise ValueError("Master URL has no hostname")
                with socket.create_connection((host, args.master_port), timeout=3):
                    reachable = True
            status, record = decision(
                previous,
                pid=pid,
                index=local["lastEventAppliedIdx"],
                master_fresh=master_fresh,
                master_in_history=args.master_id in local.get("lastSeen", {}),
                idle=idle_worker(local, args.node_id),
                master_reachable=reachable,
                now=now,
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            # Uncertain network or API state is not permission to restart a worker.
            previous.isolated_since = None
            save(state_path, previous)
            print(
                json.dumps({"status": "probe_failed", "error": str(error)}), flush=True
            )
            return
        if status == "restart" and args.apply:
            # Preserve the cooldown even if launchctl itself fails.
            record.last_restart = now
            record.isolated_since = None
            save(state_path, record)
            subprocess.Popen(
                ["/usr/bin/caffeinate", "-dimsu", "-t", "120"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            subprocess.run(
                ["launchctl", "kickstart", "-k", service], check=True, timeout=20
            )
            status = "restarted"
        else:
            save(state_path, record)
            if status == "restart":
                status = "would_restart"
        print(
            json.dumps(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                    "master_reachable": reachable,
                    **asdict(record),
                }
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--master-id", required=True)
    parser.add_argument("--master-url", default="http://192.168.1.21:52415")
    parser.add_argument("--master-port", type=int, default=52618)
    parser.add_argument("--local-url", default="http://127.0.0.1:52415")
    parser.add_argument("--state-file", default="~/.cache/exo/reconnect-guard.json")
    parser.add_argument("--apply", action="store_true")
    check(parser.parse_args())


if __name__ == "__main__":
    main()
