import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "reconnect_guard", Path(__file__).with_name("reconnect_guard.py")
)
assert spec is not None and spec.loader is not None
guard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guard
spec.loader.exec_module(guard)


def decide(previous=None, **changes):
    values = dict(
        pid=10,
        index=100,
        master_fresh=False,
        master_in_history=False,
        idle=True,
        master_reachable=True,
        now=1000,
    )
    values.update(changes)
    return guard.decision(previous or guard.Record(), **values)


def test_requires_sustained_isolation_before_restart():
    status, record = decide()
    assert status == "waiting"
    assert decide(record, now=1119)[0] == "waiting"
    assert decide(record, now=1120)[0] == "restart"


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"master_fresh": True}, "healthy"),
        ({"master_reachable": False}, "master_unreachable"),
        ({"idle": False}, "busy"),
    ],
)
def test_uncertain_or_active_state_does_not_restart(changes, expected):
    previous = guard.Record(pid=10, index=90, isolated_since=0)
    status, record = decide(previous, **changes)
    assert status == expected
    assert record.isolated_since is None


def test_replay_progress_does_not_restart_even_with_old_timestamps():
    previous = guard.Record(pid=10, index=90, isolated_since=0)
    status, record = decide(previous, master_in_history=True)
    assert status == "catching_up"
    assert record.isolated_since is None


def test_singleton_master_index_progress_does_not_hide_partition():
    previous = guard.Record(pid=10, index=90, isolated_since=0)
    assert decide(previous, master_in_history=False)[0] == "restart"


def test_pid_change_gets_new_startup_grace_and_restart_has_cooldown():
    previous = guard.Record(pid=9, index=90, isolated_since=0)
    assert decide(previous)[0] == "waiting"
    previous = guard.Record(pid=10, index=90, isolated_since=0, last_restart=900)
    assert decide(previous)[0] == "cooldown"
    assert decide(previous, now=1500)[0] == "restart"


def test_guard_preserves_local_model_but_allows_other_nodes_model():
    instance = {
        "MlxRingInstance": {"shardAssignments": {"shards": [["local", "runner", {}]]}}
    }
    state = {"instances": {"instance": instance}}
    assert not guard.idle_worker(state, "local")
    assert guard.idle_worker(state, "another-node")


def test_node_freshness_does_not_treat_historical_presence_as_connected():
    state = {"lastSeen": {"master": "2026-10-23T00:00:00Z"}}
    stamp = guard.datetime.fromisoformat("2026-10-23T00:00:00+00:00").timestamp()
    assert guard.fresh(state, "master", stamp + 10)
    assert not guard.fresh(state, "master", stamp + 100)
