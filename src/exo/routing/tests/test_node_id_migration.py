"""Regression tests for the cache→config migration of the node-ID
keypair (Codex P1, PR #16 round 5).

The keypair used to live under ``EXO_CACHE_HOME``, which is subject
to normal cache cleanup (e.g. ``trash ~/.cache/exo``) and would
silently regenerate a new node-ID. The fix relocates the keypair to
``EXO_CONFIG_HOME`` and migrates legacy files transparently.
"""

from __future__ import annotations

from pathlib import Path

from exo_pyo3_bindings import Keypair

from exo.routing.router import (
    _migrate_legacy_node_id_keypair,  # pyright: ignore[reportPrivateUsage]
    get_node_id_keypair,
)


def test_legacy_keypair_is_migrated_to_new_location(tmp_path: Path) -> None:
    """Legacy cache-dir keypair must be moved to the new config-dir
    location and the legacy file removed -- so the node retains its
    identity across the upgrade and a future cache wipe doesn't
    resurrect a stale copy."""
    legacy_path = tmp_path / "cache" / "node_id.keypair"
    new_path = tmp_path / "config" / "node_id.keypair"
    legacy_path.parent.mkdir(parents=True)

    keypair = Keypair.generate()
    legacy_bytes = keypair.to_bytes()
    legacy_path.write_bytes(legacy_bytes)

    _migrate_legacy_node_id_keypair(new_path, legacy_path)

    assert new_path.exists(), "migration must place keypair at new location"
    assert new_path.read_bytes() == legacy_bytes, (
        "migration must preserve the byte-for-byte keypair contents "
        "so the node retains its peer ID"
    )
    assert not legacy_path.exists(), (
        "migration must remove the legacy file once the new location "
        "holds the keypair, otherwise a later cache wipe could "
        "resurrect a now-stale copy"
    )


def test_migration_is_idempotent_when_new_location_already_present(
    tmp_path: Path,
) -> None:
    """If the new location already has a keypair, migration must be
    a no-op even when a legacy file exists -- otherwise we'd
    overwrite the (canonical) new keypair with a stale legacy one."""
    legacy_path = tmp_path / "cache" / "node_id.keypair"
    new_path = tmp_path / "config" / "node_id.keypair"
    legacy_path.parent.mkdir(parents=True)
    new_path.parent.mkdir(parents=True)

    canonical = Keypair.generate().to_bytes()
    legacy = Keypair.generate().to_bytes()
    new_path.write_bytes(canonical)
    legacy_path.write_bytes(legacy)

    _migrate_legacy_node_id_keypair(new_path, legacy_path)

    assert new_path.read_bytes() == canonical, (
        "migration must NOT overwrite an existing new-location keypair"
    )
    # We deliberately leave the legacy file alone in this branch:
    # touching it would surprise an operator who is intentionally
    # keeping both copies during an upgrade window.
    assert legacy_path.exists()


def test_migration_skipped_when_no_legacy_file(tmp_path: Path) -> None:
    """Fresh installs must not error when the legacy path is absent."""
    new_path = tmp_path / "config" / "node_id.keypair"
    new_path.parent.mkdir(parents=True)

    _migrate_legacy_node_id_keypair(new_path, tmp_path / "missing.keypair")

    assert not new_path.exists()


def test_get_node_id_keypair_uses_migrated_legacy_keypair(tmp_path: Path) -> None:
    """End-to-end: ``get_node_id_keypair`` must surface the legacy
    keypair bytes when only the legacy path holds a valid file at
    call time, completing the cache→config migration on first use."""
    legacy_path = tmp_path / "cache" / "node_id.keypair"
    new_path = tmp_path / "config" / "node_id.keypair"
    legacy_path.parent.mkdir(parents=True)

    keypair = Keypair.generate()
    expected_bytes = keypair.to_bytes()
    legacy_path.write_bytes(expected_bytes)

    loaded = get_node_id_keypair(path=new_path, legacy_path=legacy_path)

    assert loaded.to_bytes() == expected_bytes
    assert new_path.exists()
    assert not legacy_path.exists()


# ---------------------------------------------------------------------------
# Codex P1 (PR #16 round-(N+2), router.py:297): per-process scoping
# ---------------------------------------------------------------------------
#
# The new same-host multi-node workflow (per-process
# ``--peer-download-port``) requires distinct ``NodeId``s per
# process so peer-discovery's self-skip and routing's unique-NodeId
# invariants hold. ``get_node_id_keypair`` therefore accepts a
# ``process_scope`` argument that is folded into the on-disk
# filename.


def test_distinct_process_scopes_produce_distinct_keypairs(tmp_path: Path) -> None:
    """Two processes that pass different scopes (e.g. distinct
    peer-download ports) MUST end up with different keypair files
    and different on-disk identities; otherwise two same-host
    nodes would race on the same NodeId."""
    base_path = tmp_path / "config" / "node_id.keypair"

    keypair_a = get_node_id_keypair(
        path=base_path, legacy_path=None, process_scope=52416
    )
    keypair_b = get_node_id_keypair(
        path=base_path, legacy_path=None, process_scope=52417
    )

    assert keypair_a.to_bytes() != keypair_b.to_bytes(), (
        "distinct process scopes must yield distinct keypairs so "
        "same-host multi-node deployments don't share a NodeId"
    )

    scoped_a = base_path.parent / "node_id.52416.keypair"
    scoped_b = base_path.parent / "node_id.52417.keypair"
    assert scoped_a.exists()
    assert scoped_b.exists()
    assert scoped_a.read_bytes() != scoped_b.read_bytes()


def test_same_process_scope_is_stable_across_calls(tmp_path: Path) -> None:
    """Per-process scoping must remain *persistent*: the same
    process (same scope) must load the same keypair on subsequent
    calls -- otherwise restart would silently churn NodeIds."""
    base_path = tmp_path / "config" / "node_id.keypair"

    first = get_node_id_keypair(path=base_path, legacy_path=None, process_scope=52416)
    second = get_node_id_keypair(path=base_path, legacy_path=None, process_scope=52416)

    assert first.to_bytes() == second.to_bytes()


def test_migration_runs_inside_file_lock(tmp_path: Path) -> None:
    """Codex P2 (PR #16 round-(N+2), router.py:322): the legacy
    migration must execute *inside* ``FileLock`` so two processes
    booting concurrently can't both pass the existence check, race
    each other into divergent in-memory keypairs, and end up with
    mismatched identities for the same on-disk file.

    We assert this structurally by hooking ``_migrate_legacy_node_id_keypair``
    and ``filelock.FileLock`` and verifying the lock is acquired
    *before* the migrator is called. A pre-lock migration would
    show ``migrate_called=True`` while the lock is still
    ``unacquired``."""
    import exo.routing.router as router_mod

    legacy_path = tmp_path / "cache" / "node_id.keypair"
    base_path = tmp_path / "config" / "node_id.keypair"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_bytes(Keypair.generate().to_bytes())

    lock_state: dict[str, bool] = {"acquired": False, "acquired_before_migrate": False}

    # We hook ``router_mod.FileLock`` (the symbol the production
    # code dereferences) with a thin wrapper class. The wrapper
    # delegates to the real ``FileLock`` instance but flips the
    # ``acquired`` flag on entry, which the migrator hook below
    # then snapshots. This keeps the type of ``FileLock`` intact
    # while letting us observe acquire-vs-migrate ordering.
    real_filelock = router_mod.FileLock

    class _ObservingFileLock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._inner = real_filelock(*args, **kwargs)  # pyright: ignore[reportArgumentType]

        def __enter__(self) -> object:
            lock_state["acquired"] = True
            return self._inner.__enter__()

        def __exit__(self, *exc: object) -> object:
            return self._inner.__exit__(*exc)  # pyright: ignore[reportArgumentType]

    original_migrate = router_mod._migrate_legacy_node_id_keypair  # pyright: ignore[reportPrivateUsage]

    def _track_migrate(new_path: Path, legacy: Path) -> None:
        lock_state["acquired_before_migrate"] = lock_state["acquired"]
        original_migrate(new_path, legacy)

    router_mod.FileLock = _ObservingFileLock
    router_mod._migrate_legacy_node_id_keypair = _track_migrate  # pyright: ignore[reportPrivateUsage]
    try:
        _ = get_node_id_keypair(path=base_path, legacy_path=legacy_path)
    finally:
        router_mod.FileLock = real_filelock
        router_mod._migrate_legacy_node_id_keypair = original_migrate  # pyright: ignore[reportPrivateUsage]

    assert lock_state["acquired_before_migrate"] is True, (
        "legacy migration must run INSIDE the FileLock to prevent a "
        "concurrent-startup race on the on-disk keypair"
    )


def test_legacy_migration_adopts_into_scoped_path(tmp_path: Path) -> None:
    """When a process passes a scope and a legacy unscoped keypair
    exists, the legacy bytes must be adopted into the scoped path.
    This is the upgrade-time behaviour: the first process to boot
    after the upgrade keeps the operator's existing identity; later
    processes (different scopes) start with fresh identities, which
    is exactly what per-process isolation requires."""
    legacy_path = tmp_path / "cache" / "node_id.keypair"
    base_path = tmp_path / "config" / "node_id.keypair"
    legacy_path.parent.mkdir(parents=True)

    expected_bytes = Keypair.generate().to_bytes()
    legacy_path.write_bytes(expected_bytes)

    loaded = get_node_id_keypair(
        path=base_path, legacy_path=legacy_path, process_scope=52416
    )

    scoped = base_path.parent / "node_id.52416.keypair"
    assert loaded.to_bytes() == expected_bytes
    assert scoped.exists(), "legacy bytes must land at the scoped path"
    assert scoped.read_bytes() == expected_bytes
    assert not legacy_path.exists()
