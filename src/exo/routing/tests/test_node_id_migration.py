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
