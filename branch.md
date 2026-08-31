---
branch: chore/6a959fcc-sync-exo-upstream
created: 2026-08-31
owner: codex-agent
status: active
scope: "Merge current exo-explore/exo main into the team-wcv fork while preserving the fork's custom inference, fleet, and twin-recovery work"
orchestraitor:
  ticket: 6a959fcce6b6efd51fc85507
  task_url: orchestraitor://task/6a959fcce6b6efd51fc85507
pr:
  url: https://github.com/team-wcv/exo/pull/42
  state: open
cleanup:
  merged_into: team-wcv/main
  archived_after: null
  successor_branch: null
source_branches:
  - team-wcv/main (fork integration baseline at 0909a1ed)
  - origin/main (official upstream at 21a54c5e)
---

- Why this branch exists: preserve the known-good two-node exo deployment, then bring the fork current with official upstream without losing team-wcv custom behavior.
- Changed paths: upstream Zenoh/exo_rs transition; Python package and dashboard locks; persistent scoped identity; event routing and election recovery; model-card/backend compatibility; peer-download, custom drafting, placement, runner supervision, API serialization, tests, and formatting.
- Validation run: matching smbp/smbpt snapshots archived with Git bundle at f033e597; official and fork tips are ancestors of the merge head; uv lock/sync succeeded; Ruff and Ruff format pass; BasedPyright reports 0 errors; Cargo check/test/fmt pass; dashboard npm ci/build, Svelte check, and Prettier pass; CPU-safe pytest result is 611 passed, 5 skipped, 3 environment-deselected. The first twin launch test found that upstream rejected the legacy `--bootstrap-peers` argument; this branch now retains it as a warned no-op under Zenoh and covers the behavior with regression tests. Nix is unavailable on this host.
- Known follow-ups: Deliberaitor plan creation is unavailable because its workspace-local route rejects this bastion-scoped Codex session; the PR will use the required outage note. Complete the corrected twin cutover and validate MLX/Metal inference plus Zenoh discovery before merge. The legacy `--libp2p-port` flag is accepted as a Zenoh-port alias, while `--bootstrap-peers` remains accepted for launch-script compatibility and Zenoh performs discovery automatically.
