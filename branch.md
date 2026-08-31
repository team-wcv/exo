---
branch: fix/6a95c4b0-resilient-downloads-vq
created: 2026-08-31
owner: codex-agent
status: active
scope: "Stack peer-safe stale-download recovery and VQ codebook replication on the hardware-validated exo upstream merge"
orchestraitor:
  ticket: 6a95c4b0e6b6efd51fc8569d
  task_url: orchestraitor://task/6a95c4b0e6b6efd51fc8569d
pr:
  url: https://github.com/team-wcv/exo/pull/43
  state: open
cleanup:
  merged_into: team-wcv/chore/6a959fcc-sync-exo-upstream
  archived_after: null
  successor_branch: null
source_branches:
  - chore/6a959fcc-sync-exo-upstream (stack base at 7763d51e, PR #42)
  - exo-explore/exo PR #2278 (adapted HTTP 416 recovery)
  - exo-explore/exo PR #2268 (codebook replication guard only)
---

- Why this branch exists: add two bounded upstream improvements without changing the exact PR #42 build currently running on smbp/smbpt.
- Changed paths: peer-safe partial-download reset and bounded HTTP 416 recovery; vector-quantized codebook replication policy; focused recovery and sharding-policy tests.
- Validation run: all 153 download/policy tests pass; the sandbox-safe suite reports 594 passed, 5 skipped, and 2 Metal-only cases deselected, plus 8 stacked-base tests pass; BasedPyright reports 0 errors; whole-repo Ruff check and format check pass; Cargo workspace tests and fmt check pass. Nix is unavailable on this host.
- Known follow-ups: Deliberaitor plan creation is unavailable because its workspace-local route rejects this bastion-scoped Codex session; the PR will use the required outage note. VQ model cards from upstream #2268 are intentionally out of scope.
