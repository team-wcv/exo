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
  url: null
  state: pending
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
- Changed paths: pending implementation in download recovery, peer partial metadata, tensor sharding, and focused tests.
- Validation run: pending.
- Known follow-ups: Deliberaitor plan creation is unavailable because its workspace-local route rejects this bastion-scoped Codex session; the PR will use the required outage note. VQ model cards from upstream #2268 are intentionally out of scope.
