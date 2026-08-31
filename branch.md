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
  url: null
  state: pending
cleanup:
  merged_into: team-wcv/main
  archived_after: null
  successor_branch: null
source_branches:
  - team-wcv/main (fork integration baseline at 0909a1ed)
  - origin/main (official upstream; fetched before merge)
---

- Why this branch exists: preserve the known-good two-node exo deployment, then bring the fork current with official upstream without losing team-wcv custom behavior.
- Changed paths: pending merge inventory and conflict resolution.
- Validation run: pending.
- Known follow-ups: Deliberaitor plan creation is unavailable because its workspace-local route rejects this bastion-scoped Codex session; the PR will use the required outage note. No live-host deployment is included.
