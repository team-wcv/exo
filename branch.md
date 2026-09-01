---
branch: agent/fix-dashboard-current-shard-assignments
created: 2026-09-01
owner: codex-agent
status: active
scope: "Refresh PR #35 shard-assignment dashboard parsing against current main and resolve review findings"
orchestraitor:
  ticket: 6a96fe4ae6b6efd51fc8623e
pr:
  url: https://github.com/team-wcv/exo/pull/35
  state: open
---

- Why this branch exists: parse both legacy and current shard-assignment wire formats in the dashboard.
- Changed paths: dashboard instance metadata, readiness, download, and topology parsing.
- Validation run: merged `team-wcv/main` at `9e1b8ef4`; Svelte check reports 0 errors/warnings; production build and Prettier checks pass.
- Known follow-ups: current-head CI and Codex review.
