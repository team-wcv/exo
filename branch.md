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
- Validation run: pending refresh against current main and current-head review.
- Known follow-ups: include the asymmetric drafter runner in readiness checks before requesting review.
