---
branch: feature/asymmetric-tp-integration
created: 2026-04-28
owner: cursor-agent
status: active
scope: "Adapt upstream asymmetric tensor parallelism for safe review in team-wcv/exo"
orchestraitor:
  ticket: none
  task_url: none
pr:
  url: https://github.com/team-wcv/exo/pull/5
  state: open
---

- Why this branch exists: prepare upstream exo PR #1821 for safe integration into the team-wcv fork.
- Changed paths: placement, shard metadata, MLX loading, asymmetric MLX sharding, placement previews, focused tests, and PR #6 stability/dashboard fixes.
- Validation run: targeted ruff, basedpyright, focused pytest, and dashboard svelte-check.
- Known follow-ups: real two-node Apple Silicon distributed generation test before enabling automatic asymmetric placement broadly.
