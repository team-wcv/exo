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
  url: pending
  state: pending
---

- Why this branch exists: prepare upstream exo PR #1821 for safe integration into the team-wcv fork.
- Changed paths: placement, shard metadata, MLX loading, asymmetric MLX sharding, placement previews, focused tests.
- Validation run: targeted ruff, basedpyright, and focused pytest.
- Known follow-ups: real two-node Apple Silicon distributed generation test before enabling automatic asymmetric placement broadly.
