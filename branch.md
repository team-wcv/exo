---
branch: feature/6a8facfc-qwen4-exp-tensor
created: 2026-08-26
owner: codex-agent
status: active
scope: "Add first-class Qwen4-Exp MLX tensor support and upstream the verified pipeline/cache compatibility fixes"
orchestraitor:
  ticket: 6a8facfc92a48d1b2d984ef6
  task_url: null
pr:
  url: https://github.com/team-wcv/exo/pull/40
  state: open
cleanup:
  merged_into: team-wcv/main
  archived_after: null
  successor_branch: null
source_branches:
  - feature/asymmetric-tp-integration (merged via PR #5; stale base ledger replaced)
---

- Why this branch exists: make Qwen3.8 Flash Next run across the Exo twins with JACCL/RDMA tensor sharding and preserve the live-proven compatibility fixes in reviewed source.
- Changed paths: model-card tensor eligibility, MLX auto-parallel Qwen4-Exp strategy, pipeline cache compatibility, focused model-card/PLE tests, and an isolated-source launchd override with operator documentation.
- Validation run: focused Ruff and BasedPyright pass; model-card pytest passes locally; all seven focused tests pass on the Metal-capable `smbp` twin; launch wrapper passes `zsh -n`; live TP2 + JACCL/RDMA returned two exact HTTP 200 completions at 42 tokens/s with both runners remaining ready.
- Known follow-ups: Deliberaitor plan creation is unavailable because its MCP rejects the authenticated Codex caller identity; the PR will document this required-plan fallback. The current Hugging Face loader still needs its external batch-position overlay until that model repository publishes the fix.
