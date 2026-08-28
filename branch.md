---
branch: feature/6a8facfc-qwen4-exp-tensor
created: 2026-08-26
owner: codex-agent
status: active
scope: "Add Qwen4-Exp tensor support and recover half-open Exo control-plane delivery"
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
- Changed paths: model-card tensor eligibility, MLX auto-parallel Qwen4-Exp strategy, pipeline cache compatibility, focused model-card/PLE tests, an isolated-source launchd override, a fleet-scoped event-delivery watchdog, post-settle election recovery, and Thunderbolt-pinned control routing with operator documentation.
- Validation run: remote election tests 28 passed, routing+master tests 37 passed, Qwen tests 7 passed, Ruff/BasedPyright pass, and the wrapper passes `zsh -n`; a worker-only restart rejoined the forced master through the delayed campaign, final twin control uses the ~0.35 ms Thunderbolt path, and live TP2 returned two HTTP 200 completions at 42.00/41.41 tokens/s with both runners ready and `out_for_delivery=0`.
- Known follow-ups: Deliberaitor plan creation is unavailable because its MCP rejects the authenticated Codex caller identity; the PR will document this required-plan fallback. The current Hugging Face loader still needs its external batch-position overlay until that model repository publishes the fix.
