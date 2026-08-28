---
branch: fix/6a8facfc-exo-twin-restart-recovery
created: 2026-08-28
owner: codex-agent
status: active
scope: "Make Exo twin watchdog restarts converge and pin control to Thunderbolt"
orchestraitor:
  ticket: 6a8facfc92a48d1b2d984ef6
  task_url: null
pr:
  url: https://github.com/team-wcv/exo/pull/41
  state: open
cleanup:
  merged_into: team-wcv/main
  archived_after: null
  successor_branch: null
source_branches:
  - feature/6a8facfc-qwen4-exp-tensor (merged via PR #40; recovery continued after merge)
---

- Why this branch exists: PR #40 merged before the live watchdog follow-up; preserve the restart-election and Thunderbolt-control fixes in a reviewable successor PR and address the Qwen pipeline cache P1 reported on the merged PR.
- Changed paths: post-settle election recovery, focused election tests, Thunderbolt-pinned twin launch configuration, operator documentation, and adaptive Qwen4-Exp pipeline cache handling.
- Validation run: successor head passed 28 election tests, 37 routing+master tests, 5 focused Qwen cache tests, Ruff, BasedPyright, and wrapper syntax on `wc-smbp`; live recovery uses ~0.35 ms Thunderbolt control, both TP2 runners are Ready, inference returned HTTP 200 at 42.00/41.41 tokens/s, and both delivery queues remain zero.
- Known follow-ups: Deliberaitor plan creation is unavailable because its MCP rejects the authenticated Codex caller identity; the PR will document this required-plan fallback. The current Hugging Face loader still needs its external batch-position overlay until that model repository publishes the fix.
