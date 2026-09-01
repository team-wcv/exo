---
branch: feature/6a961781-apple-usb-ncm-prefill
created: 2026-08-31
owner: codex-agent
status: active
scope: "Admit verified Apple CDC-NCM link-local paths for an isolated Studio/Spark prefill benchmark"
orchestraitor:
  ticket: 6a961781e6b6efd51fc859bf
  task_url: orchestraitor://task/6a961781e6b6efd51fc859bf
pr:
  url: https://github.com/team-wcv/exo/pull/44
  state: open
cleanup:
  merged_into: team-wcv/fix/6a95c4b0-resilient-downloads-vq
  archived_after: null
  successor_branch: null
source_branches:
  - fix/6a95c4b0-resilient-downloads-vq (stack base at 920e64aa, PR #43)
  - torvalds/linux a5148bc2fa27092862ac4b9e7b5c8340d60cff34 (Apple CDC-NCM device match reference)
---

- Why this branch exists: Exo sees the patched Apple USB CDC-NCM interface but rejects every IPv4 link-local address before probing, so the physical Studio/Spark link cannot enter topology.
- Intended changes: classify only Apple 05ac:1905 CDC-NCM interfaces, admit IPv4 link-local probing only for that verified type, prefer 10GbE over the measured USB path and USB over Wi-Fi, add focused tests and operator documentation, then benchmark the isolated Studio/Spark pair over both routes.
- Safety boundary: do not relax link-local filtering for generic interfaces and do not alter the production smbp/smbpt services during benchmarking.
- Validation pending: focused tests, type checking, lint/format checks, isolated route proof and A/B benchmark, production post-test health, hosted checks, and current-head Codex review.
