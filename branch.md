---
branch: chore/6a96fa36-bmbp-bigbrain-endpoint
created: 2026-09-01
owner: codex-agent
status: active
scope: "Add a durable isolated Bigbrain endpoint profile for wc-bmbp and revalidate every host"
orchestraitor:
  ticket: 6a96fa36e6b6efd51fc86202
pr:
  url: pending
  state: pending
---

- Why this branch exists: Give wc-bmbp a persistent solo Exo control plane for local runtimes without changing the Twin, Studio, or Spark namespaces.
- Changed paths: `ops/team-wcv/bigbrain/bmbp/` LaunchAgent profile and endpoint-isolation runbook update.
- Validation run: all four dashboard roots return HTTP 200 from peer devices; Twins remain exactly smbp/smbpt with two RunnerReady runners; Studio, Spark, and BMBP each report one isolated topology node and 124 models; all persistent services are running; Spark Laguna remains active with zero restarts.
- Known follow-ups: BMBP should use localhost for same-host runtimes because its own Tailnet hostname does not hairpin from the host; peer devices reach the Tailnet URL normally.
