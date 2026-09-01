---
branch: chore/6a96f375-bigbrain-endpoints
created: 2026-09-01
owner: codex-agent
status: active
scope: "Add durable isolated Bigbrain endpoint profiles for Twins, Studio, and Spark"
orchestraitor:
  ticket: 6a96f375e6b6efd51fc86192
pr:
  url: pending
  state: pending
---

- Why this branch exists: Keep the Twin cluster on svc:bigbrain while Studio and Spark expose independent Exo control planes and Tailscale Services.
- Changed paths: pending
- Validation run: pending
- Known follow-ups: Spark cannot load another large model while Laguna occupies most unified memory; endpoint availability and simultaneous heavyweight inference are separate constraints.
