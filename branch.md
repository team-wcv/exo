# Twins control delivery and inference errors

- Branch: fix/6b053e26-control-delivery
- Scope: reproduce control acknowledgement races; surface Responses failures; diagnose live Twins restarts without reducing context or disrupting unrelated services.
- Task: 6b053e263e896bda5f645d9a
- Plan: b7c44d90-0829-4807-a45b-6f766a8ec1e0
- Base: 43bfab7 origin/main
- PR: pending
- Status: candidate validated locally; production activation and agent rerun pending.
- Validation: 1,312 Python tests pass, 15 skipped, 212 slow tests deselected; native regressions cover receiving while publishing is blocked and ingress closure; full basedpyright 1.37.1 and Ruff checks pass. Nix unavailable; cargo fmt and Ruff formatting checked directly.
- Evidence boundary: the regression verifies continued receive during blocked publishing; it is not asserted as the sole cause of all live GPU/control failures.
- Peer review: Spark identified that awaiting a blocked operation after ingress closes can hang. A regression failed before the correction; ingress closure now cancels the operation and its caller sees the closed result channel.
