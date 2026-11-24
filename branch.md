# Twins control delivery and inference errors

- Branch: fix/6b053e26-control-delivery
- Scope: reproduce control acknowledgement races; surface Responses failures; diagnose live Twins restarts without reducing context or disrupting unrelated services.
- Task: 6b053e263e896bda5f645d9a
- Plan: b7c44d90-0829-4807-a45b-6f766a8ec1e0
- Base: 43bfab7 origin/main
- PR: https://github.com/team-wcv/exo/pull/68
- Status: activated on Studio, SMBP and SMBPT; real local-agent acceptance run in progress.
- Validation: 1,312 Python tests pass, 15 skipped, 212 slow tests deselected; native regressions cover receiving while publishing is blocked and ingress closure; full basedpyright 1.37.1 and Ruff checks pass. Nix unavailable; cargo fmt and Ruff formatting checked directly.
- Evidence boundary: the regression verifies continued receive during blocked publishing; it is not asserted as the sole cause of all live GPU/control failures.
- Peer review: Spark identified that awaiting a blocked operation after ingress closes can hang. A regression failed before the correction; ingress closure now cancels the operation and its caller sees the closed result channel.
- Native validation: 6 networking tests pass; the Python 3.14 release package exchanged 10 bidirectional loopback messages. Staging receipts/backups are under each Mac's `~/.orchestraitor/deployments/exo-control-delivery-20261124/`; source artifact is `1ade078`.
- Partial rollout: both Twins imported/reloaded `1ade078`; Studio remains staged pending explicit approval. A short live probe still lost its instance and exposed a second empty-completion path: `InstanceDeleted` closed the queue without an error. Added a failing-then-passing regression and explicit error delivery before closure. No stability acceptance claimed.
- Full rollout follow-up: explicit operator approval in General message `msg_mwuyfxhe6vy97`; Studio activated at 17:17 UTC on November 24 with Python `b6fd265` and native `1ade078`. Its receipt and the current two-rank JACCL/RDMA instance were read back. A short request completed in 2.39 seconds; the subsequent long agent run still has substantial prefill latency. Full three-agent acceptance remains in progress. The preceding partial-rollout entry is historical.
