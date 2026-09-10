---
branch: fix/6aa208b9-qwen4-long-prefill
created: 2026-09-10
owner: codex-agent
status: active
scope: "Fix Qwen4 hybrid-cache long-context prefill on Twin Tensor/RDMA instances"
orchestraitor:
  ticket: 6aa208b9559bc082bf61ea61
pr:
  url: pending
  state: pending
---

- Why this branch exists: Qwen3.8 returns an immediate empty completion once recurrent-cache prefills exceed roughly 900 tokens.
- Changed paths: `src/exo/worker/engines/mlx/generator/generate.py` and focused tests.
- Validation run: reproduce at 913/1013/1847 tokens, reload the Twin Tensor/JACCL instance, and verify the exact 1847-token continuation plus short-chat throughput.
- Known follow-ups: none.
