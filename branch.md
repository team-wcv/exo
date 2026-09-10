---
branch: fix/6aa208b9-qwen4-long-prefill
created: 2026-09-10
owner: codex-agent
status: active
scope: "Keep Qwen4 sparse-indexer state aligned during prefix reuse and prefill rollback"
orchestraitor:
  ticket: 6aa208b9559bc082bf61ea61
pr:
  url: pending
  state: pending
---

- Why this branch exists: Qwen3.8 prefix reuse trims the normal KV cache but leaves its sparse-attention indexer at the old prompt length, producing incompatible attention-mask shapes and an empty HTTP 200 response.
- Changed paths: `src/exo/worker/engines/mlx/cache.py`, `src/exo/worker/engines/mlx/generator/generate.py`, and `src/exo/worker/tests/unittests/test_mlx/test_auxiliary_cache_trim.py`.
- Validation run: reproduce the 2,114-vs-1,612 mask mismatch, reload the Twin Tensor/JACCL instance, and verify the exact 1,847-token continuation plus repeated divergent-prefix prompts.
- Known follow-ups: none.
