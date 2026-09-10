---
branch: fix/6aa208b9-qwen4-long-prefill
created: 2026-09-10
owner: codex-agent
status: active
scope: "Keep Qwen4 prefix state aligned and prevent empty assistant turns from poisoning chat history"
orchestraitor:
  ticket: 6aa208b9559bc082bf61ea61
pr:
  url: https://github.com/team-wcv/exo/pull/48
  state: open
---

- Why this branch exists: Qwen3.8 prefix reuse trims the normal KV cache but leaves its sparse-attention indexer at the old prompt length, producing incompatible attention-mask shapes and an empty HTTP 200 response.
- Changed paths: `src/exo/worker/engines/mlx/cache.py`, `src/exo/worker/engines/mlx/generator/generate.py`, `src/exo/worker/tests/unittests/test_mlx/test_auxiliary_cache_trim.py`, and `dashboard/src/lib/stores/app.svelte.ts`.
- Validation run: 14 focused cache tests pass on Metal; BasedPyright reports zero errors/warnings; Svelte check reports zero errors/warnings; the exact 1,805-token conversation and cached follow-up return visible answers at 40.8–41.0 tok/s on Twin Tensor/JACCL.
- Known follow-ups: none.
