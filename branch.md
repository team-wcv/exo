---
branch: fix/6aa208b9-twin-mx-task-agreement
created: 2026-09-09
owner: codex-agent
status: active
scope: "Replace the JACCL point-to-point task-admission broadcast that crashed the Twin Qwen runner"
orchestraitor:
  ticket: 6aa208b9559bc082bf61ea61
pr:
  url: pending
  state: pending
---

- Why this branch exists: the 2026-09-09 first generation request crashed the non-root Twin runner with SIGSEGV in `mx_broadcast_int_list` / `mx_all_gather_tasks`.
- Changed paths: MLX task-agreement broadcast implementation and focused transport regression coverage.
- Validation run: Ruff and BasedPyright pass locally; 30/30 focused tests pass on wc-smbpt with a real Metal device; live two-host load, generation, and soak checks remain.
- Known follow-ups: deploy only after review and a restart-consensus poll.
