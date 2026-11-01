# Twins runtime stability — October 31, 2026

## Incident and evidence

During a sequential cat-mode run on the full Qwen3.8 mixed 4/8-bit checkpoint,
rank zero failed at 16:14:47 Pacific with:

```
mx_all_gather_tasks: broadcast count 1001667840 outside [0, 16]; broadcast buffer corrupt
```

The stack is `SequentialGenerator.step` → `mlx_generate` →
`on_generation_token` → `agree_on_cancellations` → `mx_all_gather_tasks`.
Task control collectives therefore run during decode, despite old helper
comments saying they only ran before model forwards. MLX LM's decode loop
queues a lookahead token with `mx.async_eval` before yielding the current one.
The invalid integer has float32 bits corresponding to approximately 0.0055004,
consistent with a control/model payload mix-up, but this alone does not prove
how the original corruption occurred.

A separate recovery failure followed. New smbpt runners failed in
`multiprocessing.resource_sharer` with aborted connections, bad descriptors,
and EOF before model initialization. Recreating the model did not clear it.
Restarting only smbpt's parent EXO worker restored runner startup. Neither Mac
reported a thermal/performance warning during the incident.

## Changes

- Finish queued MLX LM generation and current-stream work before issuing
  task broadcasts, cancellation votes or barriers on the same distributed
  group. The model remains tensor-parallel over JACCL/RDMA.
- Transfer captured stdout/stderr as multiprocessing `Connection` handles.
  Spawn transfers their descriptors through its bootstrap instead of requiring
  a separate live resource-sharing server before the child can begin.
- Add an isolated regression test that makes the resource-sharing connection
  unavailable in spawned children, plus task-agreement coverage for pending
  model work. The old stdio code fails the regression; the new code passes.

## Validation scope

An isolated 2,000-round JACCL probe mixing asynchronous float collectives and
integer control collectives passed on the original implementation. It did not
reproduce the original corruption. The ordering fence closes an exposed hazard;
it is not proof that every native RDMA failure is eliminated.

The Python suite passed 1,264 tests, with 15 skipped and 212 deselected.
Basedpyright and Ruff pass; changed files pass Ruff formatting. Nix is not
installed on the operator machine.

The patched deployment passed a controlled fault test: after streamed generation
started, one known model child was terminated with SIGTERM. The request received
an explicit error; both ranks automatically reconnected, loaded and returned
ready after 40.39 seconds, retaining the instance. Parent EXO launchd PIDs were
unchanged. No manual instance recreation or parent restart was used during
this recovery test.

Sustained request, cancellation and concurrency acceptance is recorded separately
in the operator receipts; consult the final status rather than assuming a
started soak completed successfully.

## Cancellation race found by the initial soak

After six 1,024-token requests and a known-answer probe, the first explicit
cancellation closed its client stream but did not stop either model rank.
The readiness check timed out after 30 seconds; both ranks subsequently finished
all 1,024 tokens and returned ready. This was a failed cancellation acceptance,
not a permanent model hang. The failed run is preserved in `soak-status.json`.

The API sends `TaskCancelled`, closes its response, then cleanup sends
`TaskFinished`. Workers could consume the resulting Cancelled and Deleted
events before the planner's next tick, losing the cancellation intent. A
regression test reproduces that event ordering and fails on the old worker.
The fix schedules cancellation delivery when the ordered event is applied and
makes supervisor delivery idempotent so the planner cannot enqueue a duplicate.
This change does not depend on retaining a deleted task or delaying cleanup.

## Native runtime finding and candidate

The updated cancellation soak reproduced the native broadcast error again at
its second four-request burst (count `1039154603`). Python fences alone are
therefore insufficient. The runner rebuilt automatically after this real fault.

The deployed MLX was a custom `rltakashige/mlx-jaccl-fix-small-recv` build at
`cc3f3e60`, including altered CPU/GPU fence and RDMA mesh code. Its shipped
`Device`/`Stream` ordering operators contain the defect documented in upstream
[MLX issue 4083](https://github.com/ml-explore/mlx/issues/4083). A C++ probe
compiled against the installed headers returned both `cpu < gpu` and `gpu < cpu`
as true. The same probe against official MLX 0.32.4 returns a strict ordering.
This proves an affected native component, not that this comparator alone caused
the observed EXO header corruption.

Private APFS clones of the existing environments were prepared under
`~/.orchestraitor/deployments/twins-mlx-stable-20261031/venv` on both Macs.
Only MLX and mlx-metal change to 0.32.4; MLX LM stays 0.31.3. Entry-point shebangs
were relocated, the original environments remain untouched, and `uv run
--no-sync` was checked with `UV_PROJECT_ENVIRONMENT` selecting the candidate.
The candidate passes GPU arithmetic and 2,000-round JACCL tests. The deployment
snapshot selects the candidate environment and sets `EXO_FAST_SYNCH=false` to
use the standard CPU/GPU synchronization path. Full-model acceptance is required
before treating this candidate as the final runtime.

## Deployment and rollback

Only `src/exo/utils/async_process.py` and the control-collective changes in
`src/exo/worker/engines/mlx/utils_mlx.py` were overlaid on both Twins. The latter
was applied to each deployed baseline so unrelated source changes were not
introduced. Existing native artifacts, dependency environments and model weights
remain unchanged. The deployed MLX build is `0.32.0.dev20260507+cc3f3e60` and
MLX LM is 0.31.3; this is not an upgrade to a new native communication library.

On both Macs, backups, staged files and a receipt are under
`~/.orchestraitor/deployments/twins-stability-20261031/`.
`before/` preserves the two source files and prior deployment manifest.
The live deployment manifest records commit `2937a1a3`, before/after source
hashes and unchanged native hashes in `twins_stability_20261031`.
Both EXO workers were restarted under the user's explicit approval after the
old model instance was idle and removed; a fresh two-rank tensor instance was
then created. Studio and Spark services were not deliberately restarted.

Rollback must restore both source files and the saved manifest together, then
reload idle Twin workers and recreate the two-rank model. Do not mix arbitrary
source/native builds or change the transport to pipeline/TCP as a workaround.

Receipts on the operator: `~/.cache/exo/twins-stability-20261031/`, including
`fault-probe-result.json`, `soak-status.json` and `soak-requests.jsonl`.
The fault-probe script contains the one-time child PID used for that run and
must not be rerun unchanged. Task: `6ae678a63e53081e51971459`.
PR: https://github.com/team-wcv/exo/pull/60.

Both Macs are required for this sharded model. API upstream failover does not
provide single-Mac inference; a failed rank can interrupt its current request
while the model processes rebuild.
