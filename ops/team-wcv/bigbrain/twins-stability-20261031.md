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
before treating this candidate as the final runtime. The first full-model
probe produced a coherent answer; the candidate took 6.14 seconds versus
3.91 seconds for the previous profile (78 versus 80 generated tokens; reported
decode rates 19.28 versus 40.26 tok/s). This is one short sample, not a general
benchmark. The conservative synchronization setting trades throughput for
removing the custom fast-fence path. Four explicit cancel-and-follow-up cycles
also passed; the longer native-runtime soak is still pending.

## Deployment and rollback

Four Python files were overlaid on the existing deployed baseline:
`src/exo/utils/async_process.py`, the control-collective sections of
`src/exo/worker/engines/mlx/utils_mlx.py`, `src/exo/worker/main.py`, and
`src/exo/worker/runner/supervisor.py`. The EXO Rust/native artifact and model
weights remain unchanged. The native-runtime candidate now selects MLX and
mlx-metal 0.32.4 in a private environment; the original development environment
is retained for rollback. MLX LM remains 0.31.3.

On both Macs, backups, staged files and a receipt are under
`~/.orchestraitor/deployments/twins-stability-20261031/`.
`before/` preserves the two source files and prior deployment manifest.
The live deployment manifest records commit `2937a1a3`, before/after source
hashes and unchanged native hashes in `twins_stability_20261031`.
The cancellation overlay has its own backups and receipt under
`twins-cancellation-20261031/` and records commit `01149ef7` in the manifest.
Both EXO workers were restarted under the user's explicit approval after the
old model instance was idle and removed; a fresh two-rank tensor instance was
then created. Studio and Spark services were not deliberately restarted.

Before activating the native candidate, each host's actual
`~/Library/LaunchAgents/com.teamwcv.exo.bigbrain.plist` is backed up to
`twins-mlx-stable-20261031/com.teamwcv.exo.bigbrain.before.plist`. Candidate
activation changes only `UV_PROJECT_ENVIRONMENT` and `EXO_FAST_SYNCH`; the
original environment is retained intact.

For a native-runtime rollback, wait for idle inference, remove the current
model placement, restore that host's saved launchd plist, and bootout/bootstrap
the same job so launchd actually reloads its environment. This explicitly
restores the prior environment selection and synchronization setting (removing
the new overrides when they were previously absent). Verify the loaded job's
settings and Python/MLX paths before recreating the tensor model. A kickstart
alone does not reload a changed plist. Wait for the unloaded job and old API
listener to disappear before bootstrap: an immediate bootstrap during teardown
returned launchctl error 5 during this rollout, then succeeded after teardown
completed. Hold a temporary caffeinate assertion through that transition.
The Python recovery fixes may remain
installed during a runtime-only rollback.

For a full rollback, additionally restore the four changed Python files from
`twins-cancellation-20261031/before/` and `twins-stability-20261031/before/`, and
restore the deployment manifest from the first overlay's backup. Restore the
saved launchd plist as above before restarting. Compare against current files
and preserve unrelated subsequent edits. Do not mix arbitrary source/native
builds or change tensor/RDMA to pipeline/TCP as a workaround.

Receipts on the operator: `~/.cache/exo/twins-stability-20261031/`, including
`fault-probe-result.json`, `soak-status.json` and `soak-requests.jsonl`.
The fault-probe script contains the one-time child PID used for that run and
must not be rerun unchanged. Task: `6ae678a63e53081e51971459`.
PR: https://github.com/team-wcv/exo/pull/60.

Both Macs are required for this sharded model. API upstream failover does not
provide single-Mac inference; a failed rank can interrupt its current request
while the model processes rebuild.
