# Control delivery and Responses failures

## Observed incident

The November 24 three-agent test exposed distinct failures. A Twins runner hit
a Metal command-buffer timeout; subsequent attempts self-restarted after local
`NodeGatheredInfo` events were not acknowledged within the configured 15-second
watchdog limit. The master then removed instances after a shard disappeared.
The client sometimes received an empty successful completion. A later attempt
failed even after the observed host load subsided. High host load alone does
not explain every failure, and these observations do not establish context
capacity as their cause.

Evidence is recorded in task `6b053e263e896bda5f645d9a` and the linked three-agent
plan `b7c44d90-0829-4807-a45b-6f766a8ec1e0`. The existing model settings remain
294,912 tokens on Spark and 524,288 on the Twins, with tensor/JACCL RDMA.

## Changes and limits

The native `Swarm` previously awaited each client operation inside its receive
loop. Zenoh publishers use blocking congestion control, so a pending publish
could also prevent that loop from forwarding incoming acknowledgements and
liveness events. It now forwards incoming events while the operation remains
pending. Client operations are still processed in order; no unbounded task
spawning, message dropping, watchdog disablement, or context change is added.
The regression holds an operation pending and verifies that an incoming
acknowledgement arrives before the operation is released. This closes a
reachable starvation path, not proof of the sole cause of every live failure.

The Responses streaming adapter now emits terminal `response.failed` with
structured error details when a runner reports `ErrorChunk`. It does not emit
`response.completed` afterward. The collected JSON adapter also returns an
explicit failed response rather than raising after streaming headers were sent
and leaving an empty HTTP 200. That collected path still uses HTTP 200 with
`status: failed` in the body; it preserves disconnect cancellation behavior.

## Deployment acceptance

Source changes and unit tests alone are not production acceptance. Build a
matching `exo_rs` wheel for each target Python/platform, compare the deployed
base files and dependency lock, and back up the old native package plus changed
Python files. Keep the source and native component as one deployment unit.

Before activating a candidate, confirm no live inference will be interrupted
and follow the existing scoped restart authorization/consensus. Preserve all
model files, context settings, RDMA placement, and unrelated services. Verify
the imported native path and hashes after restart, the four-node topology,
and a real request through `/twins/v1`. Then repeat the local-agent task, plan,
DM, General-channel and review-state checks. Observe control acknowledgements
through the run; a health endpoint or loaded model alone is insufficient.

Rollback restores both the saved Python files and native package under the
same service label, followed by the same health and inference checks.

## Validation commands

Use the repository's locked Python tooling and a matching native extension:

```bash
cargo test -p networking --lib
cargo fmt --all --check
python -m pytest src ops/team-wcv -q
basedpyright
ruff check .
```

The runtime deployment and post-deployment observations are recorded separately
in the task; this runbook does not imply a candidate has been activated.
