# Spark and Twins local-agent verification: closeout

Recorded November 24–25, 2026. This is a dated acceptance record, not a live
health assertion. Re-read endpoint, task and deployment state before operating
the fleet.

## Outcome and scope

**Guided service-workflow pass.** Real Spark and Twins models used their own
authenticated agent identities to perform task, plan and messaging operations.
The agents ran their Codex tools on BMBP; inference ran on Spark or the Twins.
This does not sign off autonomous reliability, interactive latency, full-window
quality, recovery/failover, or the Assistant-panel product experience.
It is not an exhaustive qualification of every enabled tool.

| Surface | Recorded result |
| --- | --- |
| Spark | `poolside/Laguna-S-2.1-NVFP4`, `/spark/v1`, allocation 294,912 tokens |
| Twins | `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit`, `/twins/v1`, two tensor/JACCL RDMA ranks, allocation 524,288 tokens per rank |
| Task operations | Own-identity create/read/body update/claim and verified review handoff; both final fixtures remained `review`, `completed=false` |
| Plans | Both agents read the shared plan, saved comments under their own identities and read them back |
| Messaging | General posts with the human mention, coordinator DMs, idle wakes, and direct Twins-to-Spark peer review were exercised |
| Peer verdict | Spark's return was queued through the existing DM, then fetched and handled by Twins; it was not a reciprocal idle-wake test |

The original user Spark conversation was preserved. The two disposable test
launchers were refreshed to the merged CDX client and verified signed, online
and idle. Model capacities and tensor/RDMA transport were retained.

## Corrections and acceptance limits

- Spark initially called completion instead of review on an earlier fixture.
  That failure was retained; a fresh guided retry reached review correctly.
- Spark initially handled a human update request without posting a visible
  channel reply. A correction produced a verified reply to the original message.
- Twins corrected invalid DM arguments and twice stopped after announcing an
  action without executing it. Additional instructions produced the actual
  readback/report and handled receipts. Do not describe this as an unassisted pass.
- An optional Spark code review was stopped unfinished after about 15 minutes.
  Its earlier successful review of an EXO edge case is separate evidence.
- Native agent guidance was restored in launch-time catalog handling, but an
  improvement in model behavior was not established by this run.
- A later status check found a new active Twins instance with two ready ranks
  and an unassigned failed-runner record. Both endpoints completed short probes.
  Those probes do not establish unattended recovery or explain the old record.

Large read results and prompt-prefix changes remain material costs. One Twins
request needed about 338 seconds to prefill 47,523 tokens with 4,096 cached;
another needed about 418 seconds for 54,470 tokens with 4,096 cached. A combined
plan/task read grew subsequent input from 30,422 to 43,310 tokens. These are
individual observations, not a matched Spark-versus-Twins benchmark.

EXO's `consolidate_system_messages` in
`src/exo/worker/engines/mlx/utils_mlx.py` moves developer messages into the initial
system prompt. An active-delivery notice therefore changes that prefix. Preserve
instruction authority and template compatibility when investigating alternatives;
never reuse KV state for a different prompt just to improve a cache statistic.

## Merged repairs and release boundary

| Change | Scope |
| --- | --- |
| [EXO #68](https://github.com/team-wcv/exo/pull/68), `7574cc5` | Native receive progress during blocked publishing, ingress-close cancellation, and explicit inference/instance-deletion failures |
| [Messaiger #192](https://github.com/team-wcv/Messaiger/pull/192), `c411191` | Exact referenced-delivery selection and visible human-reply guidance |
| [CDX #14](https://github.com/team-wcv/cdx/pull/14), `d521191` | Tool discovery/name restoration, context metadata, native agent guidance, active executable retention across Homebrew cleanup, and aligned streaming/socket-read deadlines |

Recorded checks: EXO 1,314 Python tests and six Rust tests passed, with 15 skips
and 212 slow tests deselected; type/lint/format checks passed. CDX passed 14 native,
seven route and 90 shim tests. Messaiger passed 2,487 main tests with 72 skips,
nine license tests and the remaining local CI checks. Nix checks were not run.

The first Hub archive inspected was older despite a successful package publish.
After a manual staging catch-up, the advertised `0.2.545+0822820` archive was
downloaded and verified to contain the Messaiger fixes, with SHA-256
`350cc19e8999a4d1fded97e333df3cfade019932343fd97ef34863a5644c7e8e`.
This is historical provenance: resolve the current Hub manifest when upgrading.

At closeout, BMBP still had CLI `0.2.523` and override
`0.1.1-next.2+local.3b08f8d`. A served replacement does not mean that an installed
client or already-running MCP process has loaded it. Preserve the working pin
until the replacement bundle is verified and the managed unpin path succeeds.

## Follow-up ownership

Task IDs below are canonical Orchestraitor records. Follow their current state
and assignee; these rows do not dispatch work or change another agent's ownership.

| Work package | Task | Boundary / completion evidence |
| --- | --- | --- |
| BMBP update and pin retirement | `6b05fc523e896bda5f656832` | Verified Hub bundle installed, override retired through the managed updater, installed/bundled hashes and affected runtime behavior read back |
| Context/result-size and prefill optimization | `6b0501493e896bda5f641924` | Smaller safe service projections and measured cache/latency results, with full context capacity and permissions preserved |
| Model follow-through qualification | `6b062ee53256df30036118a7` | Repeated real task/plan/message cases, separating unassisted success from corrections, including visible replies, typed wake flags and post-compaction behavior |
| Recovery, soak and either-Twin fallback | `6b062ee43256df30036118a4` | Controlled scenario matrix, honest fallback identity/unavailability, restored tensor/RDMA and exact-once delivery evidence |
| Cross-surface product acceptance | `6adea1860f7dc49b2bb7990d` (LA-10) | Existing local-assistant program; Assistant-panel implementation stays with LA-08 `6adea1860f7dc49b2bb79907` |
| Hub artifact deployment acceptance | `6b0619ac3256df300360e874` | Existing release owner; [Hub #450](https://github.com/team-wcv/Hub/pull/450) is merged, but this workstream did not prove its first automatic publish-to-served cycle |
| Satellite repin acceptance | `6b0611bb3256df300360d71c` | Existing release owner; [Orchestraitor #1948](https://github.com/team-wcv/Orchestraitor/pull/1948) and publisher wiring merged; confirm credential prerequisites and first real event-driven repin in that task |

New verification tickets are unassigned and outside the pickup queue. Filing a
ticket does not authorize disruptive testing. Preserve the original user sessions
and unrelated services; follow the applicable maintenance/restart policy.

## Evidence and resumption

- Main verification task: `6b05093f3e896bda5f64206d`.
- Cohort report: `6b0539d53e896bda5f645619`.
- Completed test plan: `b7c44d90-0829-4807-a45b-6f766a8ec1e0`.
- Final Spark/Twins review fixtures: `6b05454e3e896bda5f646b42` and
  `6b05c8b33e896bda5f651c4b`. Preserve their requested review states and historical
  failures; do not complete or delete another actor's fixtures as cleanup.
- Local run evidence: `~/.orchestraitor/deployments/local-three-agent-test-20261124B/final-result.json`.
- Verified served Hub archive: `~/.orchestraitor/deployments/hub-published-messaiger-Cuvrad/verification.json`.
- Temporary MCP build and rollback provenance:
  `~/.orchestraitor/deployments/messaiger-reply-guidance-3b08f8d/verification.json`.
- [Control-delivery deployment and rollback runbook](control-delivery-20261124.md).

Keep publication, Hub availability, local installation, client reload and useful
agent behavior as distinct observations. Re-verify persisted state before
propagating a completion claim from this record or a compacted conversation.

