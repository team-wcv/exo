# Persistent Studio and Spark inference

Verified October 24, 2026. This deployment uses the devices as two independent
118B coding/review workers while the Twins continue serving Qwen through EXO.
It does not transport KV cache between native vLLM and oMLX.

| Worker | Browser | Private API base | Runtime / model |
| --- | --- | --- | --- |
| Spark | `http://laguna.localhost:18888/` or [tailnet](https://bigbrain.taile43e67.ts.net/laguna/) | `https://bigbrain.taile43e67.ts.net/laguna/v1` | vLLM, Laguna S 2.1 NVFP4 118B + DFlash |
| Studio | `http://studio-ai.localhost:18888/` or [tailnet](https://bigbrain.taile43e67.ts.net/studio/) | `https://bigbrain.taile43e67.ts.net/studio/v1` | oMLX, Laguna S 2.1 oQ5e 118B |
| Twins | `http://bigbrain.localhost:18888/` or [tailnet](https://bigbrain.taile43e67.ts.net/) | `https://bigbrain.taile43e67.ts.net/v1` | EXO, Qwen3.8 Flash Next mixed 4/8-bit, RDMA |

Localhost aliases are installed on the operator Mac and Studio. Other clients
use the private tailnet URLs. Caddy on Studio listens on 18889, dispatching
`/laguna/` to Spark:8888, `/studio/` to loopback:1235, and the root to EXO:52415.
The `svc:bigbrain` route in `watch-gateway-edge` now targets
`http://host.docker.internal:18889`. Preserve the unrelated Watch Funnel route
to 18809. No inference route is advertised through Funnel.

The localhost aliases use the private HTTPS gateway because direct operator
access to Studio:18889 was unreachable. They must not be restored to that direct
upstream. Browser configuration/assets are served with `no-store`; readiness
checks the native health endpoint as well as the model list. Studio is only
shown ready after its pinned engine is loaded.

## Models and ownership

| Item | Pinned revision | Capacity |
| --- | --- | --- |
| Spark target `poolside/Laguna-S-2.1-NVFP4` | `826aacdf6d8b2699d4e367def6f17c83b06044c2` | 262,144 token context, one active sequence |
| Spark draft `poolside/Laguna-S-2.1-DFlash-NVFP4` | `b3b5921a900b9e0a1e27e50bdaeb480692a6d19b` | 7 speculative tokens |
| Studio `mlx-community/Laguna-S-2.1-oQ5e` | `70eb70f56f89ae90dfc415c35547bc31fe4a083a` | 65,536 token context, two active requests |
| Studio oMLX v0.7.2 | `23b671f64f0a64f532e38d692a6c480ea704418d` | Separate Python 3.13 environment |

Spark's actual cached target is approximately 92.85 GiB, plus 2.08 GiB draft;
do not size it using the model card's stale weight-size table. Studio's 15
weight shards total 78,138,371,517 bytes (72.77 GiB), verified against all
publisher LFS SHA256 values. The service-owned checkpoint is under
`~/.orchestraitor/deployments/always-on-inference-20261024/checkpoints/`, with
`models/laguna-studio` pointing there. It is an APFS clone of the downloaded
checkpoint, independent of LM Studio's library lifecycle.

Studio's idle LM Studio catalog failed to expose the completed MLX checkpoint
even after the approved restart. oMLX serves it successfully using the standard
MLX GPU kernels. Optional oMLX custom kernels were not built because Studio lacks
the Apple Metal Toolchain. This is not a dependency change to EXO's environment.

## Persistent startup and recovery

Spark uses `vllm-laguna-current-256k.service`, enabled with user lingering, and
`laguna-recovery.conf` as its `60-always-on.conf` drop-in. It retries on failure
after 15 seconds. The pinned launch script includes a 110 GiB available-memory
gate, FP8 KV, prefix caching, poolside tool/reasoning parsers, and FlashInfer.
Initial engine startup can take about ten minutes. The checked-in script/unit
are snapshots of the validated native deployment, not instructions to install
an arbitrary newer vLLM wheel into EXO.

Studio uses `~/Library/LaunchAgents/com.teamwcv.studio-laguna.plist`. It starts
after JJ logs in, keeps oMLX resident, restarts on exit, and holds a `caffeinate`
assertion. The loaded model is pinned with no idle eviction. The memory guard
is 100 GiB and the SSD cache is limited to 20 GB. The 64K context is a configured
ceiling, not a promise that every two-request 64K workload fits simultaneously.

The deployment root contains `omlx-venv`, `omlx-state`, `models`, `checkpoints`,
UI assets, current template snapshots, and a manifest. Use
`OMLX_WITH_CUSTOM_KERNEL=0` when installing `omlx-requirements.txt` into that
private venv on the current host. Settings templates omit generated secrets;
never overwrite an existing `omlx-state/settings.json` wholesale with them.
The listener is loopback-only. Caddy exposes `/v1/*` and `/health`, not oMLX's
administration endpoints.

Spark's EXO observer has `EXO_INFERENCE_DISABLED=true` in
`exo-spark-bigbrain.service.d/70-native-laguna.conf`. It publishes telemetry
and stays visible but advertises no inference backend. The worker planner
drains any local runner assigned while reserved. This avoids two independent
engines competing for Spark's unified memory. The unused Spark Qwen prefill
instance/link was removed; the Twins retain their existing RDMA instance.

Check the actual post-state after maintenance:

```sh
# Studio
launchctl print gui/$(id -u)/com.teamwcv.studio-laguna
curl -fsS http://127.0.0.1:1235/health
curl -fsS http://127.0.0.1:1235/v1/models
# Spark
systemctl --user show vllm-laguna-current-256k.service -p ActiveState -p MainPID -p Restart -p UnitFileState
curl -fsS http://127.0.0.1:8888/health
# Any authorized tailnet client
curl -fsS https://bigbrain.taile43e67.ts.net/studio/health
curl -fsS https://bigbrain.taile43e67.ts.net/laguna/v1/models
curl -fsS https://bigbrain.taile43e67.ts.net/state
```

Only the idle Studio LM Studio process and Spark EXO observer were approved for
restart during this deployment. Studio's EXO master, the Twins, and the native
Spark engine remained running. Future maintenance still follows the installed
restart protocol. No host reboot or deliberate inference-engine crash was
performed as an acceptance test; recovery settings were read back from the
loaded managers.

## Workflow integration and acceptance

Use Spark for long-context coding work and Studio for additional review,
summarization, and parallel tasks. These services create independent capacity;
they do not automatically consume the entire Orchestraitor queue or replace
the faster Twins route. Availability does not require artificial GPU activity.

CDX owns the additive `laguna-spark` and `laguna-studio` Codex profiles and model
catalog. Installed on the operator Mac and Studio, they preserve global cloud
defaults and existing permissions. Orchestraitor's provider list shows both
named providers. Its legacy fleet inference registry is separate and was not
changed. Use the profiles or the explicit OpenAI-compatible API bases above.

Both models passed real chat, function-tool/result, and Responses requests;
both passed streaming Responses function-call/result round trips. Studio also
served two concurrent 1,721-token coding prompts with 160 output tokens each,
finishing in 14.8 and 17.0 seconds. Spark served a 5,020-token coding prompt
with 256 output tokens: first content at 2.57 seconds, total 15.69 seconds.
These are serving checks with different workloads, not a matched speed or
model-quality comparison. Browser replies were verified through both each
service's localhost alias and HTTPS path; Studio's mobile layout also passed.

Native free-form custom tools are incompatible: Studio returned prose and
Spark mislabeled a custom call as a function call. The Codex profiles therefore
use JSON function tools and shell editing. Do not claim a complete autonomous
Codex coding evaluation from these protocol checks alone.

## Rollback

Backups are under each affected deployment's `before/` and the operator's
`~/.cache/exo/always-on-studio-spark-20261024/`. Compare current files before
restoring any complete configuration; preserve unrelated changes and symlinks.
To return Spark to EXO inference, first stop the native Laguna engine under
the restart protocol, remove only the reservation drop-in, restart its EXO
observer, and explicitly restore a compatible prefill placement/link. Never
run both large engines concurrently. The former Qwen prefill placement is not
restored simply by making Spark visible again.
