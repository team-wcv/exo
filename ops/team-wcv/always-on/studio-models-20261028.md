# Studio local model catalog — 28 October 2026

Studio's existing oMLX service now exposes three model IDs through
`https://bigbrain.taile43e67.ts.net/studio/v1`. The browser at that path's parent
and `http://studio-ai.localhost:18888/` offers the same three choices.

| Model ID | Publisher revision | Weight bytes | Configured context |
| --- | --- | ---: | ---: |
| `laguna-studio` | `mlx-community/Laguna-S-2.1-oQ5e` / `70eb70f56f89ae90dfc415c35547bc31fe4a083a` | 78,138,371,517 | 262,144 |
| `Muse-Glimmer-30B-4bit` | `mlx-community/Muse-Glimmer-30B-4bit` / `3e7677d7a40d348a3daba263a2b1c0aa41910710` | 19,414,804,113 | 131,072 |
| `Qwen3.8-27B-4bit` | `mlx-community/Qwen3.8-27B-4bit` / `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` | 16,054,541,349 | 262,144 |

The context values are configured ceilings, not proof of full-window acceptance.
New weight shards were checked against publisher LFS SHA256 values before
registration. Checkpoints are on Studio's internal SSD under
`~/.orchestraitor/deployments/always-on-inference-20261024/checkpoints/`.
The previously used `/Volumes/wc-ssd` was not mounted; no external disk was
formatted or modified. The discovery directory has new
`models/mlx-community/<checkpoint>` symlinks into those verified checkpoints.

## Residency and settings

The existing launchd job, oMLX environment, loopback listener, and Caddy routes
are retained. No service restart was required. Models were registered via the
installed downloader's completion callback, which rescans without stopping
loaded engines. The API and browser default remain `laguna-studio`.

Laguna is now unpinned, as are the two additions. The runtime can release an
idle model when another is requested, inside its existing 100 GiB guard. This
avoids requiring all three models to fit at once. Model switching can therefore
incur a cold load; availability does not mean every model is resident. The UI
says “Available · loads on request” and retains a separate history per model,
including in-memory histories if browser storage is full or unavailable.

Per-model settings were applied through oMLX's settings API and persisted in
`omlx-state/model_settings.json`. Muse uses a 4,096-token output default,
131,072 context and `reasoning_strength=low`; Qwen uses a 4,096-token output
default, 262,144 context and `reasoning_effort=low`. Clients can request Qwen's
`chat_template_kwargs.enable_thinking=false` for routine direct answers.
Neither new model uses speculative decoding in this deployment.

The existing runtime already includes both architectures: oMLX 0.7.2,
MLX 0.32.4, mlx-lm `0.31.4.dev132+g94cdcae13`, and mlx-vlm 0.7.8.
No dependency upgrade was needed. Spark and the Twins retain their services.

## Practical comparison

All three models selected task T2, identified blocked tasks T1/T3/T6, and
ignored a conflicting instruction inside task data. Both additions also returned the expected text through `/v1/responses`.
All three emitted the
expected `get_task` function with the exact task ID, then correctly summarized
a supplied tool result. This is a small protocol/task-guidance check, not a
coding benchmark or proof of autonomous agent behavior. Qwen's first thinking
response wrapped its JSON in Markdown; strict JSON consumers should request
structured output explicitly.

A second test explicitly disabled thinking for Qwen and Laguna. Each received
three equivalent triage prompts with a new leading nonce; every receipt reports
zero cached prompt tokens. Trial zero warms the model; the table summarizes
the two following requests. Both returned correct, unfenced JSON.

| Installed model | Median client completion | Median server first token | Median generation rate |
| --- | ---: | ---: | ---: |
| Laguna S 2.1 oQ5e | 2.17 s | 0.74 s | 40.78 tok/s |
| Qwen3.8 27B 4-bit | 4.33 s | 1.31 s | 32.17 tok/s |

These compare actual serving variants, not equal quantization. Laguna generated
51 output tokens and Qwen 79, so completion time includes their different
answer lengths. Prompts were approximately 216–220 tokens. Cold-load time is
excluded from those two warm requests: Laguna's preceding load took 27.88 s;
Qwen's earlier initial load took 5.44 s. Small sample, short prompts, one host:
do not extrapolate to long context, coding success, or concurrent throughput.

Muse's separate low-reasoning triage completed in 15.84 s client time, producing
411 tokens including 358 reasoning tokens at about 29.4 tok/s. It is not an
apples-to-apples direct-mode comparison. Its tool call and result requests took
5.26 s and 3.75 s; Qwen's took 3.78 s and 2.82 s; Laguna's took 1.44 s and 1.37 s.

Qwen's published Terminal-Bench 2.1 / SWE-bench Pro scores are 73.0 / 61.7,
versus Laguna's 70.2 / 59.4. These are separate publisher evaluations, not this
host's measurements. Qwen also supports vision and has a much smaller resident
footprint; Laguna's MoE activates roughly 8B of its 118B parameters per token,
so total parameter count does not predict per-token speed. Prefer Qwen when
Studio's memory headroom and general/vision assistance matter; retain Laguna
for its demonstrated fast loaded responses and as an alternative coding worker.

Sources: [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B),
[Poolside model card](https://huggingface.co/poolside/Laguna-S-2.1),
[Meta model card](https://huggingface.co/meta-models/Muse-Glimmer-30B).

## Acceptance, evidence and rollback

Browser checks passed on the private HTTPS path and localhost alias at mobile
width: all three selections, catalog availability, no JavaScript errors or
horizontal overflow. Offline browser fixtures additionally verify exact model
routing, isolated histories, busy controls, storage-quota failures, and the
unchanged single-model Spark page. The Python suite passed 1,258 tests, with
15 skipped and 212 deselected; type/lint and changed-file format checks passed.

Operator receipts and reproducible request bodies are under
`~/.cache/exo/studio-models-20261027/`. Deployment backups and the UI hash manifest
are on Studio under `~/.orchestraitor/deployments/studio-models-20261027/`.
The UI publishes versioned JS/CSS before replacing HTML; old assets remain for
cached pages. Studio's prior UI is in `ui-before/`, and `models-before.json`
records model settings before these changes.

Rollback is scoped: restore the prior Studio UI files from that backup, and
restore Laguna's pin through its settings API when idle memory capacity allows.
The new checkpoint files can remain on disk without being resident. Do not
replace global settings, change the Spark/Twins routes, or delete checkpoints
as part of a UI rollback. No host reboot or service restart was used to test
recovery; persistent configuration and discovery links were read back.
