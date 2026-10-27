# Qwen prefill comparison — 23 October 2026

**Corrected 27 October 2026:** the accelerated split-route runs returned
`!!!!!!!!!!!!!!!!`, so their latency figures are invalid as useful-inference
benchmarks. The reference Spark and Twins receipts contain readable responses.
Do not infer a speedup or a verified route from cache-ingestion logs alone.
The kernel remains an explicit experiment and is disabled in the checked-in
fleet startup snapshot; normal Qwen serving uses local prefill on the Twins.
The cause of the accelerated output failure has not been established.

## Controlled request results

Checkpoint: `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit`, revision
`b2c422f3c643e36f04227a64d61796b44a4b1029`, 106,195,031,237 weight bytes.
The same checkpoint and tensor decoder placement were used throughout.

| Prefill path | ~1,600 input tokens, median total | ~6,400 input tokens, median total |
|---|---:|---:|
| Original Spark, chunk 256 | 27.76 s | 100.52 s |
| Spark fused CUDA recurrence, chunk 256 — **invalid output** | 19.61 s (rejected) | 69.61 s (rejected) |
| Spark fused CUDA recurrence, chunk 1,024 — **invalid output** | 19.68 s (rejected) | 70.15 s (rejected) |
| Twins locally over RDMA | **3.13 s** | **5.87 s** |

These historical timings are retained for diagnosis, not as acceptance of the
accelerated path. They are full request times, including 16 generated tokens,
measured through
`http://bigbrain.localhost:18888/bench/chat/completions`. There were three short
runs and two long runs per configuration, temperature 0, seed 1, thinking and
speculation disabled, and `use_prefix_cache=false`. A fresh leading nonce
prevented source prefix reuse. The second decode rank legitimately fetched the
same newly computed Spark cache. Both Mac ranks logged successful remote cache
ingestion on split-route runs; local runs used the same tensor placement with
the link's decode list temporarily empty.

The workload was an async-cache code review, repeated four times for the long
prompt. Baseline long-prompt pairs used identical prompt text. Other trials
changed the nonce, with one-token reporting differences between local/remote
prefill and nonce lengths. These are small-sample workload measurements, not
universal hardware rankings or a context-length limit test. One short Mac run
had a 9.45-second total despite fast model prefill; its median remains 3.13 s.
Median reported model prefill rates were approximately 61–64 tok/s on original
Spark, 88–93 tok/s on accelerated Spark, and 1,327 tok/s on the Twins.

## Cause and implementation

Spark's installed MLX LM 0.31.3 executes gated-delta attention as a Python loop
over tokens on CUDA; its optimized kernel is Metal-only. The opt-in
`gated_delta_cuda.py` uses MLX 0.32.0's custom CUDA API to keep the recurrent
matrix in registers across token steps. Gates retain the reference computation,
state accumulates in float32, and the output/cache dtypes match the CUDA
reference. Unsupported shapes, masked inputs and training requests use the
reference helper. Activation is restricted to `qwen4_exp` modules importing the
known MLX LM function, including trusted checkpoint modules absent from
`sys.modules`. Other models retain their existing path.

Standalone comparisons of outputs and final state covered float32, float16,
bfloat16, batches 1/2, grouped heads, and lengths 1/7/256/1,024. The real Spark
pytest run passed 13 numerical/fallback cases, including resumed state. The
local suite passed 1,238 tests (15 skipped, 212 deselected), including activation
and opt-in guards. Basedpyright and Ruff passed. These isolated kernel tests
do not establish full-model or transferred-cache
correctness. The invalid generated text supersedes the original speedup claim.

A first attempted activation did not find the unregistered checkpoint module;
those results are labeled `inactive-kernel-*` and excluded. An overlapping
request during reload required recreating the decoder placement with fresh
runner identities. The recovered placement retains the same checkpoint,
full-layer tensor layout and RDMA interfaces. Interrupted requests are excluded.

## Historical EXO operating choice and current safety gate

- The historical Spark EXO snapshot retains `EXO_PREFILL_CHUNK_SIZE=256`, but
  leaves `EXO_QWEN_CUDA_GATED_DELTA` unset. Explicit opt-in is for experiments
  only until full-model output and cache-transfer correctness are established.
- This exact model's custom card uses `prefill_eligible_nodes = []`. New
  placements therefore do not automatically force Spark prefill. Other model
  variants are not assigned this policy without measurements.
- The October 23 Spark source-only link has an empty decode list. Spark stays
  available as a prefill source and cannot accidentally receive decode traffic.
  An operator can explicitly link it to a compatible decoder for comparison.
- To restore split routing, PUT the existing source/decode instance IDs to
  `/v1/instance-links/{link_id}` using `prefill_instances` and `decode_instances`.
  Restore the model card's Spark eligibility only if automatic split placement
  is intended. Verify fresh nodes, ready runners, cache ingestion on both ranks,
  and useful output against the reference path with matched prompts.
- Before Spark maintenance, select local prefill and wait for all generation
  tasks to finish. A Ready runner status alone does not exclude pending work.
  Verify the new listener and sustained readiness before linking Spark again.

Runtime backups are under
`~/.orchestraitor/deployments/exo-cuda-prefill-20261023/` on Spark. The final
source overlay is recorded in the active deployment manifest. Disable the
opt-in environment flag and restart Spark to return to the original recurrence;
the Twins default does not depend on the CUDA kernel. This source merge does
not modify a running service. The [always-on runbook](../always-on/README.md)
describes the newer native Laguna allocation on Spark and Studio.

Raw results and memory samples are in
`/Users/JJ/.cache/exo/spark-prefill-validation/prefill-ab-20261023/` on the operator
machine, including `results-summary.json`. Review: team-wcv/exo PR #54.

The failed receipts include `short-spark-cuda256-1.json`,
`long-spark-cuda256-1.json`, and `long-spark-cuda1024-1.json`; inspect
`response.choices[0].message.content`, not just timing and token counters.
Re-enabling the optimization in a service requires fresh matched end-to-end
checks with useful output for both prompt sizes and every participating decoder.
