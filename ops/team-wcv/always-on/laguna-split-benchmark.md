# Matched Laguna split benchmark

This experiment compares Studio-only inference with Spark prefill followed by
Studio decode, using the same `pipenetwork/Laguna-S-2.1-MLX-4bit` 118B checkpoint
at revision `5544297f819d50330bc3616dd15cbc7edb598b2f`. It does not compare the
native Spark NVFP4/DFlash engine with Studio's native oQ5e engine, and does not
enable a production EXO placement.

## Storage and staging

On October 24, 2026, Studio's new external SSD was identified as a WD_BLACK
SN8100 2000GB, connected over PCIe, mounted ExFAT at `/Volumes/wc-ssd`.
The benchmark checkpoint is at:

```
/Volumes/wc-ssd/exo-benchmarks/laguna-4bit-5544297f
```

All 13 weight shards and four manifest metadata files passed SHA-256 verification
after copying. The old copy on `/Volumes/888` is preserved. The native Studio
oQ5e provider continues to use its internal-drive checkpoint; it was not moved
by this experiment. Filesystem formatting and existing SSD contents are unchanged.

Both hosts have isolated dependency copies and source under:

```
~/.orchestraitor/deployments/laguna-split-20261025-d6aed88e
```

The active staged entrypoint is `benchmark-loader-fix.py` (commit `b0ab8859`); its source is
`laguna_split_benchmark.py` in this directory. Source and resources were archived
from commit `d6aed88e`, with the entrypoint updated through `b0ab8859`. The cache fix is
commit `c10dbaf8`. The deployment contains a private bearer-token file; never
copy that token into documentation, command arguments, or a PR.

## Measurement controls

- Same checkpoint, tokenizer JSON, chat template, and exact prompt token IDs.
- EXO source-review corpus, with prepared 65,536- and 245,041-token prompts.
- Warm model weights, fresh prefix caches for every measured request.
- EXO's chunked prefill helper, 512-token chunks on both machines.
- Final prompt token consumed by Studio before greedy decoding, capped at 256
  generated tokens or EOS. Report actual output length and finish reason.
- Spark cache traffic uses the LAN endpoint `192.168.1.150:52425`.
- Record prefill compute, fetch/serialization/network overhead, cache ingestion,
  first-token latency, decode throughput, output text, and peak MLX memory.
- Any remote-cache error fails the run; there is no local-prefill fallback.

Laguna's sliding layers retain only 512 tokens while its full-attention layers
retain the complete prefix. The transport fix preserves chronological sliding
windows and logical offsets, including wrapped and incremental transfers.

The 37 transport regression tests pass (six unrelated cases deselected by the
repository test configuration), as do targeted strict type checking and Ruff.
An authenticated HTTP round trip on each host produced exactly the same next
eight tokens as local prefill using a tiny Laguna architecture. These checks
prove transport correctness for those cases, not full-model hardware performance.

## Maintenance boundary

The user authorized continuation on October 24. Both native engines are paused
under isolated maintenance controllers with restoration in cleanup. Do not load
the full benchmark alongside the native engines. The 64K baseline completed with
439.30 seconds to first token, 438.06 seconds in prefill, and 24.72 tokens/sec for
256 output tokens. The matched split run is in progress. Spark took 295.19
seconds for its first 16K tokens versus Studio's 59.78 seconds. This pass is
limited to the matched 64K comparison to bound the service interruption; the
prepared 245K prompt is not being benchmarked. A finish watcher requests both
controllers to restore the native providers once the 64K split receipt is saved.
Do not stop the Studio EXO master or the Twins to run this test.

For the approved window, record native PIDs/configuration, verify no active or
queued requests, then stop only `com.teamwcv.studio-laguna` and
`vllm-laguna-current-256k.service`. Ensure their children have exited before
loading the benchmark. Restore those exact services in cleanup, even if a
measurement fails, and verify `/studio/v1/models` and `/spark/v1/models` through
BigBrain. Spark's native reload previously took approximately eleven minutes.

The independent pre-test health check found BigBrain's Twins route already
returning HTTP 503. Both MacBook EXO processes remained alive after an
`EventDeliveryStalledError`, but neither API listened on port 52415; their
reconnect guards reported connection refusal. Studio's master was reachable
with no active instances. No service was stopped by benchmark preparation.
This was recorded as Orchestraitor report `report-1792906062967`. The user then
requested root-cause repair. Live Python stack sampling identified an exit-time
multiprocessing queue feeder join after its runner consumer had exited. The
fix was deployed on both Twins and merged as EXO PR #56. Both tensor/JACCL/RDMA
ranks are ready, and a real request through `/twins/v1` returned the expected
`TWINS_RECOVERY_OK` response in 1.72 seconds. The native Studio/Spark maintenance
remains separate from the restored Twins route.
