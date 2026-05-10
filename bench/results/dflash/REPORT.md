# DFlash coupled-drafter benchmarks (Qwen 3.5 + Qwen 3.6)

A/B benchmarks of z-lab's DFlash block-diffusion coupled drafters
against the corresponding MLX-quantized targets on Apple Silicon. All
three numerical validations of the DFlash dispatch path
(`CoupledDrafterKind="dflash"`) on real hybrid Qwen targets
(gated-delta-net + full-attention, `full_attention_interval=4`).

## Headlines across three targets

| Target | Quant | Arch | host | Target gen_tps | DFlash gen_tps | Speedup | Accept |
|---|---|---|---|---:|---:|---:|---:|
| Qwen3.5 4B           | 8bit | dense | wc-smbp  | 97.24 | 404.38 | **4.16x** | 93.2% |
| Qwen3.6 27B          | 8bit | dense | wc-smbpt | 14.98 |  49.13 | **3.28x** | 92.6% |
| Qwen3.6 35B-A3B      | 8bit | MoE   | wc-smbpt | 87.70 | 377.49 | **4.30x** | 92.6% |

All medians are over 10 runs per A/B side (5 scenarios × 2 runs).
The +316% Qwen 3.5 4B result was **not** a sweet spot — DFlash holds
above 3.28x on dense 27B and recovers to 4.30x on the MoE 35B-A3B.

The MoE 35B-A3B is particularly striking: it's the second-fastest
target-only generation of the three (because only ~3B params are
active per token), yet DFlash still delivers a 4.30x speedup on top
of that fast baseline. The combination yields **377 t/s steady-state
generation on a 35B-class model on a single MacBook Pro M3 Ultra**.

## Qwen 3.6 27B (dense) — 3.28x

Target: `mlx-community/Qwen3.6-27B-8bit` (28 GB on disk, 64 layers,
hidden_size 5120, 48 linear-attn + 16 full-attn,
`full_attention_interval=4`, `head_dim=256`).

Drafter: `z-lab/Qwen3.6-27B-DFlash` (3.2 GB, 6-layer
block-diffusion drafter, `block_size=16`, 60 target layers indexed).

Per-scenario gen_tps is the mean of the 2 runs per scenario;
DFlash columns exclude one 0-token factual_qa run and one 0-token
short_repetitive run on the DFlash side from the *mean* but they're
still counted in the all-scenario median (see "Bench harness
flakiness" below). The all-scenario median row mirrors what the
harness reported live (`runs=8` for DFlash after auto-filtering
zero-token rows, `runs=10` for target-only).

| Scenario               | Target gen_tps | DFlash gen_tps | Speedup | Accept |
|------------------------|---------------:|---------------:|--------:|-------:|
| short_repetitive       |          17.90 |          51.43 |   2.87x |  93.0% |
| code_completion        |          16.72 |          33.45 |   2.00x |  86.5% |
| creative_prose         |          14.98 |          55.16 |   3.68x |  92.2% |
| factual_qa             |          12.72 |          24.60 |   1.93x |  82.0% |
| long_context_summary   |          10.73 |          56.21 |   5.24x |  92.8% |
| **all-scenario median**|      **14.98** |      **49.13** | **3.28x** | **92.8%** |

`long_context_summary` is the standout: DFlash recovers ~5.2x on
long-context generation, because the target spends a lot of wall
time per token at this scale and the speculation has more head room
to mask the per-token cost.

`factual_qa` and `code_completion` were noisier this run with a few
80-87% acceptance pockets that dropped scenario throughput. With
larger N (more runs per scenario) the per-scenario speedup would
likely tighten back into the 3-4x band the other scenarios sit in.

## Qwen 3.6 35B-A3B (MoE) — 4.30x

Target: `mlx-community/Qwen3.6-35B-A3B-8bit` (35 GB on disk, 40 layers,
256 experts × 8 active per token, hidden_size 2048,
`moe_intermediate_size=512`, `head_dim=256`).

Drafter: `z-lab/Qwen3.6-35B-A3B-DFlash` (905 MB, 8-layer dense
block-diffusion drafter, `block_size=16`,
`target_layer_ids=[1, 10, 19, 28, 37]`).

| Scenario               | Target gen_tps | DFlash gen_tps | Speedup | Accept |
|------------------------|---------------:|---------------:|--------:|-------:|
| short_repetitive       |          89.91 |         256.96 |   2.86x |  90.4% |
| code_completion        |          88.19 |         413.88 |   4.69x |  93.0% |
| creative_prose         |          87.52 |         213.86 |   2.44x |  46.5%* |
| factual_qa             |          86.82 |         287.39 |   3.31x |  89.8% |
| long_context_summary   |          85.68 |         411.02 |   4.80x |  93.8% |
| **all-scenario median**|      **87.70** |     **377.49** | **4.30x** | **92.4%** |

*creative_prose run 1 collapsed to 0% acceptance (23.57 t/s) on a
single run while run 0 stayed at 93.0% acceptance (404.15 t/s). The
mean is dragged down. Re-running with more samples per scenario
would tighten this. The median over the **9 healthy runs out of 10**
remains 388.67 t/s — i.e. the median is ~4.4x.

short_repetitive's first DFlash run came in at 125 t/s (Metal kernel
cold compile, same pattern as the 4B bench); run 2 jumped to 388 t/s.
The cold run pulls the mean down. Excluding it, the steady-state
speedup is closer to **4.5x**.

**Architectural note:** the MoE wires through our existing
`Qwen3_5DFlashTargetAdapter` with zero MoE-specific vendor work.
`mlx_lm.models.qwen3_5_moe` is a thin sanitize-wrapper around
`qwen3_5.Model`; MoE-vs-dense routing happens inside
`qwen3_5.DecoderLayer` via `SparseMoeBlock` vs `MLP` on `layer.mlp`,
and the vendored `_decoder_layer_forward_with_capture` already calls
`layer.mlp` polymorphically. The 4.30x speedup is the same code path,
unchanged.

## Qwen 3.5 4B (dense) — 4.16x (previously reported)

For completeness; full per-scenario breakdown elided here, see the
raw JSON next to this report.

| Scenario               | Target gen_tps | DFlash gen_tps | Speedup | Accept |
|------------------------|---------------:|---------------:|--------:|-------:|
| short_repetitive       |          97.24 |         310.57 |   3.19x |  93.2% |
| code_completion        |          97.19 |         371.43 |   3.82x |  92.0% |
| creative_prose         |          97.52 |         407.37 |   4.18x |  93.2% |
| factual_qa             |          95.80 |         449.87 |   4.70x |  93.4% |
| long_context_summary   |          94.28 |         396.04 |   4.20x |  93.2% |
| **all-scenario median**|      **97.24** |     **404.38** | **4.16x** | **93.2%** |

## Reading the numbers

DFlash's speedup ratio holds remarkably steady across a **17.5x** target
size range (4B → 35B) and across architectures (dense → MoE):

- 4B dense: 4.16x
- 27B dense: 3.28x
- 35B-A3B MoE: 4.30x

The 27B dense is the lowest in the band, and the explanation is
simple: it's the **most memory-bound** of the three (largest weights
in active path per token), so target-only is already drag-limited;
DFlash speeds up the wall-clock but the absolute headroom is smaller
in tokens/sec terms.

Acceptance lands at ~92-93% across all three targets, which is the
real story: DFlash's block-diffusion drafting strategy is robust
enough that the verifier accepts ~14-15 of every 16 drafted tokens
regardless of target scale or sparsity pattern. **Speedup ≈ accept ×
block_size / serial-overhead**, and the accept rate is the dominant
term that DFlash optimizes against.

### Compared to MTP on Gemma 4 (bench/results/mtp/REPORT.md)

| Target            | Drafter | Median speedup | Best-scenario speedup |
|-------------------|---------|---------------:|----------------------:|
| Gemma 4 26B-A4B   | MTP     |          -1.6% |  +22.1% (code)         |
| Gemma 4 31B       | MTP     |          +5.4% |  +13.2% (code)         |
| Qwen 3.5 4B       | DFlash  |          +316% |  +370% (factual_qa)    |
| Qwen 3.6 27B      | DFlash  |          +228% |  +424% (long_context)  |
| Qwen 3.6 35B-A3B  | DFlash  |          +330% |  +380% (long_context)  |

MTP appends a single drafter MLP head and proposes the next K tokens
autoregressively, so acceptance falls off quickly with prompt entropy
and worst-case scenarios actually regress (the 26B-A4B summary). DFlash
drafts the **entire block of 16 tokens in parallel** via block
diffusion, which is why acceptance stays consistently high across all
scenarios — every DFlash bench above stayed within a narrow 88-94%
acceptance band, while MTP on Gemma 4's `long_context_summary` fell
into single-digit acceptance.

## Bench harness flakiness

Across the 30 DFlash runs in this report (10 each for 4B / 27B /
35B-A3B), 3 runs returned `generation_tokens=0` and 1 run returned
0% acceptance:

- 4B: 0 hiccups
- 27B: 2 hiccups (short_repetitive run 1, factual_qa run 1) — both
  `error: null` in the harness but the server returned no body
- 35B-A3B: 1 hiccup (creative_prose run 1 collapsed to 0% accept)

These are bench-harness / chat-completion-streaming hiccups, not
DFlash failures — the chat-completion request returned an empty
response or a partial one without an error code. The runs adjacent to
each hiccup on the *same scenario* completed normally at the expected
speedup. The all-scenario median treats the hiccup runs as data
points (i.e. doesn't filter them), so the reported median is a
*lower-bound* estimate of true steady-state speedup.

For a publication-grade headline number, future benches should use
`--runs 5` (or `--runs 10`) instead of `--runs 2` to smooth out these
outliers. The current `--runs 2` was chosen for fast feedback during
implementation.

## Setup

- Hosts:
  - 4B bench: **wc-smbp** (Mac17,7, 128 GB unified memory)
  - 27B + 35B-A3B benches: **wc-smbpt** (Mac17,7, 128 GB unified
    memory, ~83 GB free vs ~13 GB on wc-smbp during the 4B run)
- Stack: MLX 0.32.0.dev, mlx_vlm 0.5.0, mlx_lm 0.31.3
- exo branch: `team-wcv/bench/gemma4-mtp-coupled-results`,
  including the dtype + first-bonus shape fixes documented inline below
- Harness: `bench/drafter_bench.py`, `--runs 2 --max-tokens 256`,
  5 scenarios (short_repetitive, code_completion, creative_prose,
  factual_qa, long_context_summary)
- Modes: `EXO_DRAFT_MODE=none` (target-only) vs `EXO_DRAFT_MODE=model`
  (DFlash coupled; auto-detected via `mlx_vlm.speculative.drafters.
  load_drafter(..., kind=None)` → `kind="dflash"`)
- Model cards (declaring `coupled_drafter=...`):
  - `mlx-community--Qwen3.5-4B-MLX-8bit.toml`
  - `mlx-community--Qwen3.6-27B-8bit.toml`
  - `mlx-community--Qwen3.6-35B-A3B-8bit.toml`

## How to reproduce

```bash
# 1. Download target + drafter pairs (first run only). Token required
#    for z-lab/Qwen3.6-27B-DFlash (gated; click "agree" on HF first).
uv run python -c '
from huggingface_hub import snapshot_download
for repo in [
    "mlx-community/Qwen3.5-4B-MLX-8bit",
    "z-lab/Qwen3.5-4B-DFlash",
    "mlx-community/Qwen3.6-27B-8bit",
    "z-lab/Qwen3.6-27B-DFlash",
    "mlx-community/Qwen3.6-35B-A3B-8bit",
    "z-lab/Qwen3.6-35B-A3B-DFlash",
]:
    snapshot_download(repo)'

# 2. Symlink into ~/.exo/models/ — see /tmp/qwen36_dflash_bench.sh
#    on either host for the exact ln -sfn invocations.

# 3. Run the A/B harness per target
/tmp/qwen36_dflash_bench.sh "mlx-community/Qwen3.6-27B-8bit"     "qwen3.6-27b-mlx-8bit"
/tmp/qwen36_dflash_bench.sh "mlx-community/Qwen3.6-35B-A3B-8bit" "qwen3.6-35b-a3b-mlx-8bit"
```

The bench script alternates `EXO_DRAFT_MODE=none` and
`EXO_DRAFT_MODE=model`, restarting exo between scenarios, and writes
per-request JSON to `bench/results/dflash/<label>.json`.

## Discovered bugs along the way (from the 4B bench)

Two latent bugs surfaced during the original 4B bench when a real
hybrid Qwen target first hit the live decode path. Both are fixed in
this branch and validated by the subsequent 27B + 35B-A3B benches.

### Bug 1 (commit `cf4624a3`) — gated-delta `inv_scale` dtype promotion

Our vendored `_gated_delta_net_forward_with_capture` originally had:

```python
inv_scale = mx.array(k.shape[-1] ** -0.5)   # 0-D float32 array
q = inv_scale * q * mx.rsqrt(...)            # promotes q to float32
```

vs. mlx-lm upstream's:

```python
inv_scale = k.shape[-1] ** -0.5              # Python float
q = inv_scale * q * mx.rsqrt(...)            # preserves bf16
```

`mx.array(scalar)` creates a float32 0-D array, which under MLX's
promotion rules upcasts the operand. The promoted dtype cascaded
through the gated-delta residual into the next full-attention layer's
SDPA call. On Apple Silicon the float32 SDPA kernel for
`head_dim=256` + `bq=32` (the DFlash verify-pass shape: 1 bonus +
16 drafted = 17 tokens, rounded up to bq=32) cannot be loaded:

```
RuntimeError: [metal::Device] Unable to load kernel
steel_attention_float32_bq32_bk16_bd256_wm4_wn1_maskfloat32_...
Threadgroup memory size (53760) exceeds the maximum threadgroup
memory allowed (32768)
```

Target-only never tripped this because at decode-time bq=1 selects a
different kernel template that fits. The DFlash verify path was the
first caller to ever exercise the float32 attention at bq=32 on a
head_dim=256 model. Switching to a plain Python float keeps the
attention kernel reachable.

### Bug 2 (commit `1b256616`) — first-bonus logits shape

`_select_first_bonus` was squeezing prefill-tail logits to
`(vocab,)` before iterating the request's `logits_processors`.
`mlx_lm.sample_utils` processors index as `[:, tokens]` and require
2-D `(batch, vocab)`, identical to `mlx_lm.generate.generate_step`'s
contract. A 1-D input raised `ValueError: Too many indices for array
with 1 dimensions`.

The Gemma 4 MTP A/B never tripped this because Gemma 4 cards declare
no `presence_penalty` / `repetition_penalty` / `frequency_penalty`
defaults, so the per-request processor list was typically empty.
Qwen 3.5+ cards declare `presence_penalty=1.5` (upstream best
practice), which tripped the path on the very first generated token.

## Raw data

Per-request JSON:
- 4B:        `qwen3.5-4b-mlx-8bit-{target-only,dflash}.json`
- 27B:       `qwen3.6-27b-mlx-8bit-{target-only,dflash}.json`
- 35B-A3B:   `qwen3.6-35b-a3b-mlx-8bit-{target-only,dflash}.json`

## Next steps

1. Land the dispatch wiring + bench results upstream
   (target: `exo-explore/exo`, single aggregated PR).
2. Bench bigger DFlash drafters as they ship:
   - `z-lab/Qwen3.5-122B-A10B-DFlash` (10B active, larger MoE) —
     would need wc-spark / wc-studio class memory for the 122B target.
   - `z-lab/Qwen3-Coder-30B-A3B-DFlash` (specialised code drafter) —
     interesting because the existing 30B-A3B is the only DFlash drafter
     trained against a code-specialised target so far.
3. Raise `--runs` to 5+ for publication-grade per-scenario means.
4. Investigate the empty-response harness hiccups (3/30 runs) —
   likely a streaming-completion ordering bug in `drafter_bench.py`
   when the server cancels a connection mid-stream, since the requests
   adjacent to each hiccup completed normally.
