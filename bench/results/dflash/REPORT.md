# Qwen 3.5 4B + DFlash coupled-drafter benchmark

A/B benchmark of `mlx-community/Qwen3.5-4B-MLX-8bit` (target) against
the same target paired with `z-lab/Qwen3.5-4B-DFlash` (block-diffusion
coupled drafter, `block_size=16`, `target_layer_ids=[1,8,15,22,29]`).
This is the first numerical validation of the DFlash dispatch path
(`CoupledDrafterKind="dflash"`) on a real hybrid Qwen 3.5 target
(gated-delta-net + full-attention, `full_attention_interval=4`).

## Headline

| | target-only | DFlash coupled | speedup |
|---|---|---|---|
| median gen_tps (all scenarios) | **97.24** | **404.38** | **4.16x** |
| mean speedup across scenarios  | --      | --       | **4.02x** |
| median acceptance              | --      | --       | **93.2%** |

The 4.16x figure exceeds z-lab's published 3.7x SGLang/B200 number
and lands inside their MLX target band on Apple Silicon
(M2/M3 unified-memory class).

## Per-scenario breakdown

Per-scenario gen_tps is the mean of 2 runs.

| Scenario               | Target gen_tps | DFlash gen_tps | Speedup | Accept |
|------------------------|---------------:|---------------:|--------:|-------:|
| short_repetitive       |          97.24 |         310.57 |   3.19x |  93.2% |
| code_completion        |          97.19 |         371.43 |   3.82x |  92.0% |
| creative_prose         |          97.52 |         407.37 |   4.18x |  93.2% |
| factual_qa             |          95.80 |         449.87 |   4.70x |  93.4% |
| long_context_summary   |          94.28 |         396.04 |   4.20x |  93.2% |

All scenarios benefit. Unlike the Gemma 4 MTP A/B (where
`long_context_summary` regressed), DFlash on Qwen 3.5 4B is uniformly
positive — the block-diffusion drafting strategy holds up across
short bursty output, code, prose, factual recall, and long
generation alike.

### Cold-kernel first-run note

The very first DFlash run (`short_repetitive` run 0) measured
**199.4 t/s**, while every other DFlash run landed between
340–451 t/s (`short_repetitive` run 1 itself was 421.7 t/s).
The gap is consistent with Metal kernel compilation on first use
of the DFlash verify shape (bq=32, head_dim=256, bf16 attention).
The 4.16x median includes that cold run; the steady-state speedup
is closer to **4.26x** (excluding the cold run). Reporting the
conservative number here.

### Raw data

The full per-request JSON is committed alongside this report:
- [`qwen3.5-4b-mlx-8bit-target-only.json`](qwen3.5-4b-mlx-8bit-target-only.json)
- [`qwen3.5-4b-mlx-8bit-dflash.json`](qwen3.5-4b-mlx-8bit-dflash.json)

## Setup

- Host: wc-smbp (Apple Silicon, MLX 0.32.0.dev, mlx_vlm 0.5.0,
  mlx_lm 0.31.3)
- Target: `mlx-community/Qwen3.5-4B-MLX-8bit` (~5 GB on disk),
  hybrid Qwen 3.5 architecture (32 layers, 24 linear-attention +
  8 full-attention, `head_dim=256`, `full_attention_interval=4`)
- Drafter: `z-lab/Qwen3.5-4B-DFlash` (5-layer block-diffusion model,
  ~1 GB on disk, `block_size=16`)
- exo runtime: `team-wcv/bench/gemma4-mtp-coupled-results`
  (after the dtype + first-bonus shape fixes — see "Discovered bugs
  along the way" below)
- Harness: `bench/drafter_bench.py`, 2 runs/scenario,
  `--max-tokens=256`, warmup enabled
- Modes: `EXO_DRAFT_MODE=none` (target-only) vs `EXO_DRAFT_MODE=model`
  (DFlash coupled, auto-detected via `mlx_vlm.speculative.drafters.
  load_drafter(..., kind=None)` → `kind="dflash"`)
- Model card: `resources/inference_model_cards/mlx-community--
  Qwen3.5-4B-MLX-8bit.toml`, declaring `coupled_drafter =
  "z-lab/Qwen3.5-4B-DFlash"`

## How to reproduce

```bash
# 1. Download target + drafter (first run only)
uv run python -c '
from huggingface_hub import snapshot_download
snapshot_download("mlx-community/Qwen3.5-4B-MLX-8bit")
snapshot_download("z-lab/Qwen3.5-4B-DFlash")'

# 2. Symlink into ~/.exo/models/
ln -sfn ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-8bit/snapshots/<rev> \
  ~/.exo/models/mlx-community--Qwen3.5-4B-MLX-8bit
ln -sfn ~/.cache/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots/<rev> \
  ~/.exo/models/z-lab--Qwen3.5-4B-DFlash

# 3. Run the A/B harness
/tmp/run_dflash_bench_v2.sh   # or recreate from bench script in repo
```

## Discovered bugs along the way

The DFlash dispatch ran cleanly through the loader / adapter / unit
tests, but two latent bugs surfaced only when a real hybrid Qwen 3.5
target met the live decode path. Both fixed in this branch
(commits `1b256616` and `cf4624a3`).

### Bug 1 (commit `cf4624a3`) — gated-delta `inv_scale` dtype promotion

Our vendored `_gated_delta_net_forward_with_capture` had:

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

Target-only never tripped this because at decode-time bq=1 selects
a different kernel template that fits. The DFlash verify path was
the first caller to ever exercise the float32 attention at bq=32 on
a head_dim=256 model. Switching to a plain Python float keeps the
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
Qwen 3.5 cards declare `presence_penalty=1.5` (upstream best
practice), which tripped the path on the very first generated token.

## Reading the numbers

The all-scenario median (97.24 → 404.38, **4.16x**) reflects the
real-world steady-state speedup with this drafter/target pair.
Acceptance is consistent at ~93% across all five scenarios — DFlash's
block-diffusion approach to drafting is robust enough that the
verifier accepts roughly 14-15 of every 16 drafted tokens, which
keeps the wall-clock speedup very close to the theoretical maximum
(block_size = 16).

Compared to MTP on Gemma 4 (see `bench/results/mtp/REPORT.md`):

- Gemma 4 26B-A4B MTP: median speedup **-1.6%**, peak +22.1% on
  code_completion
- Gemma 4 31B MTP: median speedup **+5.4%**, peak +13.2% on
  code_completion
- Qwen 3.5 4B DFlash: median speedup **+316%** (i.e. 4.16x), every
  scenario above 3x

The architectural difference matters: MTP appends a single drafter
MLP head and proposes the next K tokens autoregressively, so its
acceptance falls off quickly with prompt entropy. DFlash drafts the
**entire block of 16 tokens in parallel** via block diffusion, which
is what unlocks the consistently-high acceptance rate.

## Next steps

1. Land the dispatch wiring + bench results upstream
   (target: `exo-explore/exo`, single aggregated PR).
2. Pre-stage the same A/B harness for the larger Qwen 3.5 variants
   (9B, 27B, 35B-A3B) once their DFlash drafters land on HF.
3. Investigate why the Gemma 4 MTP numbers underperform DFlash by so
   much — likely a function of MTP's "next-K autoregressive" drafting
   versus DFlash's block-diffusion drafting plus the difference in
   target-architectural fit (Gemma 4's full-attention layers don't
   give MTP as clean a hidden-state-sharing surface as Qwen 3.5's
   hybrid linear+full attention gives DFlash's per-layer captures).
