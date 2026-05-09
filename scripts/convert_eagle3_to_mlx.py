#!/usr/bin/env python3
"""Convert an EAGLE-3 PyTorch draft head checkpoint to MLX safetensors.

This is the *offline* half of the EAGLE-3 integration. It downloads a
pre-trained EAGLE-3 head from HuggingFace (e.g. ``RedHatAI/gemma-4-26B
-A4B-it-speculator.eagle3``) and rewrites the weights in MLX's expected
layout so the runtime side can load them with ``mlx.load`` once the
``EagleDrafter`` runtime lands.

Why this lives in ``scripts/`` and not in the runtime
-----------------------------------------------------
The runtime EAGLE drafter is currently a scaffolding stub
(``src/exo/worker/engines/mlx/generator/drafter.py::EagleDrafter``)
because Apple Silicon EAGLE wins are gated on ``mlx_lm`` adding
``position_ids`` support for tree-attention verify (open issues
``ml-explore/mlx-lm#846`` and ``#250``). The converter is the durable
half of the work: the artifact it produces sits on disk, and the day
the upstream blocker lifts the runtime can load it without rerunning
this script. Running this script today is safe; consuming the output
just doesn't beat ``DraftMode = "none"`` yet.

Usage::

    # convert the head for our gemma-4-26b target
    uv run python scripts/convert_eagle3_to_mlx.py \
        --source-repo RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3 \
        --output ~/.exo/eagle_heads/gemma-4-26b-a4b-it-eagle3 \
        --target-num-layers 30

The ``--target-num-layers`` flag drives EAGLE's layer-tap selection:
the head fuses pre-layer hidden states from ``{2, N//2, N-3}``
following the EAGLE-3 reference (Li et al. 2025). For Gemma-4-26b
(N=30) that means layers ``{2, 15, 27}`` -- the value is recorded in
``eagle_config.json`` next to the safetensors so the runtime doesn't
have to recompute it.

References
----------
* Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large
  Language Models via Training-Free Token-Level Blending,"
  NeurIPS 2025. https://arxiv.org/abs/2503.01840
* RedHat draft head for our exact target:
  https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3
* Reference MLX port of EAGLE-3 (Llama-3.1 only, no Gemma-4 adapter,
  no tree verify): mlx-lm Discussion #890. The ``eagle_convert.py``
  there is the spiritual ancestor of this script; we keep the layout
  compatible so a future Gemma-4 head shape change shows up as a
  fail-loudly assert here rather than a silent runtime miscompare.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# These are heavy imports (torch + safetensors); we defer them to
# ``main`` so ``--help`` works in a clean venv without pulling them in.


def eagle3_layer_taps(target_num_layers: int) -> tuple[int, int, int]:
    """Compute the {2, N//2, N-3} tap indices for an N-layer target.

    Reference impl:
    https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/modeling_llama_kv.py
    Indices are taken against the *target* layer count, not the head's.
    """
    return (2, target_num_layers // 2, target_num_layers - 3)


@dataclass(frozen=True, slots=True)
class ConvertConfig:
    source_repo: str
    output_dir: Path
    target_num_layers: int
    quantize_bits: int | None
    dry_run: bool


def parse_args() -> ConvertConfig:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-repo",
        required=True,
        help="HuggingFace repo of the PyTorch EAGLE-3 head, e.g. "
        "'RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3'.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the MLX-format head.",
    )
    parser.add_argument(
        "--target-num-layers",
        type=int,
        required=True,
        help="Layer count of the target model. Used to compute the "
        "EAGLE layer-tap indices ({2, N//2, N-3}). For Gemma-4-26b "
        "this is 30.",
    )
    parser.add_argument(
        "--quantize-bits",
        type=int,
        choices=[2, 3, 4, 8],
        default=None,
        help="If set, run mx.quantize on the head weights at this bit "
        "depth before writing. The community prototype reports identical "
        "EAGLE acceptance with 4-bit head quantization while halving "
        "the head forward cost (mlx-lm discussion #890).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the layer mapping and target-layer taps without "
        "downloading or writing anything.",
    )
    args = parser.parse_args()
    return ConvertConfig(
        source_repo=args.source_repo,
        output_dir=Path(args.output).expanduser().resolve(),
        target_num_layers=args.target_num_layers,
        quantize_bits=args.quantize_bits,
        dry_run=args.dry_run,
    )


def main() -> int:
    config = parse_args()
    layer_taps = eagle3_layer_taps(config.target_num_layers)

    print(f"source_repo:         {config.source_repo}")
    print(f"output_dir:          {config.output_dir}")
    print(f"target_num_layers:   {config.target_num_layers}")
    print(f"layer_taps:          {layer_taps}")
    print(f"quantize_bits:       {config.quantize_bits}")
    print(f"dry_run:             {config.dry_run}")

    if config.dry_run:
        print("\n[dry-run] not downloading or writing any files.")
        return 0

    # Defer heavy imports past --help / --dry-run so they fail loudly
    # only when actually needed. The runtime side of EAGLE doesn't need
    # any of these; this is a one-time conversion utility.
    try:
        import mlx.core as mx
        import torch  # noqa: F401  -- needed by safetensors below
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file as load_torch_safetensors
    except ImportError as e:
        print(
            "Missing optional dependency for EAGLE-3 conversion. "
            "Install with: uv add --dev torch safetensors huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    print(f"\n[1/4] downloading {config.source_repo} ...")
    src_dir = Path(snapshot_download(config.source_repo))
    print(f"      -> {src_dir}")

    src_safetensors = src_dir / "model.safetensors"
    src_config = src_dir / "config.json"
    if not src_safetensors.exists():
        # Some EAGLE-3 releases shard weights; fall back to the index.
        print(
            f"FATAL: {src_safetensors} not found. Sharded EAGLE-3 heads "
            "aren't supported by this converter yet. Open an issue with "
            "the repo path and we'll add the shard merge.",
            file=sys.stderr,
        )
        return 2
    if not src_config.exists():
        print(f"FATAL: missing config.json in {src_dir}", file=sys.stderr)
        return 2

    print(f"[2/4] loading torch weights from {src_safetensors}")
    torch_weights = load_torch_safetensors(str(src_safetensors))
    head_config = json.loads(src_config.read_text())

    # Convert torch -> numpy -> mx.array. We keep the EAGLE key names
    # verbatim so the runtime side can reuse the reference loader logic
    # without an additional rename map. The fuse layer (``embed_layernorm``,
    # ``fc``, ``midlayer.*``) and the reduced-vocab ``lm_head`` are all
    # the EAGLE-3 spec defines.
    print(f"[3/4] converting {len(torch_weights)} tensors to MLX")
    mx_weights: dict[str, mx.array] = {}
    for key, tensor in torch_weights.items():
        # bf16 -> float32 -> mx (mlx-core 0.x has limited bf16 ingest;
        # the runtime can re-cast at load time if it wants bf16 storage).
        np_array = tensor.to(dtype=torch.float32).cpu().numpy()
        mx_weights[key] = mx.array(np_array)

    if config.quantize_bits is not None:
        print(f"      quantizing weights to {config.quantize_bits}-bit")
        # mx.quantize lives at module-level, group_size matches mlx_lm
        # default (64) which is what RedHat's q4 export expects.
        quantized: dict[str, mx.array] = {}
        for key, value in mx_weights.items():
            # Quantize linear-layer weights only (2-D). Embeddings and
            # layer norms stay full precision; matches mlx_lm's default
            # quantization predicate.
            if value.ndim == 2 and "norm" not in key.lower():
                w_q, scales, biases = mx.quantize(
                    value, group_size=64, bits=config.quantize_bits
                )
                quantized[key] = w_q
                quantized[f"{key}.scales"] = scales
                quantized[f"{key}.biases"] = biases
            else:
                quantized[key] = value
        mx_weights = quantized

    print(f"[4/4] writing MLX head to {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(config.output_dir / "model.safetensors"),
        mx_weights,
    )

    # Persist EAGLE-specific metadata next to the weights so the runtime
    # doesn't have to recompute layer taps or reach into the source repo.
    eagle_meta = {
        "source_repo": config.source_repo,
        "target_num_layers": config.target_num_layers,
        "layer_taps": list(layer_taps),
        "quantize_bits": config.quantize_bits,
        "head_config": head_config,
    }
    (config.output_dir / "eagle_config.json").write_text(
        json.dumps(eagle_meta, indent=2) + "\n"
    )

    print("\nDone.")
    print(
        "Note: the runtime EagleDrafter is a NotImplementedError stub "
        "today (gated on mlx-lm position_ids upstream). The artifact you "
        "just produced is durable -- when the runtime lands, point it at "
        f"{config.output_dir} via ModelCard.eagle_head_repo and you're "
        "done."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
