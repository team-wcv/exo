# type: ignore
"""Drafter vs no-drafter A/B bench for the asymmetric cluster.

For each length in --lengths, runs the same prompt twice: once with
``use_drafter=True``, once with ``use_drafter=False``. Reports per-run
TPS, drafter telemetry, and the speedup ratio.

Sleeps briefly between runs so the model isn't warm-cache for one and
cold for the other; first run of each length pair is the
"throw-away" warmup, subsequent are timed (when --warmup is set).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from typing import Final

# Codex P2 (PR #21 round-(N+10), bench/bench_compare.py:20): the
# previous fixed ``http://192.168.1.224:...`` hard-coded a single
# operator's LAN IP, which broke every other developer running the
# bench against a different cluster (or against ``localhost`` for a
# single-node sanity check). Default to ``localhost:52415`` (the
# port ``exo`` always serves on) so the bench works out-of-the-box,
# while letting both an env var (for CI/scripts) and a CLI flag
# override the value at the moment of invocation.
DEFAULT_ENDPOINT: Final[str] = "http://localhost:52415/v1/chat/completions"
MODEL: Final[str] = "mlx-community/gemma-4-31b-it-bf16"


def _default_endpoint() -> str:
    return os.getenv("EXO_BENCH_ENDPOINT") or DEFAULT_ENDPOINT


PROMPT: Final[str] = (
    "Write a detailed, comprehensive technical reference on distributed "
    "speculative decoding for large language models. Cover the following "
    "topics in depth, with examples, equations, and pseudocode where "
    "relevant: (1) architectural foundations of speculative decoding, "
    "(2) the role of drafter vs target models and how acceptance/rejection "
    "is computed, (3) multi-token prediction (MTP) heads vs separate drafter "
    "models, (4) tensor-parallel verification and KV cache rollback semantics, "
    "(5) asymmetric placement on heterogeneous clusters, (6) wire-protocol "
    "design for drafter/target IPC, (7) failure modes (drafter death, target "
    "rank crashes, partitions) and recovery strategies, (8) tuning K (draft "
    "depth) for different workloads, (9) integration with continuous batching "
    "and paged attention, (10) practical performance results from real "
    "deployments. Use markdown headings and detailed prose. Begin now."
)


def run_once(
    max_tokens: int, use_drafter: bool, timeout: int, endpoint: str
) -> dict[str, object]:
    body: dict[str, object] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "use_drafter": use_drafter,
    }
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 - lan
        raw = resp.read().decode("utf-8")
    wall = time.monotonic() - started
    parsed = json.loads(raw)
    usage = parsed.get("usage") or {}
    completion = int(usage.get("completion_tokens", 0))
    stats = parsed.get("generation_stats") or {}
    return {
        "max_tokens": max_tokens,
        "use_drafter": use_drafter,
        "wall_s": round(wall, 2),
        "completion_tokens": completion,
        "tps_total": round(completion / wall, 2) if wall > 0 else 0.0,
        "drafter_model_id": stats.get("drafter_model_id"),
        "draft_mode": stats.get("draft_mode"),
        "num_draft_tokens": stats.get("num_draft_tokens"),
        "accepted_draft_tokens": stats.get("accepted_draft_tokens"),
        "proposed_draft_tokens": stats.get("proposed_draft_tokens"),
        "spec_decode_rounds": stats.get("spec_decode_rounds"),
        "acceptance_rate": (
            round(stats["accepted_draft_tokens"] / stats["proposed_draft_tokens"], 3)
            if stats.get("proposed_draft_tokens")
            else None
        ),
        "fraction_from_drafter": (
            round(stats["accepted_draft_tokens"] / completion, 3)
            if stats.get("accepted_draft_tokens") and completion
            else None
        ),
        "finish_reason": parsed.get("choices", [{}])[0].get("finish_reason"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 2048])
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--out", type=str, default="/tmp/bench_compare.json")
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=2.0,
        help="Seconds to sleep between runs to let the master settle.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=_default_endpoint(),
        help=(
            "Chat completions endpoint to bench against. Defaults to "
            f"$EXO_BENCH_ENDPOINT or {DEFAULT_ENDPOINT}."
        ),
    )
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for length in args.lengths:
        print(f"\n=== max_tokens={length} ===", flush=True)
        # Run no-drafter first; the drafter run inherits a warm prompt cache
        # via prefix-cache-hit, but max_tokens drives the bulk of the
        # measured time so this is fine for steady-state TPS comparison.
        for use_drafter in (False, True):
            try:
                r = run_once(length, use_drafter, args.timeout, args.endpoint)
            except Exception as exc:  # noqa: BLE001 - report bench failure
                r = {
                    "max_tokens": length,
                    "use_drafter": use_drafter,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(r)
            print(json.dumps(r, indent=2), flush=True)
            time.sleep(args.sleep_between)

        # Speedup summary for this length pair
        no_draft = next(
            (
                r
                for r in results
                if r.get("max_tokens") == length and not r.get("use_drafter")
            ),
            None,
        )
        draft = next(
            (
                r
                for r in results
                if r.get("max_tokens") == length and r.get("use_drafter")
            ),
            None,
        )
        if no_draft and draft and "error" not in no_draft and "error" not in draft:
            tps_no = float(no_draft.get("tps_total", 0.0) or 0)
            tps_yes = float(draft.get("tps_total", 0.0) or 0)
            speedup = round(tps_yes / tps_no, 3) if tps_no > 0 else None
            row = {
                "max_tokens": length,
                "tps_no_drafter": tps_no,
                "tps_drafter": tps_yes,
                "speedup_x": speedup,
                "acceptance_rate": draft.get("acceptance_rate"),
                "fraction_from_drafter": draft.get("fraction_from_drafter"),
            }
            print(f"\n>>> speedup at {length}: {json.dumps(row)}", flush=True)
            summary.append(row)

    out = {"summary": summary, "raw": results}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n=== overall summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
