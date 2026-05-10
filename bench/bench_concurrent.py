# type: ignore
"""Concurrent overlapping spec-decode bench for the asymmetric cluster.

Fires N parallel chat-completions requests (each with the drafter
enabled) at the master and measures:
  - Per-request wall time, completion tokens, individual TPS
  - Aggregate cluster TPS (sum of per-request tokens / max wall)
  - Time-to-first-token spread

The point: validate that EXO_MAX_CONCURRENT_REQUESTS > 1 actually
overlaps spec-decode sessions correctly. Single-rank-target placements
trivially share KV; multi-rank tensor-parallel placements with an
asymmetric drafter are the interesting case here.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request
from typing import Final

# Codex P2 (PR #21 round-(N+10), bench/bench_concurrent.py:24): see
# the matching note in ``bench_compare.py``. Both bench scripts now
# resolve the endpoint via the same precedence chain (``--endpoint``
# flag > ``EXO_BENCH_ENDPOINT`` env var > ``localhost:52415``
# default) so they stay in sync and don't reintroduce a fixed LAN
# address.
DEFAULT_ENDPOINT: Final[str] = "http://localhost:52415/v1/chat/completions"
MODEL: Final[str] = "mlx-community/gemma-4-31b-it-bf16"


def _default_endpoint() -> str:
    return os.getenv("EXO_BENCH_ENDPOINT") or DEFAULT_ENDPOINT


PROMPTS: Final[list[str]] = [
    "Explain the architecture of distributed speculative decoding in "
    "one paragraph, then list six common failure modes with mitigations.",
    "Write a 400-word technical brief on tensor-parallel KV cache "
    "rollback semantics, including pseudocode for accept/reject.",
    "Outline the difference between MTP heads and external drafter "
    "models, and discuss when each is preferable for low-latency serving.",
    "Describe how an n-gram drafter integrates with a transformer "
    "target model, with attention to stateful processors and RNG.",
    "Summarize the trade-offs of pipelined vs synchronous spec-decode, "
    "including their interaction with continuous batching.",
    "Walk through the wire protocol of a drafter-target IPC channel "
    "designed for sub-millisecond round-trip on local sockets.",
    "Explain how acceptance probability is computed for vanilla "
    "speculative decoding and when greedy acceptance is sound.",
    "Discuss the engineering trade-offs between more drafter heads "
    "and more drafter depth for a fixed quality bar.",
]


def run_one(
    idx: int,
    prompt: str,
    max_tokens: int,
    timeout: int,
    results: list[dict[str, object]],
    started_at: float,
    endpoint: str,
) -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "use_drafter": True,
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    relative_start = time.monotonic() - started_at
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - lan
            raw = resp.read().decode("utf-8")
        wall = time.monotonic() - t0
        parsed = json.loads(raw)
        usage = parsed.get("usage", {})
        completion = int(usage.get("completion_tokens", 0))
        result: dict[str, object] = {
            "idx": idx,
            "relative_start_s": round(relative_start, 2),
            "wall_s": round(wall, 2),
            "completion_tokens": completion,
            "tps": round(completion / wall, 2) if wall > 0 else 0.0,
            "finish_reason": parsed.get("choices", [{}])[0].get("finish_reason"),
            "first_64": (
                parsed.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")[:64]
            ),
        }
    except Exception as exc:  # noqa: BLE001 - report bench failure
        wall = time.monotonic() - t0
        result = {
            "idx": idx,
            "relative_start_s": round(relative_start, 2),
            "wall_s": round(wall, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }
    results.append(result)
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", type=str, default="/tmp/bench_concurrent.json")
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

    n = args.concurrency
    prompts = (PROMPTS * ((n + len(PROMPTS) - 1) // len(PROMPTS)))[:n]

    results: list[dict[str, object]] = []
    started_at = time.monotonic()
    threads: list[threading.Thread] = []
    for i, p in enumerate(prompts):
        thread = threading.Thread(
            target=run_one,
            args=(
                i,
                p,
                args.max_tokens,
                args.timeout,
                results,
                started_at,
                args.endpoint,
            ),
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    total_wall = time.monotonic() - started_at

    completed = [r for r in results if "error" not in r]
    total_tokens = sum(int(r.get("completion_tokens", 0)) for r in completed)
    aggregate_tps = round(total_tokens / total_wall, 2) if total_wall > 0 else 0.0
    summary = {
        "concurrency": n,
        "max_tokens": args.max_tokens,
        "total_wall_s": round(total_wall, 2),
        "total_tokens_completed": total_tokens,
        "aggregate_tps": aggregate_tps,
        "successful": len(completed),
        "failed": n - len(completed),
        "individual": sorted(results, key=lambda r: int(r["idx"])),
    }
    print(json.dumps(summary, indent=2), flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
