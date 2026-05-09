# type: ignore
#!/usr/bin/env python3
"""Drafter A/B benchmark for exo.

Hits a running exo cluster's OpenAI-compatible API with a fixed prompt set and
captures per-request stats (prompt tps, generation tps, TTFT, drafter
acceptance). Used to compare drafter modes (`none`, `model`, `ngram`,
`pipelined`) and deployment topologies (single-host in-process vs.
asymmetric N+1 with the drafter on a separate node over jaccl/ring).

Usage:
    uv run python bench/drafter_bench.py \
        --host 127.0.0.1 --port 52415 \
        --model mlx-community/gemma-4-26b-a4b-it-bf16 \
        --label local-none --use-drafter false \
        --runs 3 --max-tokens 256 \
        --out /tmp/drafter_bench/local-none.json

The script does NOT manage exo lifecycle or placements -- start exo with the
desired EXO_DRAFT_MODE / drafter_eligible_nodes / backend config first, wait
for the model instance to be Ready, then point this at the API. Output is
JSON written to ``--out``.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROMPTS: dict[str, dict[str, str | int]] = {
    "short_repetitive": {
        "system": "You are a careful, concise assistant.",
        "user": (
            "Write a numbered list of 20 increasingly detailed bullet points "
            "describing the daily routine of a software engineer who works "
            "from home. Each bullet should reuse the phrase 'they then' to "
            "stitch ideas together so the output is highly repetitive."
        ),
        "max_tokens": 256,
    },
    "code_completion": {
        "system": "You are an expert Python typist.",
        "user": (
            "Implement an in-memory LRU cache class in Python with the "
            "following methods: __init__(capacity: int), get(key: str) -> "
            "Optional[Any], put(key: str, value: Any) -> None. Use only the "
            "standard library, include type hints, raise ValueError on "
            "non-positive capacity, and add a one-paragraph docstring "
            "explaining the cache invariants. Do NOT include tests."
        ),
        "max_tokens": 384,
    },
    "creative_prose": {
        "system": "You are a literary writer.",
        "user": (
            "Write a 350-400 word atmospheric short story set in an "
            "abandoned lighthouse on the night a comet passes Earth. Use "
            "vivid sensory detail and avoid clich\u00e9s. End on an "
            "unresolved image."
        ),
        "max_tokens": 512,
    },
    "factual_qa": {
        "system": "You are a precise factual assistant.",
        "user": (
            "Explain how Apple's unified memory architecture differs from "
            "discrete-GPU systems for large language model inference. "
            "Include three concrete numbers (memory bandwidth, capacity "
            "ranges, typical latency) with sources cited inline. Keep it "
            "under 250 words."
        ),
        "max_tokens": 384,
    },
}


@dataclass
class RequestStats:
    prompt_id: str
    run_index: int
    label: str
    use_drafter: bool | None
    num_draft_tokens: int | None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    accepted_draft_tokens: int = 0
    drafter_model_id: str | None = None
    draft_mode: str | None = None
    accept_fraction: float | None = None
    ttft_ms: float = 0.0
    wall_seconds: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkResult:
    label: str
    host: str
    port: int
    model: str
    use_drafter: bool | None
    num_draft_tokens: int | None
    runs: int
    requests: list[RequestStats] = field(default_factory=list)


def _now() -> float:
    return time.perf_counter()


def _post_chat(
    host: str,
    port: int,
    body: dict[str, Any],
    *,
    timeout: float,
) -> tuple[dict[str, Any], float, float]:
    """Send a streaming chat completion. Returns (final_payload, ttft_ms, wall_s).

    The exo bench endpoint ``/v1/chat/completions`` already returns the
    enriched ``BenchChatCompletionResponse`` with ``generation_stats`` in the
    final stream chunk. We use streaming purely so we can timestamp the
    first token; we still parse the final non-empty SSE event for stats.
    """
    body = dict(body)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    # ``/bench/chat/completions`` returns ``BenchChatCompletionResponse``,
    # which carries the ``generation_stats`` block we read for tps / accept
    # numbers. The standard ``/v1/chat/completions`` does not include them.
    payload = json.dumps(body).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request(
        "POST",
        "/bench/chat/completions",
        body=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    start = _now()
    ttft: float | None = None
    generation_stats: dict[str, Any] = {}
    last_chat_chunk: dict[str, Any] = {}
    try:
        resp = conn.getresponse()
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status} {resp.reason}: {resp.read(200)!r}")
        # ``HTTPResponse.fp.readline`` reads one chunk-encoded line at a
        # time and flushes immediately, so we get true streaming SSE
        # without urllib's read-ahead buffering. Each event ends in
        # ``\n\n`` so the body stream contains both header lines (data:
        # / : comment) and blank separator lines we ignore.
        while True:
            raw = resp.fp.readline()  # type: ignore[union-attr]
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            # ``generate_chat_stream`` emits ``: generation_stats {json}``
            # as an SSE comment immediately before the terminal ``[DONE]``.
            # Parse the comment so we can capture stats without falling
            # back to the non-streaming endpoint (which would lose TTFT).
            if line.startswith(": generation_stats"):
                payload_str = line[len(": generation_stats") :].strip()
                with contextlib.suppress(json.JSONDecodeError):
                    generation_stats = json.loads(payload_str)
                continue
            if not line.startswith("data:"):
                continue
            payload_str = line[len("data:") :].strip()
            if payload_str == "[DONE]":
                break
            try:
                chunk = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if ttft is None and any(
                (c.get("delta") or {}).get("content") for c in choices
            ):
                ttft = _now()
            last_chat_chunk = chunk
    finally:
        conn.close()
    wall = _now() - start
    ttft_ms = ((ttft - start) * 1000.0) if ttft is not None else 0.0
    out: dict[str, Any] = dict(last_chat_chunk)
    if generation_stats:
        out["generation_stats"] = generation_stats
    return out, ttft_ms, wall


def _run_one(
    *,
    host: str,
    port: int,
    model: str,
    prompt_id: str,
    prompt: dict[str, str | int],
    label: str,
    use_drafter: bool | None,
    num_draft_tokens: int | None,
    run_index: int,
    timeout: float,
    max_tokens_override: int | None,
) -> RequestStats:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "max_tokens": max_tokens_override or int(prompt["max_tokens"]),
        "temperature": 0.0,
    }
    if use_drafter is not None:
        body["use_drafter"] = use_drafter
    if num_draft_tokens is not None:
        body["num_draft_tokens"] = num_draft_tokens
    stats = RequestStats(
        prompt_id=prompt_id,
        run_index=run_index,
        label=label,
        use_drafter=use_drafter,
        num_draft_tokens=num_draft_tokens,
    )
    try:
        chunk, ttft_ms, wall = _post_chat(host, port, body, timeout=timeout)
    except Exception as exc:  # surface but don't crash the suite
        stats.error = f"{type(exc).__name__}: {exc}"
        return stats
    stats.ttft_ms = ttft_ms
    stats.wall_seconds = wall
    gen = chunk.get("generation_stats") or {}
    stats.prompt_tokens = int(gen.get("prompt_tokens") or 0)
    stats.generation_tokens = int(gen.get("generation_tokens") or 0)
    stats.prompt_tps = float(gen.get("prompt_tps") or 0.0)
    stats.generation_tps = float(gen.get("generation_tps") or 0.0)
    stats.accepted_draft_tokens = int(gen.get("accepted_draft_tokens") or 0)
    stats.drafter_model_id = gen.get("drafter_model_id")
    stats.draft_mode = gen.get("draft_mode")
    # Codex P2 (PR #19 round-(N+3), drafter_bench.py:259): n-gram
    # speculation has no ``drafter_model_id``, so gating
    # ``accept_fraction`` on it dropped acceptance telemetry for n-gram
    # runs entirely (always ``null``) and skewed A/B comparisons. Use
    # ``draft_mode`` as the gate instead -- both ``"model"`` and
    # ``"ngram"`` produce drafts whose acceptance fraction is
    # meaningful; ``"none"`` (and missing/legacy payloads with no
    # drafter id) stay null because no drafts ran.
    drafted = stats.draft_mode in {"model", "ngram"} or bool(stats.drafter_model_id)
    if drafted and stats.generation_tokens:
        stats.accept_fraction = stats.accepted_draft_tokens / stats.generation_tokens
    return stats


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=52415)
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, help="run label (used in output)")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument(
        "--use-drafter",
        choices=["true", "false", "auto"],
        default="auto",
        help=(
            "force per-request use_drafter override; 'auto' omits the field "
            "(model-card default applies)"
        ),
    )
    p.add_argument(
        "--num-draft-tokens",
        type=int,
        default=None,
        help="per-request K override (default: runner config)",
    )
    p.add_argument(
        "--prompts",
        nargs="*",
        default=list(PROMPTS.keys()),
        choices=list(PROMPTS.keys()),
    )
    p.add_argument(
        "--warmup",
        action="store_true",
        help="run one extra warm-up request before timed runs",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    use_drafter: bool | None
    if args.use_drafter == "true":
        use_drafter = True
    elif args.use_drafter == "false":
        use_drafter = False
    else:
        use_drafter = None

    result = BenchmarkResult(
        label=args.label,
        host=args.host,
        port=args.port,
        model=args.model,
        use_drafter=use_drafter,
        num_draft_tokens=args.num_draft_tokens,
        runs=args.runs,
    )

    if args.warmup:
        prompt_id = args.prompts[0]
        prompt = PROMPTS[prompt_id]
        print(f"[warmup] {prompt_id}", file=sys.stderr)
        _run_one(
            host=args.host,
            port=args.port,
            model=args.model,
            prompt_id=prompt_id,
            prompt=prompt,
            label=args.label,
            use_drafter=use_drafter,
            num_draft_tokens=args.num_draft_tokens,
            run_index=-1,
            timeout=args.timeout,
            max_tokens_override=args.max_tokens,
        )

    for prompt_id in args.prompts:
        prompt = PROMPTS[prompt_id]
        for run_index in range(args.runs):
            print(
                f"[{args.label}] {prompt_id} run={run_index + 1}/{args.runs}",
                file=sys.stderr,
            )
            stats = _run_one(
                host=args.host,
                port=args.port,
                model=args.model,
                prompt_id=prompt_id,
                prompt=prompt,
                label=args.label,
                use_drafter=use_drafter,
                num_draft_tokens=args.num_draft_tokens,
                run_index=run_index,
                timeout=args.timeout,
                max_tokens_override=args.max_tokens,
            )
            result.requests.append(stats)
            if stats.error:
                print(f"  ERROR: {stats.error}", file=sys.stderr)
            else:
                print(
                    f"  gen={stats.generation_tokens}t @ "
                    f"{stats.generation_tps:.2f}t/s ttft={stats.ttft_ms:.1f}ms "
                    f"draft_mode={stats.draft_mode} "
                    f"accept={stats.accept_fraction}",
                    file=sys.stderr,
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(asdict(result), f, indent=2)

    successful = [
        r for r in result.requests if r.error is None and r.generation_tokens > 0
    ]
    if successful:
        gen_tps_values = [r.generation_tps for r in successful]
        ttft_values = [r.ttft_ms for r in successful]
        print(
            f"[{args.label}] median gen_tps="
            f"{statistics.median(gen_tps_values):.2f} "
            f"median ttft={statistics.median(ttft_values):.1f}ms "
            f"runs={len(successful)}",
            file=sys.stderr,
        )

    return 0 if all(r.error is None for r in result.requests) else 1


if __name__ == "__main__":
    sys.exit(main())
