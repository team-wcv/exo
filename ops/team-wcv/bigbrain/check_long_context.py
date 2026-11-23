"""Run one explicit retrieval request while monitoring both Twins' headroom."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp


async def run(base: str, request: Path, expected: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    body = json.loads(request.read_text())
    metadata = json.loads(expected.read_text())
    started = time.monotonic()
    result = {"input": metadata, "text": "", "progress": {}, "usage": None}
    baseline_swap: dict[str, int] = {}
    minimum_free: dict[str, float] = {}
    timeout = aiohttp.ClientTimeout(total=7200, sock_connect=10, sock_read=180)
    with (output / "events.jsonl").open("x") as events:
        async with aiohttp.ClientSession(timeout=timeout) as session:

            async def memory() -> None:
                async with session.get(
                    base + "/state", timeout=aiohttp.ClientTimeout(total=8)
                ) as response:
                    response.raise_for_status()
                    state = await response.json()
                sample = {}
                for node_id, identity in state["nodeIdentities"].items():
                    name = identity["friendlyName"]
                    if name not in ("wc-smbp", "wc-smbpt"):
                        continue
                    node = state["nodeMemory"][node_id]
                    free = node["ramAvailable"]["inBytes"] / 1024**3
                    swap = (
                        node["swapTotal"]["inBytes"] - node["swapAvailable"]["inBytes"]
                    )
                    baseline_swap.setdefault(name, swap)
                    minimum_free[name] = min(minimum_free.get(name, free), free)
                    sample[name] = {
                        "free_gib": round(free, 2),
                        "new_swap_gib": round(
                            (swap - baseline_swap[name]) / 1024**3, 2
                        ),
                    }
                    if free < 16 or swap - baseline_swap[name] > 2 * 1024**3:
                        raise RuntimeError(
                            f"Memory guard stopped the request: {name} {sample[name]}"
                        )
                if len(sample) != 2:
                    raise RuntimeError("Both Twins must report memory for this test")
                report = {
                    "elapsed_s": round(time.monotonic() - started, 1),
                    "memory": sample,
                    "progress": result["progress"],
                }
                events.write(json.dumps(report) + "\n")
                events.flush()
                print(json.dumps(report), flush=True)

            async def generate() -> None:
                async with session.post(
                    base + "/v1/chat/completions", json=body
                ) as response:
                    response.raise_for_status()
                    async for raw in response.content:
                        line = raw.decode().strip()
                        if line.startswith(": prefill_progress "):
                            result["progress"] = json.loads(
                                line[len(": prefill_progress ") :]
                            )
                        elif line.startswith("data:"):
                            content = line[5:].strip()
                            if content == "[DONE]":
                                result["stream_completed"] = True
                                return
                            chunk = json.loads(content)
                            if "error" in chunk:
                                raise RuntimeError(str(chunk["error"]))
                            for choice in chunk.get("choices", []):
                                text = choice.get("delta", {}).get("content") or ""
                                if text and "first_token_s" not in result:
                                    result["first_token_s"] = round(
                                        time.monotonic() - started, 2
                                    )
                                result["text"] += text
                            if chunk.get("usage"):
                                result["usage"] = chunk["usage"]
                    raise RuntimeError("Stream ended without a completion marker")

            task = None
            try:
                await memory()
                task = asyncio.create_task(generate())
                while not task.done():
                    await asyncio.wait({task}, timeout=10)
                    await memory()
                await task
                text = (
                    result["text"]
                    .strip()
                    .removeprefix("```json")
                    .removesuffix("```")
                    .strip()
                )
                result["retrieval_passed"] = json.loads(text) == metadata["expected"]
            except (Exception, asyncio.CancelledError) as error:
                result["error"] = str(error) or type(error).__name__
                result["retrieval_passed"] = False
                if task and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            finally:
                result["elapsed_s"] = round(time.monotonic() - started, 2)
                result["minimum_free_gib"] = minimum_free
                with (output / "result.json").open("x") as receipt:
                    json.dump(result, receipt, indent=2)
                print(json.dumps(result), flush=True)
    if not result.get("retrieval_passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(run(args.base.rstrip("/"), args.request, args.expected, args.output))
