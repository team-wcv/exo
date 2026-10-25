"""Measure matched Laguna prefill locally and over EXO's binary cache transport.

This isolated experiment does not register a production EXO instance. Both sides
must use the pinned MLX checkpoint; the native NVFP4/oQ5e providers are different
quantizations and cannot be substituted into this comparison.
"""

import argparse
import gc
import hashlib
import hmac
import importlib.util
import json
import os
import socket
import sys
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import psutil
from mlx_lm.generate import generate_step
from mlx_lm.utils import load_model
from transformers import PreTrainedTokenizerFast

from exo.worker.disaggregated.protocol import (
    Done,
    Header,
    KVChunk,
    read_header,
    read_message,
    write_done,
    write_header,
)
from exo.worker.engines.mlx.disaggregated.adapter import (
    send_mlx_kv_cache,
    wire_dtype_from_cache,
)
from exo.worker.engines.mlx.disaggregated.client import (
    PrefillResult,
    ingest_into_mlx_cache,
)
from exo.worker.engines.mlx.disaggregated.serve import fill_cache_prefix

MODEL = "pipenetwork/Laguna-S-2.1-MLX-4bit"
REVISION = "5544297f819d50330bc3616dd15cbc7edb598b2f"
CODE_SHA = "0a9c99d894daf32d7324694acd9b29fc2b68bdd76ba6a6946564ccc507065c3a"


def record(event, **values):
    """Emit timestamped evidence without writing credentials or prompt bodies."""
    print(json.dumps({"event": event, "time": time.time(), **values}), flush=True)


def save_new(path, value):
    """Keep earlier measurements intact, including after an interrupted run."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)


def load_benchmark_tokenizer(path):
    """Read the exact tokenizer without instantiating an unrelated HF model config.

    Transformers' AutoTokenizer validates nested RoPE settings even though this
    experiment loads the MLX model independently. The checkpoint's tokenizer JSON
    and chat template suffice; no model configuration or installed package patch
    is needed.
    """
    config = json.loads((path / "tokenizer_config.json").read_text())
    special = {k: v for k, v in config.items() if k.endswith("_token")}
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"),
        chat_template=(path / "chat_template.jinja").read_text(),
        **special,
    )


def load_laguna(path, smoke=False):
    """Load inspected, pinned model code through MLX-LM's public class hook."""
    code = path / "laguna.py"
    if hashlib.sha256(code.read_bytes()).hexdigest() != CODE_SHA:
        raise ValueError("Laguna model code does not match the inspected revision")
    name = "mlx_lm.models.laguna_split_benchmark"
    spec = importlib.util.spec_from_file_location(name, code)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if smoke:
        mx.set_default_device(mx.cpu)
        mx.disable_compile()
        args = module.ModelArgs(
            model_type="laguna",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=128,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=64,
            rms_norm_eps=1e-6,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            decoder_sparse_step=1,
            norm_topk_prob=True,
            num_attention_heads_per_layer=[8, 12, 12, 12],
            sliding_window=8,
            layer_types=["full_attention"] + ["sliding_attention"] * 3,
        )
        mx.random.seed(7)
        model = module.Model(args)
        nn.quantize(
            model,
            group_size=64,
            bits=4,
            class_predicate=lambda p, m: hasattr(m, "to_quantized"),
        )
        mx.eval(model.parameters())
        return model
    config = json.loads((path / "config.json").read_text())
    if (
        config.get("num_hidden_layers") != 48
        or config.get("quantization", {}).get("bits") != 4
    ):
        raise ValueError("Expected the full 48-layer Laguna MLX 4-bit checkpoint")
    if len(list(path.glob("model-*-of-00013.safetensors"))) != 13:
        raise ValueError("Expected all 13 weight shards of the 118B checkpoint")
    if psutil.virtual_memory().available < 78 * 1024**3:
        raise RuntimeError(
            "Laguna benchmark needs at least 78 GiB available; unload the native model first"
        )
    mx.set_memory_limit(100 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    start = time.perf_counter()
    model, _ = load_model(
        path, strict=True, get_model_classes=lambda _: (module.Model, module.ModelArgs)
    )
    record(
        "model_loaded",
        model=MODEL,
        revision=REVISION,
        seconds=time.perf_counter() - start,
        active_bytes=mx.get_active_memory(),
    )
    return model


def prefill(model, tokens):
    """Use fresh caches and the production EXO chunked prefill helper."""
    cache = model.make_cache()
    start = time.perf_counter()
    for start_pos in range(0, len(tokens), 16384):
        fill_cache_prefix(model, cache, mx.array(tokens[start_pos : start_pos + 16384]))
        if len(tokens) > 16384:
            record(
                "prefill_progress",
                processed=min(start_pos + 16384, len(tokens)),
                total=len(tokens),
                elapsed=time.perf_counter() - start,
            )
    return cache, time.perf_counter() - start


def server_for(model, token, host, port):
    """Expose a serial, authenticated experiment endpoint on the chosen address."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            if self.path != "/prefill" or not hmac.compare_digest(
                self.headers.get("Authorization", ""), f"Bearer {token}"
            ):
                self.send_error(403)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4 * 1024 * 1024:
                self.send_error(413)
                return
            request = json.loads(self.rfile.read(length))
            tokens = request["tokens"]
            if (
                request.get("model") != MODEL
                or request.get("revision") != REVISION
                or not isinstance(tokens, list)
                or not 0 < len(tokens) <= 262144
                or any(type(t) is not int or t < 0 for t in tokens)
            ):
                self.send_error(400)
                return
            cache = None
            try:
                cache, seconds = prefill(model, tokens)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("X-Prefill-Seconds", str(seconds))
                self.send_header("Connection", "close")
                self.end_headers()
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                start = time.perf_counter()
                dtype = wire_dtype_from_cache(cache)
                write_header(
                    self.wfile,
                    Header(
                        request_id=request["id"],
                        model_id=MODEL,
                        num_layers=len(cache),
                        dtype=dtype,
                    ),
                )
                count = send_mlx_kv_cache(self.wfile, cache, dtype=dtype)
                write_done(self.wfile, count)
                record(
                    "prefill_sent",
                    request_id=request["id"],
                    tokens=count,
                    prefill_seconds=seconds,
                    send_seconds=time.perf_counter() - start,
                    peak_bytes=mx.get_peak_memory(),
                )
            finally:
                del cache
                gc.collect()
                mx.clear_cache()

    return HTTPServer((host, port), Handler)


def fetch_cache(endpoint, token, tokens, caches):
    """Fail the experiment on any transfer error; never silently prefill locally."""
    identity = str(uuid.uuid4())
    body = json.dumps(
        {"id": identity, "model": MODEL, "revision": REVISION, "tokens": tokens}
    ).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/prefill",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    start = time.perf_counter()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=3600) as stream:
        compute = float(stream.headers["X-Prefill-Seconds"])
        header = read_header(stream)
        if (
            header.request_id != identity
            or header.model_id != MODEL
            or header.start_pos != 0
        ):
            raise ValueError("Prefill response identity or starting position differs")
        result = PrefillResult(header=header)
        transferred = 0
        while True:
            message = read_message(stream)
            if isinstance(message, Done):
                result.total_tokens = message.total_tokens
                break
            if not isinstance(message, KVChunk):
                raise ValueError(
                    f"Expected Laguna KV state, received {type(message).__name__}"
                )
            result.kv_chunks.setdefault(message.layer_idx, []).append(message)
            transferred += len(message.keys) + len(message.values)
    fetched = time.perf_counter() - start
    ingest_start = time.perf_counter()
    imported = ingest_into_mlx_cache(result, caches)
    if imported != len(tokens):
        raise ValueError(f"Imported {imported} tokens, expected {len(tokens)}")
    return {
        "spark_prefill_seconds": compute,
        "fetch_seconds": fetched,
        "transport_overhead_seconds": fetched - compute,
        "ingest_seconds": time.perf_counter() - ingest_start,
        "cache_bytes": transferred,
        "imported_tokens": imported,
        "request_id": identity,
    }


def measure(model, tokens, output_tokens, endpoint=None, token=None, eos_ids=()):
    """Measure warm weights, cold prefix, and greedy decode to EOS or a fixed cap."""
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
    if endpoint:
        cache = model.make_cache()
        metrics = fetch_cache(endpoint, token, tokens[:-1], cache)
    else:
        cache, elapsed = prefill(model, tokens[:-1])
        metrics = {"local_prefill_seconds": elapsed}
    output = []
    first = None
    finish = "length"
    for value, _ in generate_step(
        mx.array(tokens[-1:]), model, max_tokens=output_tokens, prompt_cache=cache
    ):
        value = int(value)
        now = time.perf_counter()
        if first is None:
            first = now
        output.append(value)
        if value in eos_ids:
            finish = "eos"
            break
    end = time.perf_counter()
    metrics.update(
        {
            "route": "spark-studio" if endpoint else "studio-only",
            "prompt_tokens": len(tokens),
            "output_tokens": len(output),
            "ttft_seconds": first - start,
            "total_seconds": end - start,
            "decode_tokens_per_second": (len(output) - 1) / max(end - first, 1e-9),
            "finish_reason": finish,
            "output_ids": output,
            "peak_bytes": mx.get_peak_memory(),
            "prompt_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        }
    )
    del cache
    gc.collect()
    mx.clear_cache()
    return metrics


def main():
    """Run preparation, correctness smoke test, producer, or paired measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "smoke", "serve", "run"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=52425)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=65536)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()
    os.environ["EXO_PREFILL_CHUNK_SIZE"] = str(args.chunk)
    if args.mode == "prepare":
        tokenizer = load_benchmark_tokenizer(args.checkpoint)
        text = args.corpus.read_text()
        marker = "__BENCHMARK_SOURCE_BUNDLE__"
        rendered = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Review this EXO source bundle for safe remote prefill and decoding.\n"
                    + marker
                    + "\nExplain how the attention cache transfer works, its correctness checks, "
                    "and likely performance bottlenecks. Give a detailed engineering review with concrete examples from the code.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        before, after = rendered.split(marker)
        prefix = tokenizer.encode(before, add_special_tokens=False)
        suffix = tokenizer.encode(after, add_special_tokens=False)
        body = tokenizer.encode(text, add_special_tokens=False)
        needed = args.prompt_tokens - len(prefix) - len(suffix)
        if not 0 < needed <= len(body):
            raise ValueError(f"Corpus contains {len(body)} tokens, need {needed}")
        save_new(
            args.output,
            {
                "model": MODEL,
                "revision": REVISION,
                "tokens": prefix + body[:needed] + suffix,
            },
        )
        record("prompt_prepared", tokens=args.prompt_tokens)
        return
    model = load_laguna(args.checkpoint, smoke=args.mode == "smoke")
    if args.mode == "smoke":
        os.environ["EXO_PREFILL_CHUNK_SIZE"] = "7"
        server = server_for(model, "local-correctness-only", "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            tokens = [i % 64 for i in range(68)]
            baseline = measure(model, tokens, 8)
            split = measure(
                model,
                tokens,
                8,
                f"http://127.0.0.1:{server.server_port}",
                "local-correctness-only",
            )
            if baseline["output_ids"] != split["output_ids"]:
                raise AssertionError("Imported cache changes greedy decode output")
            record(
                "smoke_pass",
                baseline=baseline,
                split=split,
                warning="Tiny architecture correctness test; not a hardware benchmark",
            )
        finally:
            server.shutdown()
            server.server_close()
        return
    warm, _ = prefill(model, [1] * 1024)
    del warm
    gc.collect()
    mx.clear_cache()
    record("warmup_done")
    token = args.token_file.read_text().strip() if args.token_file else None
    if args.mode == "serve":
        if not token or len(token) < 32:
            raise ValueError(
                "Producer requires a private random token of at least 32 characters"
            )
        server = server_for(model, token, args.host, args.port)
        record("ready", address=args.host, port=args.port)
        server.serve_forever()
    else:
        prompt = json.loads(args.prompt_file.read_text())
        if (prompt["model"], prompt["revision"]) != (MODEL, REVISION):
            raise ValueError("Prompt was prepared for a different checkpoint")
        tokenizer = load_benchmark_tokenizer(args.checkpoint)
        result = measure(
            model,
            prompt["tokens"],
            args.output_tokens,
            args.endpoint,
            token,
            {tokenizer.eos_token_id},
        )
        result.update(
            {
                "model": MODEL,
                "revision": REVISION,
                "chunk": args.chunk,
                "output_text": tokenizer.decode(result["output_ids"]),
            }
        )
        save_new(args.output, result)
        record(
            "measurement_saved",
            path=str(args.output),
            **{
                k: v
                for k, v in result.items()
                if k not in ("output_ids", "output_text")
            },
        )


if __name__ == "__main__":
    main()
