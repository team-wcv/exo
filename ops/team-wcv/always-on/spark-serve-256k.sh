#!/usr/bin/env bash
set -euo pipefail

TARGET_SHA="826aacdf6d8b2699d4e367def6f17c83b06044c2"
DRAFT_SHA="b3b5921a900b9e0a1e27e50bdaeb480692a6d19b"
TARGET="$HOME/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/snapshots/$TARGET_SHA"
DRAFT="$HOME/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-DFlash-NVFP4/snapshots/$DRAFT_SHA"
VENV="$HOME/venvs/vllm025-current"
STATE_HOME="$HOME/.local/share/laguna-vllm-current"

test -x "$VENV/bin/vllm"
test -f "$TARGET/model.safetensors.index.json"
test -f "$DRAFT/model.safetensors"

if ss -H -lnt | awk '{print $4}' | grep -Eq '[:.]8888$'; then
  echo "port 8888 is already in use" >&2
  exit 1
fi

required_kib=$((110 * 1024 * 1024))
for attempt in $(seq 1 120); do
  available_kib="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"
  if [ "$available_kib" -ge "$required_kib" ]; then
    echo "memory gate passed: $((available_kib / 1024 / 1024)) GiB available"
    break
  fi
  if [ "$attempt" -eq 120 ]; then
    echo "memory gate failed: $((available_kib / 1024 / 1024)) GiB available; require 110 GiB" >&2
    exit 1
  fi
  sleep 5
done

export CUTE_DSL_ARCH=sm_121a
export MAX_JOBS=2
export NVCC_THREADS=1
export FLASHINFER_NVCC_THREADS=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_TRUST_REMOTE_CODE=1
export HF_HUB_OFFLINE=1
export PATH="/usr/local/cuda/bin:$PATH"
export TRITON_CACHE_DIR="$STATE_HOME/cache/triton"
export FLASHINFER_WORKSPACE_BASE="$STATE_HOME/cache/flashinfer"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

exec "$VENV/bin/vllm" serve "$TARGET" \
  --served-model-name poolside/Laguna-S-2.1-NVFP4 \
  --trust-remote-code \
  --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":7,\"method\":\"dflash\"}" \
  --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":20}' \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --attention-backend FLASHINFER \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":2}' \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.905 \
  --host 0.0.0.0 \
  --port 8888
