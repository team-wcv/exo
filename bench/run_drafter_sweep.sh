#!/bin/bash
# Orchestration helper for the drafter benchmark sweep.
#
# Runs on the operator's workstation; SSH's into the target host(s) to
# control exo lifecycle so we can sweep ``EXO_DRAFT_MODE`` across runs
# (the env var is read once per process).
#
# Usage (local single-host):
#     bash bench/run_drafter_sweep.sh local wc-smbp
#
# Usage (asymmetric twin):
#     bash bench/run_drafter_sweep.sh twin-tcp  wc-smbp wc-smbpt
#     bash bench/run_drafter_sweep.sh twin-rdma wc-smbp wc-smbpt
#
# Output: bench/results/drafter/<scenario>/<mode>.json
set -euo pipefail

SCENARIO="${1:?usage: run_drafter_sweep.sh <local|twin-tcp|twin-rdma> <host> [drafter_host]}"
TARGET_HOST="${2:?missing target host}"
DRAFTER_HOST="${3:-}"

MODEL="mlx-community/gemma-4-26b-a4b-it-bf16"
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/bench/results/drafter/${SCENARIO}"
mkdir -p "${RESULTS_DIR}"

REMOTE_REPO="/Users/JJ/Development/Tooling/exo"
UV_BIN_SMBP="/opt/homebrew/bin/uv"
UV_BIN_SMBPT="/Users/JJ/.local/bin/uv"

uv_for() {
    case "$1" in
        wc-smbp) echo "${UV_BIN_SMBP}" ;;
        wc-smbpt) echo "${UV_BIN_SMBPT}" ;;
        *) echo "uv" ;;
    esac
}

# Kill any stale exo processes on a host.
exo_kill() {
    local host="$1"
    ssh "${host}" "pkill -f 'exo.main' 2>/dev/null; pkill -f 'uv run exo' 2>/dev/null; sleep 2; pkill -9 -f 'exo.main' 2>/dev/null; true"
}

# Start exo in the background; returns when API is up.
exo_start() {
    local host="$1"
    local mode="$2"
    local extra_env="${3:-}"
    local uv_bin
    uv_bin="$(uv_for "${host}")"
    echo "[${host}] starting exo (EXO_DRAFT_MODE=${mode}, extra_env=${extra_env})..." >&2
    ssh "${host}" "cd ${REMOTE_REPO} && rm -f /tmp/exo-${mode}.log && nohup env EXO_DRAFT_MODE=${mode} ${extra_env} ${uv_bin} run exo -v >/tmp/exo-${mode}.log 2>&1 & disown; sleep 1; echo started"
    # Wait for API
    local tries=0
    until ssh "${host}" "curl -sf http://127.0.0.1:52415/v1/models >/dev/null 2>&1"; do
        tries=$((tries + 1))
        if [ "${tries}" -gt 60 ]; then
            echo "[${host}] exo failed to start within 120s; tail of log:" >&2
            ssh "${host}" "tail -40 /tmp/exo-${mode}.log" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "[${host}] exo API ready" >&2
}

# Place a single-host instance and wait for Ready.
place_instance() {
    local host="$1"
    local meta="${2:-MlxRing}"
    echo "[${host}] placing instance (meta=${meta}) for ${MODEL}..." >&2
    ssh "${host}" "curl -sf -X POST http://127.0.0.1:52415/place_instance \
        -H 'Content-Type: application/json' \
        -d '{\"model_id\":\"${MODEL}\",\"instance_meta\":\"${meta}\",\"min_nodes\":1}' \
        | head -c 400 ; echo"
    # Poll for instance ready
    local tries=0
    until ssh "${host}" "curl -sf http://127.0.0.1:52415/instance/placement 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(any((p.get(\"instance\",{}).get(\"shard_assignments\",{}).get(\"model_id\")==\"${MODEL}\") and p.get(\"phase\",{}).get(\"variant\")==\"Ready\" for p in d.get(\"placements\",[])))' 2>/dev/null | grep -q True"; do
        tries=$((tries + 1))
        if [ "${tries}" -gt 240 ]; then
            echo "[${host}] instance failed to reach Ready within 240s" >&2
            ssh "${host}" "tail -40 /tmp/exo-*.log" >&2 || true
            return 1
        fi
        sleep 5
    done
    echo "[${host}] instance Ready" >&2
}

run_bench() {
    local label="$1"
    local use_drafter="${2:-auto}"
    local out="${RESULTS_DIR}/${label}.json"
    echo "[bench] ${label} -> ${out}" >&2
    cd "$(dirname "$0")/.."
    /opt/homebrew/bin/uv run python bench/drafter_bench.py \
        --host "${TARGET_HOST_IP:-${TARGET_HOST}}" \
        --port 52415 \
        --model "${MODEL}" \
        --label "${label}" \
        --runs 2 \
        --warmup \
        --use-drafter "${use_drafter}" \
        --out "${out}" || true
}

case "${SCENARIO}" in
    local)
        # Resolve TARGET_HOST -> IP because curl/requests on this box go via LAN.
        TARGET_HOST_IP="$(ssh "${TARGET_HOST}" "ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en16 2>/dev/null" | head -n1)"
        : "${TARGET_HOST_IP:?could not resolve LAN IP for ${TARGET_HOST}}"
        export TARGET_HOST_IP
        echo "[local] sweeping draft modes on ${TARGET_HOST} (${TARGET_HOST_IP})" >&2

        for mode in none model ngram pipelined; do
            exo_kill "${TARGET_HOST}"
            sleep 3
            exo_start "${TARGET_HOST}" "${mode}"
            place_instance "${TARGET_HOST}" "MlxRing"
            run_bench "local-${mode}" auto
            exo_kill "${TARGET_HOST}"
            sleep 3
        done
        ;;
    twin-tcp|twin-rdma)
        : "${DRAFTER_HOST:?twin scenarios need a drafter host as third arg}"
        # Custom card with drafter_eligible_nodes installed *before* exo start.
        # Friendly names + asymmetric placement code use NodeId; we look up
        # the drafter NodeId via the master's REST API the first time exo
        # is up. Two-stage: bring up exo plain to read node IDs, then
        # rewrite the card and bring up exo with the asymmetric env.
        echo "twin scenarios are wired but require operator-driven node-id"
        echo "discovery; see bench/results/drafter/${SCENARIO}/README.md"
        echo "for the manual recipe."
        ;;
    *)
        echo "unknown scenario: ${SCENARIO}" >&2
        exit 2
        ;;
esac
