#!/usr/bin/env bash
# Safe OVO-Bench evaluation for SimpleStream:
# Qwen3-VL-2B-Instruct, recent_frames_only=4, chunk_duration=1.0, fps=1.0.
#
# This script only uses benchmark data inside this repo:
#   data/ovo_bench/ovo_bench_new.json
#   data/ovo_bench/chunked_videos/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29541}"

OVO_ANNO_PATH="${REPO_ROOT}/data/ovo_bench/ovo_bench_new.json"
OVO_CHUNKED_DIR="${REPO_ROOT}/data/ovo_bench/chunked_videos"
OVO_RESULT_DIR="${REPO_ROOT}/main_experiments/results/repro_recent4/ovo_qwen3vl_2b_recent4"

ensure_under_repo_data() {
    local path="$1"
    local resolved
    resolved="$(readlink -f "$path")"
    case "$resolved" in
        "${REPO_ROOT}/data/"*) ;;
        *)
            echo "[ERROR] Refusing to use data outside this repo: ${resolved}" >&2
            exit 2
            ;;
    esac
}

if [[ ! -f "${OVO_ANNO_PATH}" ]]; then
    echo "[ERROR] Missing OVO annotation: ${OVO_ANNO_PATH}" >&2
    exit 2
fi
if [[ ! -d "${OVO_CHUNKED_DIR}" ]]; then
    echo "[ERROR] Missing OVO chunked videos dir: ${OVO_CHUNKED_DIR}" >&2
    exit 2
fi
ensure_under_repo_data "${OVO_ANNO_PATH}"
ensure_under_repo_data "${OVO_CHUNKED_DIR}"

cd "${REPO_ROOT}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$(${ROCM_HOME:-/opt/rocm}/bin/rocm-smi --showid 2>/dev/null | awk '/GPU\[/{print NR-1}' | head -n "${NUM_PROCESSES}" | paste -sd, -)"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

echo "[INFO] Using PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Results: ${OVO_RESULT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m accelerate.commands.launch     --num_processes "${NUM_PROCESSES}"     --main_process_port "${MAIN_PROCESS_PORT}"     --multi_gpu     --mixed_precision bf16     main_experiments/eval_qwen3vl_ovo.py     --model_path "Qwen/Qwen3-VL-2B-Instruct"     --anno_path "${OVO_ANNO_PATH}"     --chunked_dir "${OVO_CHUNKED_DIR}"     --result_dir "${OVO_RESULT_DIR}"     --recent_frames_only 4     --chunk_duration 1.0     --fps 1.0     --max_qa_tokens 256
