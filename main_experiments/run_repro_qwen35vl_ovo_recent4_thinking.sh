#!/usr/bin/env bash
# OVO-Bench evaluation for Qwen3.5-0.8B thinking mode using the Qwen3-style
# cached-vision path: recent_frames_only=4, chunk_duration=1.0, fps=1.0,
# max_qa_tokens=1024.
#
# This uses main_experiments/eval_qwen35vl_ovo_thinking.py, which imports:
#   lib/recent_window_eval_qwen35_thinking.py
#
# Benchmark data must stay inside this repo:
#   data/ovo_bench/ovo_bench_new.json
#   data/ovo_bench/chunked_videos/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29591}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

OVO_ANNO_PATH="${REPO_ROOT}/data/ovo_bench/ovo_bench_new.json"
OVO_CHUNKED_DIR="${REPO_ROOT}/data/ovo_bench/chunked_videos"
OVO_RESULT_DIR="${REPO_ROOT}/main_experiments/results/repro_recent4/ovo_qwen35_08b_recent4_thinking"

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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "[INFO] Using PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Using ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[INFO] Results: ${OVO_RESULT_DIR}"
echo "[INFO] Implementation: lib/recent_window_eval_qwen35_thinking.py"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
"${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --multi_gpu \
    --mixed_precision bf16 \
    main_experiments/eval_qwen35vl_ovo_thinking.py \
    --model_path "Qwen/Qwen3.5-0.8B" \
    --anno_path "${OVO_ANNO_PATH}" \
    --chunked_dir "${OVO_CHUNKED_DIR}" \
    --result_dir "${OVO_RESULT_DIR}" \
    --recent_frames_only 4 \
    --chunk_duration 1.0 \
    --fps 1.0 \
    --max_qa_tokens 1024
