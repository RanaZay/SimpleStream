#!/usr/bin/env bash
# Safe StreamingBench evaluation for Qwen3.5-0.8B in thinking mode:
# top_k=0, recent_frames_only=4, chunk_duration=1.0, fps=1.0, max_qa_tokens=1024.
#
# This script only uses benchmark data inside this repo:
#   data/streamingbench/questions_real.json
#   data/streamingbench/videos/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"

SB_ANNO_PATH="${REPO_ROOT}/data/streamingbench/questions_real.json"
SB_VIDEO_DIR="${REPO_ROOT}/data/streamingbench/videos"
SB_RESULT_DIR="${REPO_ROOT}/main_experiments/results/repro_recent4/streamingbench_qwen35_08b_recent4_hf_thinking1024"

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

if [[ ! -f "${SB_ANNO_PATH}" ]]; then
    echo "[ERROR] Missing StreamingBench questions: ${SB_ANNO_PATH}" >&2
    exit 2
fi
if [[ ! -d "${SB_VIDEO_DIR}" ]]; then
    echo "[ERROR] Missing StreamingBench videos dir: ${SB_VIDEO_DIR}" >&2
    exit 2
fi
ensure_under_repo_data "${SB_ANNO_PATH}"
ensure_under_repo_data "${SB_VIDEO_DIR}"

cd "${REPO_ROOT}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="0"
fi

export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export QWEN35_ENABLE_THINKING="${QWEN35_ENABLE_THINKING:-1}"

echo "[INFO] Using PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Using ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[INFO] Using QWEN35_ENABLE_THINKING=${QWEN35_ENABLE_THINKING}"
echo "[INFO] Results: ${SB_RESULT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
QWEN35_ENABLE_THINKING="${QWEN35_ENABLE_THINKING}" \
"${PYTHON_BIN}" main_experiments/eval_qwen35vl_streamingbench.py \
    --anno-path "${SB_ANNO_PATH}" \
    --video-dir "${SB_VIDEO_DIR}" \
    --qa-model "Qwen/Qwen3.5-0.8B" \
    --qa-device auto \
    --top-k 0 \
    --recent-frames-only 4 \
    --chunk-duration 1.0 \
    --fps 1.0 \
    --max-qa-tokens 1024 \
    --output-dir "${SB_RESULT_DIR}"
