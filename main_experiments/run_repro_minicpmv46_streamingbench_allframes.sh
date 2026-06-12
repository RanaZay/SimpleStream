#!/usr/bin/env bash
# StreamingBench evaluation for MiniCPM-V-4.6 using all 1 FPS frames up to
# each question timestamp.
#
# This is not the SimpleStream recent-4 recipe. It evaluates the full causal
# video prefix for each question unless ALLFRAMES_CONTEXT_TIME is set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE:-16x}"
MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS:-1}"
MINICPM_PROFILE_COMPONENTS="${MINICPM_PROFILE_COMPONENTS:-1}"
MINICPM_AUTO_MAX_MEMORY_PER_GPU="${MINICPM_AUTO_MAX_MEMORY_PER_GPU:-60GiB}"
ALLFRAMES_CONTEXT_TIME="${ALLFRAMES_CONTEXT_TIME:--1}"

SB_ANNO_PATH="${REPO_ROOT}/data/streamingbench/questions_real.json"
SB_VIDEO_DIR="${REPO_ROOT}/data/streamingbench/videos"
SB_RESULT_DIR="${REPO_ROOT}/main_experiments/results/repro_allframes/streamingbench_minicpmv46_allframes_fps1"

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

echo "[INFO] Using PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Using ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[INFO] Using MINICPM_DOWNSAMPLE_MODE=${MINICPM_DOWNSAMPLE_MODE}"
echo "[INFO] Using MINICPM_MAX_SLICE_NUMS=${MINICPM_MAX_SLICE_NUMS}"
echo "[INFO] Using MINICPM_PROFILE_COMPONENTS=${MINICPM_PROFILE_COMPONENTS}"
echo "[INFO] Using MINICPM_AUTO_MAX_MEMORY_PER_GPU=${MINICPM_AUTO_MAX_MEMORY_PER_GPU}"
echo "[INFO] Frame selection: all frames at 1 FPS"
echo "[INFO] ALLFRAMES_CONTEXT_TIME=${ALLFRAMES_CONTEXT_TIME}"
echo "[INFO] Results: ${SB_RESULT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE}" \
MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS}" \
MINICPM_PROFILE_COMPONENTS="${MINICPM_PROFILE_COMPONENTS}" \
MINICPM_AUTO_MAX_MEMORY_PER_GPU="${MINICPM_AUTO_MAX_MEMORY_PER_GPU}" \
"${PYTHON_BIN}" main_experiments/eval_minicpm_streamingbench.py \
    --anno-path "${SB_ANNO_PATH}" \
    --video-dir "${SB_VIDEO_DIR}" \
    --qa-model "openbmb/MiniCPM-V-4.6" \
    --qa-device auto \
    --top-k 0 \
    --frame-selection all \
    --recent-frames-only 4 \
    --chunk-duration 1.0 \
    --fps 1.0 \
    --context-time "${ALLFRAMES_CONTEXT_TIME}" \
    --max-qa-tokens 256 \
    --output-dir "${SB_RESULT_DIR}"
