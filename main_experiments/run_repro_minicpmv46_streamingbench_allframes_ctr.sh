#!/usr/bin/env bash
# StreamingBench all-frame evaluation for MiniCPM-V-4.6 with CTR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29721}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE:-16x}"
MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS:-1}"
MINICPM_PROFILE_COMPONENTS="${MINICPM_PROFILE_COMPONENTS:-1}"
MINICPM_QA_DEVICE="${MINICPM_QA_DEVICE:-cuda:0}"
HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-false}"
HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-1}"
HF_DEACTIVATE_ASYNC_LOAD="${HF_DEACTIVATE_ASYNC_LOAD:-1}"
MINICPM_LOAD_ON_CPU_FIRST="${MINICPM_LOAD_ON_CPU_FIRST:-0}"
MINICPM_SERIALIZE_MODEL_LOAD="${MINICPM_SERIALIZE_MODEL_LOAD:-1}"
MINICPM_MODEL_LOAD_TIMEOUT="${MINICPM_MODEL_LOAD_TIMEOUT:-7200}"
ALLFRAMES_CONTEXT_TIME="${ALLFRAMES_CONTEXT_TIME:--1}"
CTR_BUDGET="${CTR_BUDGET:-50}"
CTR_TAU="${CTR_TAU:-0.9}"

SB_ANNO_PATH="${REPO_ROOT}/data/streamingbench/questions_real.json"
SB_VIDEO_DIR="${REPO_ROOT}/data/streamingbench/videos"
SB_RESULT_DIR="${SB_RESULT_DIR:-${REPO_ROOT}/main_experiments/results/repro_allframes/streamingbench_minicpmv46_allframes_fps1_ctr_g${CTR_BUDGET}_tau${CTR_TAU}}"

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
echo "[INFO] Using NUM_PROCESSES=${NUM_PROCESSES}"
echo "[INFO] Using ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[INFO] Using MINICPM_DOWNSAMPLE_MODE=${MINICPM_DOWNSAMPLE_MODE}"
echo "[INFO] Using MINICPM_MAX_SLICE_NUMS=${MINICPM_MAX_SLICE_NUMS}"
echo "[INFO] Using MINICPM_PROFILE_COMPONENTS=${MINICPM_PROFILE_COMPONENTS}"
echo "[INFO] Using MINICPM_QA_DEVICE=${MINICPM_QA_DEVICE}"
echo "[INFO] Using HF_ENABLE_PARALLEL_LOADING=${HF_ENABLE_PARALLEL_LOADING}"
echo "[INFO] Using HF_PARALLEL_LOADING_WORKERS=${HF_PARALLEL_LOADING_WORKERS}"
echo "[INFO] Using HF_DEACTIVATE_ASYNC_LOAD=${HF_DEACTIVATE_ASYNC_LOAD}"
echo "[INFO] Using MINICPM_LOAD_ON_CPU_FIRST=${MINICPM_LOAD_ON_CPU_FIRST}"
echo "[INFO] Using MINICPM_SERIALIZE_MODEL_LOAD=${MINICPM_SERIALIZE_MODEL_LOAD}"
echo "[INFO] Using MINICPM_MODEL_LOAD_TIMEOUT=${MINICPM_MODEL_LOAD_TIMEOUT}"
echo "[INFO] Using CTR_BUDGET=${CTR_BUDGET}"
echo "[INFO] Using CTR_TAU=${CTR_TAU}"
echo "[INFO] Frame selection: all frames at 1 FPS"
echo "[INFO] ALLFRAMES_CONTEXT_TIME=${ALLFRAMES_CONTEXT_TIME}"
echo "[INFO] Results: ${SB_RESULT_DIR}"

COMMON_ARGS=(
    --anno-path "${SB_ANNO_PATH}" \
    --video-dir "${SB_VIDEO_DIR}" \
    --qa-model "openbmb/MiniCPM-V-4.6" \
    --top-k 0 \
    --frame-selection all \
    --recent-frames-only 4 \
    --chunk-duration 1.0 \
    --fps 1.0 \
    --context-time "${ALLFRAMES_CONTEXT_TIME}" \
    --max-qa-tokens 256 \
    --output-dir "${SB_RESULT_DIR}" \
    --ctr-budget "${CTR_BUDGET}" \
    --ctr-tau "${CTR_TAU}"
)

if [[ -n "${MINICPM_QA_DEVICE}" ]]; then
    COMMON_ARGS+=(--qa-device "${MINICPM_QA_DEVICE}")
fi

if [[ "${NUM_PROCESSES}" -le 1 ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
    MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE}" \
    MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS}" \
    MINICPM_PROFILE_COMPONENTS="${MINICPM_PROFILE_COMPONENTS}" \
    HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING}" \
    HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS}" \
    HF_DEACTIVATE_ASYNC_LOAD="${HF_DEACTIVATE_ASYNC_LOAD}" \
    MINICPM_LOAD_ON_CPU_FIRST="${MINICPM_LOAD_ON_CPU_FIRST}" \
    MINICPM_CTR_TOKEN_BUDGET="${CTR_BUDGET}" \
    MINICPM_CTR_TAU="${CTR_TAU}" \
    "${PYTHON_BIN}" main_experiments/eval_minicpm_streamingbench_ctr.py "${COMMON_ARGS[@]}"
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
    MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE}" \
    MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS}" \
    MINICPM_PROFILE_COMPONENTS="${MINICPM_PROFILE_COMPONENTS}" \
    HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING}" \
    HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS}" \
    HF_DEACTIVATE_ASYNC_LOAD="${HF_DEACTIVATE_ASYNC_LOAD}" \
    MINICPM_SERIALIZE_MODEL_LOAD="${MINICPM_SERIALIZE_MODEL_LOAD}" \
    MINICPM_MODEL_LOAD_TIMEOUT="${MINICPM_MODEL_LOAD_TIMEOUT}" \
    MINICPM_CTR_TOKEN_BUDGET="${CTR_BUDGET}" \
    MINICPM_CTR_TAU="${CTR_TAU}" \
    "${PYTHON_BIN}" -m accelerate.commands.launch \
        --num_processes "${NUM_PROCESSES}" \
        --main_process_port "${MAIN_PROCESS_PORT}" \
        --multi_gpu \
        --mixed_precision bf16 \
        main_experiments/eval_minicpm_streamingbench_ctr_dist.py "${COMMON_ARGS[@]}"
fi
