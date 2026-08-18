#!/usr/bin/env bash
# Full StreamingBench evaluation for isolated PRISM with corrected exact-six Recent-6.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29851}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE:-16x}"
MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS:-1}"
MINICPM_PROFILE_COMPONENTS="${MINICPM_PROFILE_COMPONENTS:-1}"
MINICPM_QA_DEVICE="${MINICPM_QA_DEVICE:-}"
HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-false}"
HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-1}"
HF_DEACTIVATE_ASYNC_LOAD="${HF_DEACTIVATE_ASYNC_LOAD:-1}"
MINICPM_SERIALIZE_MODEL_LOAD="${MINICPM_SERIALIZE_MODEL_LOAD:-1}"
MINICPM_MODEL_LOAD_TIMEOUT="${MINICPM_MODEL_LOAD_TIMEOUT:-7200}"
MINICPM_SEED="${MINICPM_SEED:-42}"
RECENT_FRAMES_ONLY="${RECENT_FRAMES_ONLY:-6}"
RECENT_SAMPLER_FPS="${RECENT_SAMPLER_FPS:-4.0}"
ADAPTIVE_CONTEXT_TIME="${ADAPTIVE_CONTEXT_TIME:-70}"
ADAPTIVE_MODE="${ADAPTIVE_MODE:-progressive_sufficiency_memory}"
ADAPTIVE_MIN_WINDOW="${ADAPTIVE_MIN_WINDOW:-6}"
ADAPTIVE_MID_WINDOW="${ADAPTIVE_MID_WINDOW:-6}"
ADAPTIVE_MAX_WINDOW="${ADAPTIVE_MAX_WINDOW:-6}"
ADAPTIVE_DEDUP_THRESHOLD="${ADAPTIVE_DEDUP_THRESHOLD:-4.0}"
ADAPTIVE_DEDUP_MIN_FRAMES="${ADAPTIVE_DEDUP_MIN_FRAMES:-4}"
ADAPTIVE_MEMORY_ANCHORS="${ADAPTIVE_MEMORY_ANCHORS:-3}"
ADAPTIVE_MEMORY_SEARCH_CHUNKS="${ADAPTIVE_MEMORY_SEARCH_CHUNKS:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

SB_ANNO_PATH="${REPO_ROOT}/data/streamingbench/questions_real.json"
SB_VIDEO_DIR="${REPO_ROOT}/data/streamingbench/videos"
SB_RESULT_DIR="${SB_RESULT_DIR:-${REPO_ROOT}/main_experiments/results/repro_adaptive/streamingbench_minicpmv46_prism_current_recent6_exact6_h64_p12_m3_t0p62_full_d8}"

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
echo "[INFO] Using ADAPTIVE_MODE=${ADAPTIVE_MODE}"
echo "[INFO] Using exact current_recent6 candidate fps=${RECENT_SAMPLER_FPS}"
echo "[INFO] Results: ${SB_RESULT_DIR}"

COMMON_ARGS=(
    --anno-path "${SB_ANNO_PATH}"
    --video-dir "${SB_VIDEO_DIR}"
    --qa-model "openbmb/MiniCPM-V-4.6"
    --top-k 0
    --recent-frames-only "${RECENT_FRAMES_ONLY}"
    --chunk-duration 1.0
    --fps 1.0
    --context-time "${ADAPTIVE_CONTEXT_TIME}"
    --max-qa-tokens 256
    --adaptive-mode "${ADAPTIVE_MODE}"
    --adaptive-min-window "${ADAPTIVE_MIN_WINDOW}"
    --adaptive-mid-window "${ADAPTIVE_MID_WINDOW}"
    --adaptive-max-window "${ADAPTIVE_MAX_WINDOW}"
    --adaptive-dedup-threshold "${ADAPTIVE_DEDUP_THRESHOLD}"
    --adaptive-dedup-min-frames "${ADAPTIVE_DEDUP_MIN_FRAMES}"
    --adaptive-memory-anchors "${ADAPTIVE_MEMORY_ANCHORS}"
    --adaptive-memory-search-chunks "${ADAPTIVE_MEMORY_SEARCH_CHUNKS}"
    --output-dir "${SB_RESULT_DIR}"
)

if [[ -n "${MINICPM_QA_DEVICE}" ]]; then
    COMMON_ARGS+=(--qa-device "${MINICPM_QA_DEVICE}")
elif [[ "${NUM_PROCESSES}" -le 1 ]]; then
    COMMON_ARGS+=(--qa-device auto)
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
    COMMON_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

export MINICPM_PSM_HISTORY_SEARCH_CHUNKS="${MINICPM_PSM_HISTORY_SEARCH_CHUNKS:-64}"
export MINICPM_PSM_HISTORY_CANDIDATE_POOL="${MINICPM_PSM_HISTORY_CANDIDATE_POOL:-12}"
export MINICPM_PSM_MAX_MEMORY_FRAMES="${MINICPM_PSM_MAX_MEMORY_FRAMES:-3}"
export MINICPM_PSM_SUFFICIENCY_THRESHOLD="${MINICPM_PSM_SUFFICIENCY_THRESHOLD:-0.62}"
export MINICPM_PSM_MARGIN_WEIGHT="${MINICPM_PSM_MARGIN_WEIGHT:-0.50}"
export MINICPM_PSM_ENTROPY_WEIGHT="${MINICPM_PSM_ENTROPY_WEIGHT:-0.20}"
export MINICPM_PSM_VISUAL_SUPPORT_WEIGHT="${MINICPM_PSM_VISUAL_SUPPORT_WEIGHT:-0.30}"
export MINICPM_PSM_ASSERT_TEMPORAL_ALIGNMENT="${MINICPM_PSM_ASSERT_TEMPORAL_ALIGNMENT:-1}"
export MINICPM_EXACT_RECENT_CANDIDATE_FPS="${RECENT_SAMPLER_FPS}"

if [[ "${NUM_PROCESSES}" -le 1 ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
    MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE}" \
    MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS}" \
    MINICPM_PROFILE_COMPONENTS="${MINICPM_PROFILE_COMPONENTS}" \
    HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING}" \
    HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS}" \
    HF_DEACTIVATE_ASYNC_LOAD="${HF_DEACTIVATE_ASYNC_LOAD}" \
    MINICPM_SEED="${MINICPM_SEED}" \
    PYTHONHASHSEED="${MINICPM_SEED}" \
    QWEN_EXACT_RECENT_DECODE=0 \
    "${PYTHON_BIN}" main_experiments/minicpm_v46/streamingbench/eval_prism_exact_recent_dist.py "${COMMON_ARGS[@]}"
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
    MINICPM_SEED="${MINICPM_SEED}" \
    PYTHONHASHSEED="${MINICPM_SEED}" \
    QWEN_EXACT_RECENT_DECODE=0 \
    "${PYTHON_BIN}" -m accelerate.commands.launch \
        --num_processes "${NUM_PROCESSES}" \
        --main_process_port "${MAIN_PROCESS_PORT}" \
        --multi_gpu \
        --mixed_precision bf16 \
        main_experiments/minicpm_v46/streamingbench/eval_prism_exact_recent_dist.py "${COMMON_ARGS[@]}"
fi
