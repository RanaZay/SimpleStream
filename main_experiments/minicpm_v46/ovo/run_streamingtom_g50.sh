#!/usr/bin/env bash
# OVO-Bench all-frame evaluation for MiniCPM-V-4.6 with full StreamingTOM:
# CTR before prefill + OQM 4-bit KV memory/retrieval.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29621}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE:-16x}"
MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS:-1}"
MINICPM_PROFILE_COMPONENTS="${MINICPM_PROFILE_COMPONENTS:-1}"
MINICPM_QA_DEVICE="${MINICPM_QA_DEVICE:-}"
MINICPM_SERIALIZE_MODEL_LOAD="${MINICPM_SERIALIZE_MODEL_LOAD:-1}"
MINICPM_MODEL_LOAD_TIMEOUT="${MINICPM_MODEL_LOAD_TIMEOUT:-7200}"
CTR_BUDGET="${CTR_BUDGET:-50}"
CTR_TAU="${CTR_TAU:-0.9}"
OQM_RETRIEVAL_MAX_TOKENS="${OQM_RETRIEVAL_MAX_TOKENS:-12544}"
OQM_BITS="${OQM_BITS:-4}"
OQM_INIT_TOKENS="${OQM_INIT_TOKENS:-14}"
MAX_SAMPLES_PER_SPLIT="${MAX_SAMPLES_PER_SPLIT:-}"

OVO_ANNO_PATH="${REPO_ROOT}/data/ovo_bench/ovo_bench_new.json"
OVO_CHUNKED_DIR="${REPO_ROOT}/data/ovo_bench/chunked_videos"
OVO_RESULT_DIR="${OVO_RESULT_DIR:-${REPO_ROOT}/main_experiments/results/repro_allframes/ovo_minicpmv46_allframes_fps1_streamingtom_g${CTR_BUDGET}_tau${CTR_TAU}_oqm${OQM_BITS}_ret${OQM_RETRIEVAL_MAX_TOKENS}}"

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
    CUDA_VISIBLE_DEVICES="$(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t',' -k2,2nr \
        | head -n "${NUM_PROCESSES}" \
        | cut -d',' -f1 \
        | tr -d ' ' \
        | paste -sd, -
    )"
fi

echo "[INFO] Using PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Using NUM_PROCESSES=${NUM_PROCESSES}"
echo "[INFO] Using ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[INFO] Using MINICPM_DOWNSAMPLE_MODE=${MINICPM_DOWNSAMPLE_MODE}"
echo "[INFO] Using MINICPM_MAX_SLICE_NUMS=${MINICPM_MAX_SLICE_NUMS}"
echo "[INFO] Using MINICPM_PROFILE_COMPONENTS=${MINICPM_PROFILE_COMPONENTS}"
echo "[INFO] Using MINICPM_SERIALIZE_MODEL_LOAD=${MINICPM_SERIALIZE_MODEL_LOAD}"
echo "[INFO] Using MINICPM_MODEL_LOAD_TIMEOUT=${MINICPM_MODEL_LOAD_TIMEOUT}"
echo "[INFO] Using CTR_BUDGET=${CTR_BUDGET}"
echo "[INFO] Using CTR_TAU=${CTR_TAU}"
echo "[INFO] Using OQM_RETRIEVAL_MAX_TOKENS=${OQM_RETRIEVAL_MAX_TOKENS}"
echo "[INFO] Using OQM_BITS=${OQM_BITS}"
echo "[INFO] Using OQM_INIT_TOKENS=${OQM_INIT_TOKENS}"
if [[ -n "${MINICPM_QA_DEVICE}" ]]; then
    echo "[INFO] Using MINICPM_QA_DEVICE=${MINICPM_QA_DEVICE}"
else
    echo "[INFO] Using MINICPM_QA_DEVICE=accelerator.device per rank"
fi
if [[ -n "${MAX_SAMPLES_PER_SPLIT}" ]]; then
    echo "[INFO] Using MAX_SAMPLES_PER_SPLIT=${MAX_SAMPLES_PER_SPLIT}"
fi
echo "[INFO] Frame selection: all frames at 1 FPS"
echo "[INFO] Results: ${OVO_RESULT_DIR}"

EXTRA_ARGS=()
if [[ -n "${MINICPM_QA_DEVICE}" ]]; then
    EXTRA_ARGS+=(--qa_device "${MINICPM_QA_DEVICE}")
fi
if [[ -n "${MAX_SAMPLES_PER_SPLIT}" ]]; then
    EXTRA_ARGS+=(--max_samples_per_split "${MAX_SAMPLES_PER_SPLIT}")
fi

RUN_ARGS=(
    main_experiments/minicpm_v46/ovo/eval_streamingtom.py \
    --model_path "openbmb/MiniCPM-V-4.6" \
    --anno_path "${OVO_ANNO_PATH}" \
    --chunked_dir "${OVO_CHUNKED_DIR}" \
    --result_dir "${OVO_RESULT_DIR}" \
    --frame_selection all \
    --recent_frames_only 4 \
    --chunk_duration 1.0 \
    --fps 1.0 \
    --max_qa_tokens 256 \
    --ctr_budget "${CTR_BUDGET}" \
    --ctr_tau "${CTR_TAU}" \
    --oqm_retrieval_max_tokens "${OQM_RETRIEVAL_MAX_TOKENS}" \
    --oqm_bits "${OQM_BITS}" \
    --oqm_init_tokens "${OQM_INIT_TOKENS}" \
    "${EXTRA_ARGS[@]}"
)

if [[ "${NUM_PROCESSES}" -le 1 ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
    MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE}" \
    MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS}" \
    MINICPM_PROFILE_COMPONENTS="${MINICPM_PROFILE_COMPONENTS}" \
    MINICPM_SERIALIZE_MODEL_LOAD="${MINICPM_SERIALIZE_MODEL_LOAD}" \
    MINICPM_MODEL_LOAD_TIMEOUT="${MINICPM_MODEL_LOAD_TIMEOUT}" \
    MINICPM_CTR_TOKEN_BUDGET="${CTR_BUDGET}" \
    MINICPM_CTR_TAU="${CTR_TAU}" \
    MINICPM_OQM_GROUP_SIZE="${CTR_BUDGET}" \
    MINICPM_OQM_RETRIEVAL_MAX_TOKENS="${OQM_RETRIEVAL_MAX_TOKENS}" \
    MINICPM_OQM_QUANTIZATION_BITS="${OQM_BITS}" \
    MINICPM_OQM_INIT_TOKEN_COUNT="${OQM_INIT_TOKENS}" \
    MINICPM_OQM_ENABLE_QUANTIZATION=1 \
    "${PYTHON_BIN}" "${RUN_ARGS[@]}"
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
    MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE}" \
    MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS}" \
    MINICPM_PROFILE_COMPONENTS="${MINICPM_PROFILE_COMPONENTS}" \
    MINICPM_SERIALIZE_MODEL_LOAD="${MINICPM_SERIALIZE_MODEL_LOAD}" \
    MINICPM_MODEL_LOAD_TIMEOUT="${MINICPM_MODEL_LOAD_TIMEOUT}" \
    MINICPM_CTR_TOKEN_BUDGET="${CTR_BUDGET}" \
    MINICPM_CTR_TAU="${CTR_TAU}" \
    MINICPM_OQM_GROUP_SIZE="${CTR_BUDGET}" \
    MINICPM_OQM_RETRIEVAL_MAX_TOKENS="${OQM_RETRIEVAL_MAX_TOKENS}" \
    MINICPM_OQM_QUANTIZATION_BITS="${OQM_BITS}" \
    MINICPM_OQM_INIT_TOKEN_COUNT="${OQM_INIT_TOKENS}" \
    MINICPM_OQM_ENABLE_QUANTIZATION=1 \
    "${PYTHON_BIN}" -m accelerate.commands.launch \
        --num_processes "${NUM_PROCESSES}" \
        --main_process_port "${MAIN_PROCESS_PORT}" \
        --multi_gpu \
        --mixed_precision bf16 \
        "${RUN_ARGS[@]}"
fi
