#!/usr/bin/env bash
# OVO-Bench evaluation for MiniCPM-V-4.6 with CDAS enabled.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29581}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE:-16x}"
MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS:-1}"

CDAS_MODE="${CDAS_MODE:-three_level}"
CDAS_SKIP_THRESHOLD="${CDAS_SKIP_THRESHOLD:-0.03}"
CDAS_HIGH_THRESHOLD="${CDAS_HIGH_THRESHOLD:-0.12}"
CDAS_ANCHOR_SECONDS="${CDAS_ANCHOR_SECONDS:-2.0}"
CDAS_MIN_ACCEPTED_FPS="${CDAS_MIN_ACCEPTED_FPS:-0.25}"

OVO_ANNO_PATH="${REPO_ROOT}/data/ovo_bench/ovo_bench_new.json"
OVO_CHUNKED_DIR="${REPO_ROOT}/data/ovo_bench/chunked_videos"
OVO_RESULT_DIR="${REPO_ROOT}/main_experiments/results/repro_recent4/ovo_minicpmv46_recent4_cdas_${CDAS_MODE}"

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
echo "[INFO] Using ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[INFO] Using MINICPM_DOWNSAMPLE_MODE=${MINICPM_DOWNSAMPLE_MODE}"
echo "[INFO] Using MINICPM_MAX_SLICE_NUMS=${MINICPM_MAX_SLICE_NUMS}"
echo "[INFO] CDAS mode=${CDAS_MODE} skip=${CDAS_SKIP_THRESHOLD} high=${CDAS_HIGH_THRESHOLD} anchor=${CDAS_ANCHOR_SECONDS} min_fps=${CDAS_MIN_ACCEPTED_FPS}"
echo "[INFO] Results: ${OVO_RESULT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE}" \
MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS}" \
"${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --multi_gpu \
    --mixed_precision bf16 \
    main_experiments/eval_minicpm_ovo.py \
    --model_path "openbmb/MiniCPM-V-4.6" \
    --anno_path "${OVO_ANNO_PATH}" \
    --chunked_dir "${OVO_CHUNKED_DIR}" \
    --result_dir "${OVO_RESULT_DIR}" \
    --recent_frames_only 4 \
    --chunk_duration 1.0 \
    --fps 1.0 \
    --max_qa_tokens 256 \
    --cdas_enable \
    --cdas_mode "${CDAS_MODE}" \
    --cdas_skip_threshold "${CDAS_SKIP_THRESHOLD}" \
    --cdas_high_threshold "${CDAS_HIGH_THRESHOLD}" \
    --cdas_anchor_seconds "${CDAS_ANCHOR_SECONDS}" \
    --cdas_min_accepted_fps "${CDAS_MIN_ACCEPTED_FPS}"
