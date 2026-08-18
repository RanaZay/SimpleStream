#!/bin/bash
#SBATCH --job-name=minicpmv46_ovo_prism_current6_exact_d8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --qos=skqos
#SBATCH --partition=faculty
#SBATCH --output=/vast/users/salman.khan/SimpleStream/logs/%x-%j.out

set -euo pipefail
source ~/.bashrc
conda activate stream35

export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export ROCM_HOME=${ROCM_HOME:-/opt/rocm}
export PATH="${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export ATTN_IMPLEMENTATION=sdpa
export MINICPM_DOWNSAMPLE_MODE=16x
export MINICPM_MAX_SLICE_NUMS=1
export MINICPM_PROFILE_COMPONENTS=${MINICPM_PROFILE_COMPONENTS:-1}
export MINICPM_SERIALIZE_MODEL_LOAD=${MINICPM_SERIALIZE_MODEL_LOAD:-1}
export MINICPM_MODEL_LOAD_TIMEOUT=${MINICPM_MODEL_LOAD_TIMEOUT:-7200}
export HF_ENABLE_PARALLEL_LOADING=false
export HF_PARALLEL_LOADING_WORKERS=1
export HF_DEACTIVATE_ASYNC_LOAD=1
export MIOPEN_DISABLE_CACHE=1
export PYTORCH_TUNABLEOP_ENABLED=0
export MINICPM_SEED=${MINICPM_SEED:-42}
export PYTHONHASHSEED=${MINICPM_SEED}

REPO_ROOT=/vast/users/salman.khan/SimpleStream
cd "$REPO_ROOT"

mkdir -p logs .cache/miopen .cache/torch_kernels
export MIOPEN_USER_DB_PATH="$REPO_ROOT/.cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$REPO_ROOT/.cache/miopen"
export PYTORCH_KERNEL_CACHE_PATH="$REPO_ROOT/.cache/torch_kernels"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}

export ADAPTIVE_MODE=progressive_sufficiency_memory
export ADAPTIVE_MIN_WINDOW=6
export ADAPTIVE_MID_WINDOW=6
export ADAPTIVE_MAX_WINDOW=6
export ADAPTIVE_MEMORY_ANCHORS=3
export ADAPTIVE_MEMORY_SEARCH_CHUNKS=64
export MINICPM_EXACT_RECENT_CANDIDATE_FPS=4.0
export MINICPM_PSM_HISTORY_SEARCH_CHUNKS=64
export MINICPM_PSM_HISTORY_CANDIDATE_POOL=12
export MINICPM_PSM_MAX_MEMORY_FRAMES=3
export MINICPM_PSM_MIN_TEMPORAL_GAP=2
export MINICPM_PSM_SUFFICIENCY_THRESHOLD=0.62
export MINICPM_PSM_MIN_EVIDENCE_GAIN=0.035
export MINICPM_PSM_NEGATIVE_GAIN_TOLERANCE=0.02
export MINICPM_PSM_MARGIN_WEIGHT=0.50
export MINICPM_PSM_ENTROPY_WEIGHT=0.20
export MINICPM_PSM_VISUAL_SUPPORT_WEIGHT=0.30
export MINICPM_PSM_ASSERT_TEMPORAL_ALIGNMENT=1
export MINICPM_PSM_PRINT_TRACE=1
export NUM_PROCESSES=${NUM_PROCESSES:-8}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29931}

LIMIT_TAG=full
if [[ -n "${MAX_SAMPLES_TOTAL:-}" ]]; then
    LIMIT_TAG="limit${MAX_SAMPLES_TOTAL}"
elif [[ -n "${MAX_SAMPLES_PER_SPLIT:-}" ]]; then
    LIMIT_TAG="limit${MAX_SAMPLES_PER_SPLIT}_per_split"
fi
RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_adaptive/ovo_minicpmv46_prism_current_recent6_exact6_h64_p12_m3_t0p62_${LIMIT_TAG}_d8"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export OVO_RESULT_DIR="$RESULT_DIR"

COMMON_ARGS=(
    --model_path "openbmb/MiniCPM-V-4.6"
    --anno_path "${OVO_ANNO_PATH:-$REPO_ROOT/data/ovo_bench/ovo_bench_new.json}"
    --chunked_dir "${OVO_CHUNKED_DIR:-$REPO_ROOT/data/ovo_bench/chunked_videos}"
    --result_dir "${OVO_RESULT_DIR}"
    --recent_frames_only "${ADAPTIVE_MAX_WINDOW}"
    --chunk_duration 1.0
    --fps 1.0
    --max_qa_tokens 256
    --adaptive-mode "${ADAPTIVE_MODE}"
    --adaptive-min-window "${ADAPTIVE_MIN_WINDOW}"
    --adaptive-mid-window "${ADAPTIVE_MID_WINDOW}"
    --adaptive-max-window "${ADAPTIVE_MAX_WINDOW}"
    --adaptive-dedup-threshold 4.0
    --adaptive-dedup-min-frames 4
    --adaptive-memory-anchors "${ADAPTIVE_MEMORY_ANCHORS}"
    --adaptive-memory-search-chunks "${ADAPTIVE_MEMORY_SEARCH_CHUNKS}"
)
if [[ -n "${MAX_SAMPLES_PER_SPLIT:-}" ]]; then
    COMMON_ARGS+=(--max_samples_per_split "${MAX_SAMPLES_PER_SPLIT}")
fi
if [[ -n "${MAX_SAMPLES_TOTAL:-}" ]]; then
    COMMON_ARGS+=(--max_samples_total "${MAX_SAMPLES_TOTAL}")
fi

PYTHON_BIN=$(which python)
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
    main_experiments/minicpm_v46/ovo/eval_prism_exact_recent.py "${COMMON_ARGS[@]}"
