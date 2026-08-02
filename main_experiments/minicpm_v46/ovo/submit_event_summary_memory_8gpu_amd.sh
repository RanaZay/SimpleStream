#!/bin/bash
#SBATCH --job-name=minicpmv46_ovo_evsum_d8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --qos=skqos
#SBATCH --partition=faculty
#SBATCH --output=/vast/users/salman.khan/SimpleStream/logs/%x-%j.out

source ~/.bashrc
conda activate stream35

export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export ROCM_HOME=${ROCM_HOME:-/opt/rocm}
export PATH="${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_HOME}/lib64:${LD_LIBRARY_PATH}"
export MIOPEN_DISABLE_CACHE=1
export PYTORCH_TUNABLEOP_ENABLED=0
export MINICPM_SEED=${MINICPM_SEED:-42}
export PYTHONHASHSEED=${MINICPM_SEED}

REPO_ROOT=/vast/users/salman.khan/SimpleStream
cd "$REPO_ROOT" || exit 1
mkdir -p logs .cache/miopen .cache/torch_kernels
export MIOPEN_USER_DB_PATH="$REPO_ROOT/.cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$REPO_ROOT/.cache/miopen"
export PYTORCH_KERNEL_CACHE_PATH="$REPO_ROOT/.cache/torch_kernels"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Recent-6 backbone + Conditional Event Bookmark Memory.
# Keep a tiny bounded/diverse event index and retrieve at most one old frame
# only when a history/reference question confidently matches a bookmark.
export ADAPTIVE_MODE=event_summary_memory
export ADAPTIVE_MIN_WINDOW=6
export ADAPTIVE_MID_WINDOW=6
export ADAPTIVE_MAX_WINDOW=6
export ADAPTIVE_MEMORY_ANCHORS=${ADAPTIVE_MEMORY_ANCHORS:-1}
export ADAPTIVE_MEMORY_SEARCH_CHUNKS=${ADAPTIVE_MEMORY_SEARCH_CHUNKS:-32}
export MINICPM_EVENT_SUMMARY_MAX_ITEMS=${MINICPM_EVENT_SUMMARY_MAX_ITEMS:-5}
export MINICPM_EVENT_SUMMARY_GATE=${MINICPM_EVENT_SUMMARY_GATE:-1}
export MINICPM_EVENT_SUMMARY_IMPORTANCE_THRESHOLD=${MINICPM_EVENT_SUMMARY_IMPORTANCE_THRESHOLD:-0.45}
export MINICPM_EVENT_SUMMARY_QUERY_THRESHOLD=${MINICPM_EVENT_SUMMARY_QUERY_THRESHOLD:-0.50}
export MINICPM_EVENT_SUMMARY_QUERY_MARGIN=${MINICPM_EVENT_SUMMARY_QUERY_MARGIN:-0.08}
export MINICPM_EVENT_SUMMARY_MAX_RETRIEVED=${MINICPM_EVENT_SUMMARY_MAX_RETRIEVED:-1}
export MINICPM_EVENT_SUMMARY_MIN_GAP_SECONDS=${MINICPM_EVENT_SUMMARY_MIN_GAP_SECONDS:-3.0}
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29881}

RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_adaptive/ovo_minicpmv46_event_summary_memory_recent6_m${ADAPTIVE_MEMORY_ANCHORS}_s${ADAPTIVE_MEMORY_SEARCH_CHUNKS}_d8"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export OVO_RESULT_DIR="$RESULT_DIR"

bash main_experiments/minicpm_v46/ovo/run_adaptive.sh
