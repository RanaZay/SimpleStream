#!/bin/bash
#SBATCH --job-name=minicpmv46_ovo_storymem_d8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=8:00:00
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

REPO_ROOT=/vast/users/salman.khan/SimpleStream
cd "$REPO_ROOT" || exit 1
mkdir -p logs .cache/miopen .cache/torch_kernels
export MIOPEN_USER_DB_PATH="$REPO_ROOT/.cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$REPO_ROOT/.cache/miopen"
export PYTORCH_KERNEL_CACHE_PATH="$REPO_ROOT/.cache/torch_kernels"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export STORY_RECENT_FRAMES=${STORY_RECENT_FRAMES:-6}
export STORY_BATCH_SIZE=${STORY_BATCH_SIZE:-8}
export STORY_MAX_ITEMS=${STORY_MAX_ITEMS:-96}
export STORY_MAX_PROMPT_CHARS=${STORY_MAX_PROMPT_CHARS:-9000}
export STORY_DESC_MAX_TOKENS=${STORY_DESC_MAX_TOKENS:-192}
export STORY_DESCRIBE_STRIDE=${STORY_DESCRIBE_STRIDE:-1}
export NUM_PROCESSES=${NUM_PROCESSES:-8}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29851}

RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_story_memory/ovo_minicpmv46_story_memory_recent${STORY_RECENT_FRAMES}_b${STORY_BATCH_SIZE}_m${STORY_MAX_ITEMS}_s${STORY_DESCRIBE_STRIDE}_d8"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export OVO_RESULT_DIR="$RESULT_DIR"

bash main_experiments/minicpm_v46/ovo/run_story_memory.sh
