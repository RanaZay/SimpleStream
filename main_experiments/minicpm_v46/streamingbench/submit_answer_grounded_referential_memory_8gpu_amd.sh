#!/bin/bash
#SBATCH --job-name=minicpmv46_sb_agrefmem_d8
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

# Recent-6 backbone + previous-question memory.
# After each answer, cache only the current frame(s) whose CLIP embedding is
# closest to the question + predicted answer text.
export REFERENTIAL_RECENT_WINDOW=${REFERENTIAL_RECENT_WINDOW:-6}
export REFERENTIAL_REFERENCE_FRAMES=${REFERENTIAL_REFERENCE_FRAMES:-2}
export REFERENTIAL_MEMORY_ANCHOR_FRAMES=${REFERENTIAL_MEMORY_ANCHOR_FRAMES:-1}
export REFERENTIAL_CONTEXT_TIME=${REFERENTIAL_CONTEXT_TIME:--1}
export MINICPM_REF_CLIP_MODEL=${MINICPM_REF_CLIP_MODEL:-openai/clip-vit-base-patch32}
export MINICPM_REF_CLIP_DEVICE=${MINICPM_REF_CLIP_DEVICE:-}
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29876}

RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_referential/streamingbench_minicpmv46_answer_grounded_referential_memory_recent${REFERENTIAL_RECENT_WINDOW}_r${REFERENTIAL_REFERENCE_FRAMES}_a${REFERENTIAL_MEMORY_ANCHOR_FRAMES}_d8"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export SB_RESULT_DIR="$RESULT_DIR"

bash main_experiments/minicpm_v46/streamingbench/run_answer_grounded_referential_memory.sh
