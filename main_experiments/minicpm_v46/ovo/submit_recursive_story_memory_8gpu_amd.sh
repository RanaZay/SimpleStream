#!/bin/bash
#SBATCH --job-name=minicpmv46_ovo_rsm_d8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
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

REPO_ROOT=/vast/users/salman.khan/SimpleStream
cd "$REPO_ROOT" || exit 1
mkdir -p logs .cache/miopen .cache/torch_kernels
export MIOPEN_USER_DB_PATH="$REPO_ROOT/.cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$REPO_ROOT/.cache/miopen"
export PYTORCH_KERNEL_CACHE_PATH="$REPO_ROOT/.cache/torch_kernels"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export RSM_RECENT_FRAMES=${RSM_RECENT_FRAMES:-6}
export RSM_UPDATE_BATCH=${RSM_UPDATE_BATCH:-4}
export RSM_MAX_STORY_TOKENS=${RSM_MAX_STORY_TOKENS:-256}
export RSM_REWRITE_MAX_NEW_TOKENS=${RSM_REWRITE_MAX_NEW_TOKENS:-384}
export RSM_COMPRESS_MAX_NEW_TOKENS=${RSM_COMPRESS_MAX_NEW_TOKENS:-320}
export RSM_MAX_COMPRESSION_ATTEMPTS=${RSM_MAX_COMPRESSION_ATTEMPTS:-1}
export RSM_PROMPT_TAG=${RSM_PROMPT_TAG:-v1}
export NUM_PROCESSES=${NUM_PROCESSES:-8}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29871}

RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_recursive_story_memory/ovo_minicpmv46_recursive_story_memory_recent${RSM_RECENT_FRAMES}_b${RSM_UPDATE_BATCH}_l${RSM_MAX_STORY_TOKENS}_${RSM_PROMPT_TAG}_d8"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export OVO_RESULT_DIR="$RESULT_DIR"

bash main_experiments/minicpm_v46/ovo/run_recursive_story_memory.sh
