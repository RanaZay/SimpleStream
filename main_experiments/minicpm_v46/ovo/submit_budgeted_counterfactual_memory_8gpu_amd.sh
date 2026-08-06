#!/bin/bash
#SBATCH --job-name=minicpmv46_ovo_bcm_d8
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

# Budgeted Counterfactual Memory:
# Always keep SimpleStream Recent-6, then retrieve K=0/1/2 older frames only
# when estimated history benefit is higher than the retrieval cost.
export ADAPTIVE_MODE=budgeted_counterfactual_memory
export ADAPTIVE_MIN_WINDOW=6
export ADAPTIVE_MID_WINDOW=6
export ADAPTIVE_MAX_WINDOW=6
export ADAPTIVE_MEMORY_ANCHORS=${ADAPTIVE_MEMORY_ANCHORS:-2}
export ADAPTIVE_MEMORY_SEARCH_CHUNKS=${ADAPTIVE_MEMORY_SEARCH_CHUNKS:-32}
export MINICPM_BCM_MAX_RETRIEVED=${MINICPM_BCM_MAX_RETRIEVED:-2}
export MINICPM_BCM_UTILITY_THRESHOLD=${MINICPM_BCM_UTILITY_THRESHOLD:-0.58}
export MINICPM_BCM_HIGH_UTILITY_THRESHOLD=${MINICPM_BCM_HIGH_UTILITY_THRESHOLD:-0.82}
export MINICPM_BCM_MEMORY_RELEVANCE_THRESHOLD=${MINICPM_BCM_MEMORY_RELEVANCE_THRESHOLD:-0.46}
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29876}

RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_adaptive/ovo_minicpmv46_budgeted_counterfactual_memory_recent6_m${ADAPTIVE_MEMORY_ANCHORS}_s${ADAPTIVE_MEMORY_SEARCH_CHUNKS}_d8"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export OVO_RESULT_DIR="$RESULT_DIR"

bash main_experiments/minicpm_v46/ovo/run_adaptive.sh
