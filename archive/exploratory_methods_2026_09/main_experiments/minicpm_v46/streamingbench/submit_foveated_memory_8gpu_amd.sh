#!/bin/bash
#SBATCH --job-name=minicpmv46_sb_fovmem_d8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
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

# Recent-6 + triggered memory, with query-guided foveation for localization
# questions. This keeps the frame count unchanged while adding zoomed local
# detail inside each selected frame.
export ADAPTIVE_MODE=foveated_memory
export ADAPTIVE_MIN_WINDOW=6
export ADAPTIVE_MID_WINDOW=6
export ADAPTIVE_MAX_WINDOW=6
export ADAPTIVE_MEMORY_ANCHORS=${ADAPTIVE_MEMORY_ANCHORS:-2}
export ADAPTIVE_CONTEXT_TIME=${ADAPTIVE_CONTEXT_TIME:-60}
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29851}

RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_adaptive/streamingbench_minicpmv46_foveated_recent6_memory_m${ADAPTIVE_MEMORY_ANCHORS}_ctx${ADAPTIVE_CONTEXT_TIME}_d8"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export SB_RESULT_DIR="$RESULT_DIR"

bash main_experiments/minicpm_v46/streamingbench/run_adaptive.sh
