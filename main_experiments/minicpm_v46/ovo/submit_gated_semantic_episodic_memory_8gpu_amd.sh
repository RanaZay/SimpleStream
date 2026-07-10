#!/bin/bash
#SBATCH --job-name=minicpmv46_ovo_gsemepi_d8
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
export ATTN_IMPLEMENTATION=sdpa
export MINICPM_DOWNSAMPLE_MODE=16x
export MINICPM_MAX_SLICE_NUMS=1
export MINICPM_PROFILE_COMPONENTS=${MINICPM_PROFILE_COMPONENTS:-1}
export MINICPM_LOAD_ON_CPU_FIRST=0
export MINICPM_SERIALIZE_MODEL_LOAD=${MINICPM_SERIALIZE_MODEL_LOAD:-1}
export MINICPM_MODEL_LOAD_TIMEOUT=${MINICPM_MODEL_LOAD_TIMEOUT:-7200}
export HF_ENABLE_PARALLEL_LOADING=false
export HF_PARALLEL_LOADING_WORKERS=1
export HF_DEACTIVATE_ASYNC_LOAD=1
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-65536}
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

# Recent-6 by default. For clear history/count questions only, activate the
# bound semantic-episodic memory selector over older chunks.
export ADAPTIVE_MODE=gated_semantic_episodic_memory
export ADAPTIVE_MIN_WINDOW=6
export ADAPTIVE_MID_WINDOW=6
export ADAPTIVE_MAX_WINDOW=6
export ADAPTIVE_MEMORY_ANCHORS=${ADAPTIVE_MEMORY_ANCHORS:-3}
export ADAPTIVE_MEMORY_SEARCH_CHUNKS=${ADAPTIVE_MEMORY_SEARCH_CHUNKS:-32}
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29870}

RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_adaptive/ovo_minicpmv46_gated_semantic_episodic_memory_recent6_m${ADAPTIVE_MEMORY_ANCHORS}_s${ADAPTIVE_MEMORY_SEARCH_CHUNKS}_d8"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export OVO_RESULT_DIR="$RESULT_DIR"

bash main_experiments/minicpm_v46/ovo/run_adaptive.sh
