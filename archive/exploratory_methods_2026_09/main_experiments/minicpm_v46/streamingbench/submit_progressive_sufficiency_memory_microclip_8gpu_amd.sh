#!/bin/bash
#SBATCH --job-name=minicpmv46_sb_psm_microclip_d8
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export ADAPTIVE_MODE=progressive_sufficiency_memory_microclip
export ADAPTIVE_MIN_WINDOW=6
export ADAPTIVE_MID_WINDOW=6
export ADAPTIVE_MAX_WINDOW=6
export ADAPTIVE_MEMORY_ANCHORS=3
export ADAPTIVE_MEMORY_SEARCH_CHUNKS=64
export ADAPTIVE_CONTEXT_TIME=70
export MINICPM_PSM_HISTORY_SEARCH_CHUNKS=64
export MINICPM_PSM_HISTORY_CANDIDATE_POOL=12
export MINICPM_PSM_MIN_TEMPORAL_GAP=2
export MINICPM_PSM_SUFFICIENCY_THRESHOLD=${MINICPM_PSM_SUFFICIENCY_THRESHOLD:-0.62}
export MINICPM_PSM_MARGIN_WEIGHT=0.50
export MINICPM_PSM_ENTROPY_WEIGHT=0.20
export MINICPM_PSM_VISUAL_SUPPORT_WEIGHT=0.30
export MINICPM_PSM_MICROCLIP_OFFSETS=${MINICPM_PSM_MICROCLIP_OFFSETS:-"-1,0,1"}
export MINICPM_PSM_MICROCLIP_VARIANT=${MINICPM_PSM_MICROCLIP_VARIANT:-temporal_microclip}
export MINICPM_PSM_PRINT_TRACE=1
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29961}

# Diagnostic default: do not accidentally launch the full StreamingBench run.
export MAX_SAMPLES=${MAX_SAMPLES:-100}

LIMIT_TAG="limit${MAX_SAMPLES}"
OFFSETS_TAG="$(echo "${MINICPM_PSM_MICROCLIP_OFFSETS}" | sed -e 's/-/m/g' -e 's/\./p/g' -e 's/,/_/g' | tr -cd '[:alnum:]_')"
RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_adaptive/streamingbench_minicpmv46_psm_microclip_${MINICPM_PSM_MICROCLIP_VARIANT}_recent6_h64_p12_ctx70_offsets${OFFSETS_TAG}_${LIMIT_TAG}_d8"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export SB_RESULT_DIR="$RESULT_DIR"

bash main_experiments/minicpm_v46/streamingbench/run_adaptive.sh
