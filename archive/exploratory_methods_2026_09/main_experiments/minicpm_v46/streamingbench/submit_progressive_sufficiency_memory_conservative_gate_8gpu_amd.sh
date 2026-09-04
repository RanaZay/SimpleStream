#!/bin/bash
#SBATCH --job-name=minicpmv46_sb_psm_cgate_d8
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

export ADAPTIVE_MODE=progressive_sufficiency_memory_conservative_gate
export ADAPTIVE_MIN_WINDOW=6
export ADAPTIVE_MID_WINDOW=6
export ADAPTIVE_MAX_WINDOW=6
export ADAPTIVE_MEMORY_ANCHORS=3
export ADAPTIVE_MEMORY_SEARCH_CHUNKS=64
export ADAPTIVE_CONTEXT_TIME=70
export MINICPM_PSM_HISTORY_SEARCH_CHUNKS=64
export MINICPM_PSM_HISTORY_CANDIDATE_POOL=12
export MINICPM_PSM_MAX_MEMORY_FRAMES=3
export MINICPM_PSM_MIN_TEMPORAL_GAP=2
export MINICPM_PSM_SUFFICIENCY_THRESHOLD=${MINICPM_PSM_SUFFICIENCY_THRESHOLD:-0.62}
export MINICPM_PSM_GATE_TAU_LOW=${MINICPM_PSM_GATE_TAU_LOW:-0.62}
export MINICPM_PSM_GATE_TAU_HIGH=${MINICPM_PSM_GATE_TAU_HIGH:-0.74}
export MINICPM_PSM_GATE_CANDIDATE_THRESHOLD=${MINICPM_PSM_GATE_CANDIDATE_THRESHOLD:-0.535}
export MINICPM_PSM_GATE_TEMPORAL_DISTANCE_THRESHOLD=${MINICPM_PSM_GATE_TEMPORAL_DISTANCE_THRESHOLD:-10}
export MINICPM_PSM_MIN_EVIDENCE_GAIN=0.035
export MINICPM_PSM_NEGATIVE_GAIN_TOLERANCE=0.02
export MINICPM_PSM_MARGIN_WEIGHT=0.50
export MINICPM_PSM_ENTROPY_WEIGHT=0.20
export MINICPM_PSM_VISUAL_SUPPORT_WEIGHT=0.30
export MINICPM_PSM_PRINT_TRACE=1
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29951}

LIMIT_TAG=full
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    LIMIT_TAG="limit${MAX_SAMPLES}"
fi
LOW_TAG="${MINICPM_PSM_GATE_TAU_LOW/./p}"
HIGH_TAG="${MINICPM_PSM_GATE_TAU_HIGH/./p}"
CAND_TAG="${MINICPM_PSM_GATE_CANDIDATE_THRESHOLD/./p}"
TD_TAG="${MINICPM_PSM_GATE_TEMPORAL_DISTANCE_THRESHOLD/./p}"
RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_adaptive/streamingbench_minicpmv46_psm_conservative_gate_recent6_h64_p12_m3_ctx70_low${LOW_TAG}_high${HIGH_TAG}_cand${CAND_TAG}_td${TD_TAG}_${LIMIT_TAG}_d8"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export SB_RESULT_DIR="$RESULT_DIR"

bash main_experiments/minicpm_v46/streamingbench/run_adaptive.sh
