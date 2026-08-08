#!/bin/bash
#SBATCH --job-name=minicpmv46_sb_fpem_d8
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

# Full Progressive Evidence Memory:
# 1) start with SimpleStream Recent-6;
# 2) score MiniCPM option probabilities for A/B/C/D;
# 3) combine answer margin/entropy with evidence support;
# 4) retrieve semantic anchors one by one only while insufficient.
export ADAPTIVE_MODE=full_progressive_evidence_memory
export ADAPTIVE_MIN_WINDOW=6
export ADAPTIVE_MID_WINDOW=6
export ADAPTIVE_MAX_WINDOW=6
export ADAPTIVE_MEMORY_ANCHORS=${ADAPTIVE_MEMORY_ANCHORS:-3}
export ADAPTIVE_MEMORY_SEARCH_CHUNKS=${ADAPTIVE_MEMORY_SEARCH_CHUNKS:-32}
export ADAPTIVE_CONTEXT_TIME=${ADAPTIVE_CONTEXT_TIME:-60}
export MINICPM_FPEM_MAX_RETRIEVED=${MINICPM_FPEM_MAX_RETRIEVED:-3}
export MINICPM_FPEM_SUFFICIENCY_THRESHOLD=${MINICPM_FPEM_SUFFICIENCY_THRESHOLD:-0.55}
export MINICPM_FPEM_MARGINAL_GAIN_THRESHOLD=${MINICPM_FPEM_MARGINAL_GAIN_THRESHOLD:-0.025}
export MINICPM_FPEM_MEMORY_RELEVANCE_THRESHOLD=${MINICPM_FPEM_MEMORY_RELEVANCE_THRESHOLD:-0.35}
export MINICPM_FPEM_CONFIDENCE_WEIGHT=${MINICPM_FPEM_CONFIDENCE_WEIGHT:-0.55}
export MINICPM_FPEM_EVIDENCE_WEIGHT=${MINICPM_FPEM_EVIDENCE_WEIGHT:-0.45}
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29911}

RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_adaptive/streamingbench_minicpmv46_full_progressive_evidence_memory_recent6_m${ADAPTIVE_MEMORY_ANCHORS}_s${ADAPTIVE_MEMORY_SEARCH_CHUNKS}_ctx${ADAPTIVE_CONTEXT_TIME}_d8"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export SB_RESULT_DIR="$RESULT_DIR"

bash main_experiments/minicpm_v46/streamingbench/run_adaptive.sh
