#!/bin/bash
#SBATCH --job-name=minicpmv46_sb100_prism_clip_retrieval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
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
export ATTN_IMPLEMENTATION=sdpa
export MINICPM_DOWNSAMPLE_MODE=16x
export MINICPM_MAX_SLICE_NUMS=1
export MINICPM_PROFILE_COMPONENTS=${MINICPM_PROFILE_COMPONENTS:-1}
export MINICPM_SERIALIZE_MODEL_LOAD=${MINICPM_SERIALIZE_MODEL_LOAD:-1}
export MINICPM_MODEL_LOAD_TIMEOUT=${MINICPM_MODEL_LOAD_TIMEOUT:-7200}
export HF_ENABLE_PARALLEL_LOADING=false
export HF_PARALLEL_LOADING_WORKERS=1
export HF_DEACTIVATE_ASYNC_LOAD=1
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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}

PRISM_CLIP_MODE="${PRISM_CLIP_MODE:-clip_mmr}"
case "$PRISM_CLIP_MODE" in
    clip_mmr)
        export ADAPTIVE_MODE=progressive_sufficiency_memory_clip_mmr
        MODE_TAG=clip_mmr
        ;;
    evidence_override|clip_mmr_evidence_override)
        export ADAPTIVE_MODE=progressive_sufficiency_memory_clip_mmr_evidence_override
        GAMMA_TAG="${MINICPM_PSM_EVIDENCE_OVERRIDE_GAMMA:-0.30}"
        GAMMA_TAG="${GAMMA_TAG/./p}"
        MODE_TAG="clip_mmr_evidence_override_g${GAMMA_TAG}"
        ;;
    candidate_override|clip_mmr_candidate_override)
        export ADAPTIVE_MODE=progressive_sufficiency_memory_clip_mmr_candidate_override
        GAMMA_TAG="${MINICPM_PSM_CLIP_OVERRIDE_THRESHOLD:-0.2995}"
        GAMMA_TAG="${GAMMA_TAG/./p}"
        MODE_TAG="clip_mmr_candidate_override_g${GAMMA_TAG}"
        ;;
    candidate_override_protected_rollback|clip_mmr_candidate_override_protected_rollback)
        export ADAPTIVE_MODE=progressive_sufficiency_memory_clip_mmr_candidate_override_protected_rollback
        GAMMA_TAG="${MINICPM_PSM_CLIP_OVERRIDE_THRESHOLD:-0.2995}"
        GAMMA_TAG="${GAMMA_TAG/./p}"
        MODE_TAG="clip_mmr_candidate_override_protected_rollback_g${GAMMA_TAG}"
        ;;
    clip_question_options)
        export ADAPTIVE_MODE=progressive_sufficiency_memory_clip_question_options
        MODE_TAG=clip_question_options
        ;;
    *)
        echo "[ERROR] Unknown PRISM_CLIP_MODE=$PRISM_CLIP_MODE" >&2
        exit 2
        ;;
esac

export ADAPTIVE_MIN_WINDOW=6
export ADAPTIVE_MID_WINDOW=6
export ADAPTIVE_MAX_WINDOW=6
export ADAPTIVE_CONTEXT_TIME=70
export ADAPTIVE_MEMORY_ANCHORS=3
export ADAPTIVE_MEMORY_SEARCH_CHUNKS=64
export RECENT_FRAMES_ONLY=6
export RECENT_SAMPLER_FPS=4.0
export MAX_SAMPLES=100
export NUM_PROCESSES=${NUM_PROCESSES:-1}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29981}

export MINICPM_PSM_HISTORY_SEARCH_CHUNKS=64
export MINICPM_PSM_HISTORY_CANDIDATE_POOL=12
export MINICPM_PSM_MAX_MEMORY_FRAMES=3
export MINICPM_PSM_MIN_TEMPORAL_GAP=2
export MINICPM_PSM_SUFFICIENCY_THRESHOLD=0.62
export MINICPM_PSM_MIN_EVIDENCE_GAIN=0.035
export MINICPM_PSM_NEGATIVE_GAIN_TOLERANCE=0.02
export MINICPM_PSM_MARGIN_WEIGHT=0.50
export MINICPM_PSM_ENTROPY_WEIGHT=0.20
export MINICPM_PSM_VISUAL_SUPPORT_WEIGHT=0.30
export MINICPM_PSM_MMR_LAMBDA=${MINICPM_PSM_MMR_LAMBDA:-0.80}
export MINICPM_PSM_EVIDENCE_OVERRIDE_GAMMA=${MINICPM_PSM_EVIDENCE_OVERRIDE_GAMMA:-0.30}
export MINICPM_PSM_EVIDENCE_OVERRIDE_MIN_MARGIN=${MINICPM_PSM_EVIDENCE_OVERRIDE_MIN_MARGIN:-0.10}
export MINICPM_PSM_CLIP_OVERRIDE_THRESHOLD=${MINICPM_PSM_CLIP_OVERRIDE_THRESHOLD:-0.2995}
export MINICPM_PSM_ASSERT_TEMPORAL_ALIGNMENT=1
export MINICPM_PSM_PRINT_TRACE=1

RESULT_DIR="$REPO_ROOT/reports/prism_retrieval_variants/streamingbench_100_prism_${MODE_TAG}_controller"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export SB_RESULT_DIR="$RESULT_DIR"

echo "=== ENV CHECK ==="
which python
python -V
python -c "import torch; print('torch=', torch.__version__); print('hip=', torch.version.hip); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
echo "ADAPTIVE_MODE=$ADAPTIVE_MODE"
echo "PRISM_CLIP_MODE=$PRISM_CLIP_MODE"
echo "MINICPM_PSM_EVIDENCE_OVERRIDE_GAMMA=$MINICPM_PSM_EVIDENCE_OVERRIDE_GAMMA"
echo "MINICPM_PSM_CLIP_OVERRIDE_THRESHOLD=$MINICPM_PSM_CLIP_OVERRIDE_THRESHOLD"
echo "SB_RESULT_DIR=$SB_RESULT_DIR"
echo "=== END ENV CHECK ==="

PYTHON_BIN=$(which python) bash main_experiments/minicpm_v46/streamingbench/run_prism_exact_recent.sh
