#!/bin/bash
#SBATCH --job-name=minicpmv46_sb100_prism_retrieval_variant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
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
export HF_ENABLE_PARALLEL_LOADING=false
export HF_PARALLEL_LOADING_WORKERS=1
export HF_DEACTIVATE_ASYNC_LOAD=1
export MIOPEN_DISABLE_CACHE=1
export PYTORCH_TUNABLEOP_ENABLED=0
export MINICPM_SEED=${MINICPM_SEED:-42}
export PYTHONHASHSEED=${MINICPM_SEED}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}

REPO_ROOT=/vast/users/salman.khan/SimpleStream
cd "$REPO_ROOT"
mkdir -p logs .cache/miopen .cache/torch_kernels
export MIOPEN_USER_DB_PATH="$REPO_ROOT/.cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$REPO_ROOT/.cache/miopen"
export PYTORCH_KERNEL_CACHE_PATH="$REPO_ROOT/.cache/torch_kernels"

export ADAPTIVE_MODE=progressive_sufficiency_memory
export ADAPTIVE_MIN_WINDOW=6
export ADAPTIVE_MID_WINDOW=6
export ADAPTIVE_MAX_WINDOW=6
export ADAPTIVE_CONTEXT_TIME=70
export ADAPTIVE_MEMORY_ANCHORS=3
export ADAPTIVE_MEMORY_SEARCH_CHUNKS=64
export RECENT_FRAMES_ONLY=6
export RECENT_SAMPLER_FPS=4.0
export NUM_PROCESSES=${NUM_PROCESSES:-1}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29961}

export MINICPM_PSM_RETRIEVAL_VARIANT=${MINICPM_PSM_RETRIEVAL_VARIANT:-clip_question_options}
export MINICPM_PSM_MMR_LAMBDA=${MINICPM_PSM_MMR_LAMBDA:-0.80}
export MINICPM_PSM_HISTORY_SEARCH_CHUNKS=64
export MINICPM_PSM_HISTORY_CANDIDATE_POOL=12
export MINICPM_PSM_MAX_MEMORY_FRAMES=3
export MINICPM_PSM_SUFFICIENCY_THRESHOLD=0.62
export MINICPM_PSM_MARGIN_WEIGHT=0.50
export MINICPM_PSM_ENTROPY_WEIGHT=0.20
export MINICPM_PSM_VISUAL_SUPPORT_WEIGHT=0.30
export MINICPM_PSM_ASSERT_TEMPORAL_ALIGNMENT=1

RESULT_DIR="$REPO_ROOT/reports/prism_retrieval_variants/streamingbench_100_prism_${MINICPM_PSM_RETRIEVAL_VARIANT}"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export SB_RESULT_DIR="$RESULT_DIR"
export MAX_SAMPLES=${MAX_SAMPLES:-100}

echo "=== ENV CHECK ==="
which python
python -V
python -c "import torch; print('torch=', torch.__version__); print('hip=', torch.version.hip); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
echo "MINICPM_PSM_RETRIEVAL_VARIANT=$MINICPM_PSM_RETRIEVAL_VARIANT"
echo "SB_RESULT_DIR=$SB_RESULT_DIR"
echo "=== END ENV CHECK ==="

COMMON_ARGS=(
  --anno-path data/streamingbench/questions_real.json
  --video-dir data/streamingbench/videos
  --qa-model "openbmb/MiniCPM-V-4.6"
  --top-k 0
  --recent-frames-only "${RECENT_FRAMES_ONLY}"
  --chunk-duration 1.0
  --fps 1.0
  --context-time "${ADAPTIVE_CONTEXT_TIME}"
  --max-qa-tokens 256
  --adaptive-mode "${ADAPTIVE_MODE}"
  --adaptive-min-window "${ADAPTIVE_MIN_WINDOW}"
  --adaptive-mid-window "${ADAPTIVE_MID_WINDOW}"
  --adaptive-max-window "${ADAPTIVE_MAX_WINDOW}"
  --adaptive-memory-anchors "${ADAPTIVE_MEMORY_ANCHORS}"
  --adaptive-memory-search-chunks "${ADAPTIVE_MEMORY_SEARCH_CHUNKS}"
  --max-samples "${MAX_SAMPLES}"
  --output-dir "${SB_RESULT_DIR}"
)

PYTHON_BIN=$(which python)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE}" \
MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS}" \
MINICPM_PROFILE_COMPONENTS="${MINICPM_PROFILE_COMPONENTS}" \
HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING}" \
HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS}" \
HF_DEACTIVATE_ASYNC_LOAD="${HF_DEACTIVATE_ASYNC_LOAD}" \
MINICPM_SEED="${MINICPM_SEED}" \
PYTHONHASHSEED="${MINICPM_SEED}" \
QWEN_EXACT_RECENT_DECODE=0 \
"${PYTHON_BIN}" main_experiments/minicpm_v46/streamingbench/eval_prism_retrieval_variant_dist.py "${COMMON_ARGS[@]}"
