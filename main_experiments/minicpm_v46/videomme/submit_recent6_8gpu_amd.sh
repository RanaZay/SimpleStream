#!/bin/bash
#SBATCH --job-name=videomme_recent6_d8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=700G
#SBATCH --time=24:00:00
#SBATCH --qos=skqos
#SBATCH --partition=faculty
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source ~/.bashrc

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
conda activate "${CONDA_ENV_PATH:-$REPO_ROOT/.conda/envs/stream35}"

export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export ROCM_HOME=${ROCM_HOME:-/opt/rocm}
export PATH="${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MIOPEN_DISABLE_CACHE=1
export PYTORCH_TUNABLEOP_ENABLED=0
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
export MINICPM_DOWNSAMPLE_MODE=${MINICPM_DOWNSAMPLE_MODE:-16x}
export MINICPM_MAX_SLICE_NUMS=${MINICPM_MAX_SLICE_NUMS:-1}
export MINICPM_PROFILE_COMPONENTS=${MINICPM_PROFILE_COMPONENTS:-1}
export MINICPM_SERIALIZE_MODEL_LOAD=${MINICPM_SERIALIZE_MODEL_LOAD:-1}
export MINICPM_MODEL_LOAD_TIMEOUT=${MINICPM_MODEL_LOAD_TIMEOUT:-7200}
export HF_ENABLE_PARALLEL_LOADING=${HF_ENABLE_PARALLEL_LOADING:-false}
export HF_PARALLEL_LOADING_WORKERS=${HF_PARALLEL_LOADING_WORKERS:-1}
export HF_DEACTIVATE_ASYNC_LOAD=${HF_DEACTIVATE_ASYNC_LOAD:-1}
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-65536}
export MINICPM_SEED=${MINICPM_SEED:-42}
export PYTHONHASHSEED=${MINICPM_SEED}
export HF_HOME=${HF_HOME:-$REPO_ROOT/.hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}

mkdir -p logs .cache/miopen .cache/torch_kernels
export MIOPEN_USER_DB_PATH="$REPO_ROOT/.cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$REPO_ROOT/.cache/miopen"
export PYTORCH_KERNEL_CACHE_PATH="$REPO_ROOT/.cache/torch_kernels"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}
export NUM_PROCESSES=${NUM_PROCESSES:-8}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29971}
export QWEN_EXACT_RECENT_DECODE=${QWEN_EXACT_RECENT_DECODE:-0}

SAMPLE_TAG="full"
if [[ -n "${VIDEOMME_MAX_SAMPLES:-${MAX_SAMPLES:-}}" ]]; then
    SAMPLE_TAG="n${VIDEOMME_MAX_SAMPLES:-${MAX_SAMPLES}}"
fi
RESULT_DIR="${VIDEOMME_RESULT_DIR:-$REPO_ROOT/main_experiments/results/repro_videomme/videomme_minicpmv46_recent6_${SAMPLE_TAG}_d8}"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi

echo "=== ENV CHECK ==="
which python
python -V
python -c "import torch; print('torch=', torch.__version__); print('hip=', torch.version.hip); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
python -c "import pandas, pyarrow, transformers, accelerate; print('pandas=', pandas.__version__); print('pyarrow=', pyarrow.__version__); print('transformers=', transformers.__version__); print('accelerate=', accelerate.__version__)"
echo "VIDEOMME_RESULT_DIR=$RESULT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "=== END ENV CHECK ==="

COMMON_ARGS=(
    --annotation-parquet "${VIDEOMME_ANNOTATION_PARQUET:-$REPO_ROOT/data/video_mme/videomme/test-00000-of-00001.parquet}"
    --video-dir "${VIDEOMME_VIDEO_DIR:-$REPO_ROOT/data/video_mme/videos}"
    --output-dir "$RESULT_DIR"
    --mode recent6
    --qa-model "${MINICPM_QA_MODEL:-openbmb/MiniCPM-V-4.6}"
    --recent-frames-only 6
    --chunk-duration 1.0
    --fps 1.0
    --max-qa-tokens 256
)
if [[ -n "${VIDEOMME_MAX_SAMPLES:-${MAX_SAMPLES:-}}" ]]; then
    COMMON_ARGS+=(--max-samples "${VIDEOMME_MAX_SAMPLES:-${MAX_SAMPLES}}")
fi

python -m accelerate.commands.launch \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --multi_gpu \
    --mixed_precision bf16 \
    main_experiments/minicpm_v46/videomme/eval_videomme_dist.py "${COMMON_ARGS[@]}"
