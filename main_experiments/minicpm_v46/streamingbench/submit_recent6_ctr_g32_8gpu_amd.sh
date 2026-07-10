#!/bin/bash
#SBATCH --job-name=minicpmv46_sb_r6ctr32_d8
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
export MINICPM_QA_DEVICE=${MINICPM_QA_DEVICE:-}
export MINICPM_SERIALIZE_MODEL_LOAD=${MINICPM_SERIALIZE_MODEL_LOAD:-1}
export MINICPM_MODEL_LOAD_TIMEOUT=${MINICPM_MODEL_LOAD_TIMEOUT:-7200}
export HF_ENABLE_PARALLEL_LOADING=false
export HF_PARALLEL_LOADING_WORKERS=1
export HF_DEACTIVATE_ASYNC_LOAD=1
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-65536}
export MAX_SAMPLES=${MAX_SAMPLES:-}
export RESUME=${RESUME:-0}

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

export RECENT_FRAMES_ONLY=6
export CTR_BUDGET=32
export CTR_TAU=0.9

echo "=== ENV CHECK ==="
which python
python -V
python -c "import torch; print('torch=', torch.__version__); print('hip=', torch.version.hip); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
python -c "import transformers, accelerate; print('transformers=', transformers.__version__); print('accelerate=', accelerate.__version__)"
echo "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
echo "MINICPM_DOWNSAMPLE_MODE=$MINICPM_DOWNSAMPLE_MODE"
echo "MINICPM_MAX_SLICE_NUMS=$MINICPM_MAX_SLICE_NUMS"
echo "MINICPM_PROFILE_COMPONENTS=$MINICPM_PROFILE_COMPONENTS"
echo "MINICPM_QA_DEVICE=$MINICPM_QA_DEVICE"
echo "MINICPM_SERIALIZE_MODEL_LOAD=$MINICPM_SERIALIZE_MODEL_LOAD"
echo "MINICPM_MODEL_LOAD_TIMEOUT=$MINICPM_MODEL_LOAD_TIMEOUT"
echo "HF_ENABLE_PARALLEL_LOADING=$HF_ENABLE_PARALLEL_LOADING"
echo "HF_PARALLEL_LOADING_WORKERS=$HF_PARALLEL_LOADING_WORKERS"
echo "HF_DEACTIVATE_ASYNC_LOAD=$HF_DEACTIVATE_ASYNC_LOAD"
echo "DECORD_EOF_RETRY_MAX=$DECORD_EOF_RETRY_MAX"
echo "RECENT_FRAMES_ONLY=$RECENT_FRAMES_ONLY"
echo "CTR_BUDGET=$CTR_BUDGET"
echo "CTR_TAU=$CTR_TAU"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "RESUME=$RESUME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "MINICPM_LAUNCH_MODE=8 data-parallel processes, direct per-rank GPU load"
echo "=== END ENV CHECK ==="

RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_recent6_ctr/streamingbench_minicpmv46_recent6_ctr_g32_tau0.9_d8"
if [[ -n "$MAX_SAMPLES" ]]; then
    RESULT_DIR="${RESULT_DIR}_smoke${MAX_SAMPLES}"
fi
if [[ "$RESUME" != "1" ]]; then
    ts=$(date +%Y%m%d_%H%M%S)
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi

PYTHON_BIN=$(which python) \
MINICPM_QA_DEVICE="$MINICPM_QA_DEVICE" \
SB_RESULT_DIR="$RESULT_DIR" \
NUM_PROCESSES=8 \
MAX_SAMPLES="$MAX_SAMPLES" \
RECENT_FRAMES_ONLY="$RECENT_FRAMES_ONLY" \
CTR_BUDGET="$CTR_BUDGET" \
CTR_TAU="$CTR_TAU" \
bash main_experiments/minicpm_v46/streamingbench/run_recent_ctr.sh
