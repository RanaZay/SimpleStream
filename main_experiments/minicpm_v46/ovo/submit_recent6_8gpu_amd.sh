#!/bin/bash
#SBATCH --job-name=minicpmv46_ovo_recent6_d8
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
export MINICPM_SERIALIZE_MODEL_LOAD=${MINICPM_SERIALIZE_MODEL_LOAD:-1}
export MINICPM_MODEL_LOAD_TIMEOUT=${MINICPM_MODEL_LOAD_TIMEOUT:-7200}
export HF_ENABLE_PARALLEL_LOADING=false
export HF_PARALLEL_LOADING_WORKERS=1
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

echo "=== ENV CHECK ==="
which python
python -V
python -c "import torch; print('torch=', torch.__version__); print('hip=', torch.version.hip); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
python -c "import transformers, accelerate; print('transformers=', transformers.__version__); print('accelerate=', accelerate.__version__)"
echo "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
echo "MINICPM_DOWNSAMPLE_MODE=$MINICPM_DOWNSAMPLE_MODE"
echo "MINICPM_MAX_SLICE_NUMS=$MINICPM_MAX_SLICE_NUMS"
echo "MINICPM_PROFILE_COMPONENTS=$MINICPM_PROFILE_COMPONENTS"
echo "MINICPM_SERIALIZE_MODEL_LOAD=$MINICPM_SERIALIZE_MODEL_LOAD"
echo "MINICPM_MODEL_LOAD_TIMEOUT=$MINICPM_MODEL_LOAD_TIMEOUT"
echo "HF_ENABLE_PARALLEL_LOADING=$HF_ENABLE_PARALLEL_LOADING"
echo "HF_PARALLEL_LOADING_WORKERS=$HF_PARALLEL_LOADING_WORKERS"
echo "DECORD_EOF_RETRY_MAX=$DECORD_EOF_RETRY_MAX"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "RECENT_FRAMES_ONLY=6"
echo "=== END ENV CHECK ==="

RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_recent6/ovo_minicpmv46_recent6_d8"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi

PYTHON_BIN=$(which python) \
OVO_RESULT_DIR="$RESULT_DIR" \
NUM_PROCESSES=8 \
RECENT_FRAMES_ONLY=6 \
bash main_experiments/minicpm_v46/ovo/run_recent4.sh
