#!/bin/bash
#SBATCH --job-name=minicpmv46_ovo_all_d8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
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
export MINICPM_PROFILE_COMPONENTS=${MINICPM_PROFILE_COMPONENTS:-0}
export MINICPM_AUTO_MAX_MEMORY_PER_GPU=${MINICPM_AUTO_MAX_MEMORY_PER_GPU:-60GiB}
export HF_ENABLE_PARALLEL_LOADING=false
export HF_PARALLEL_LOADING_WORKERS=1

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
echo "MINICPM_AUTO_MAX_MEMORY_PER_GPU=$MINICPM_AUTO_MAX_MEMORY_PER_GPU"
echo "HF_ENABLE_PARALLEL_LOADING=$HF_ENABLE_PARALLEL_LOADING"
echo "HF_PARALLEL_LOADING_WORKERS=$HF_PARALLEL_LOADING_WORKERS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "MINICPM_LAUNCH_MODE=single-process device_map=auto over 8 visible GPUs"
echo "=== END ENV CHECK ==="

RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_allframes/ovo_minicpmv46_allframes_fps1_mp8"
ts=$(date +%Y%m%d_%H%M%S)
mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true

PYTHON_BIN=$(which python) \
bash main_experiments/run_repro_minicpmv46_ovo_allframes_mp8.sh
