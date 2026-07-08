#!/bin/bash
#SBATCH --job-name=qwen35_4b_sb_2048
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
unset QWEN35_ENABLE_THINKING
unset QWEN35_DEBUG_THINKING_PREFIX

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
echo "QWEN35_ENABLE_THINKING=${QWEN35_ENABLE_THINKING:-UNSET}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "MAX_QA_TOKENS=2048"
echo "=== END ENV CHECK ==="

RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_recent4/streamingbench_qwen35_4b_recent4_max2048"
ts=$(date +%Y%m%d_%H%M%S)
mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true

PYTHON_BIN=$(which python)

"$PYTHON_BIN" main_experiments/qwen/evals/eval_qwen35vl_streamingbench.py \
    --anno-path "$REPO_ROOT/data/streamingbench/questions_real.json" \
    --video-dir "$REPO_ROOT/data/streamingbench/videos" \
    --qa-model "Qwen/Qwen3.5-4B" \
    --qa-device auto \
    --top-k 0 \
    --recent-frames-only 4 \
    --chunk-duration 1.0 \
    --fps 1.0 \
    --max-qa-tokens 2048 \
    --output-dir "$RESULT_DIR"
