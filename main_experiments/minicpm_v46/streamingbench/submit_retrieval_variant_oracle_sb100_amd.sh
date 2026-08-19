#!/bin/bash
#SBATCH --job-name=minicpmv46_sb100_retrieval_oracle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
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
mkdir -p logs reports/prism_retrieval_variants

export MINICPM_EXACT_RECENT_CANDIDATE_FPS=4.0
export MINICPM_PSM_HISTORY_SEARCH_CHUNKS=64
export MINICPM_PSM_HISTORY_CANDIDATE_POOL=12
export MINICPM_PSM_MAX_MEMORY_FRAMES=3
export MINICPM_PSM_SUFFICIENCY_THRESHOLD=0.62
export MINICPM_PSM_MARGIN_WEIGHT=0.50
export MINICPM_PSM_ENTROPY_WEIGHT=0.20
export MINICPM_PSM_VISUAL_SUPPORT_WEIGHT=0.30
export MINICPM_PSM_ASSERT_TEMPORAL_ALIGNMENT=1

SAMPLE_SOURCE="${SAMPLE_SOURCE:-$REPO_ROOT/reports/prism_temporal_fix_diag/streamingbench_100}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/reports/prism_retrieval_variants/streamingbench_100}"

echo "=== ENV CHECK ==="
which python
python -V
python -c "import torch; print('torch=', torch.__version__); print('hip=', torch.version.hip); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
echo "SAMPLE_SOURCE=$SAMPLE_SOURCE"
echo "OUT_DIR=$OUT_DIR"
echo "=== END ENV CHECK ==="

PYTHONNOUSERSITE=1 python main_experiments/tools/run_streamingbench_retrieval_variant_oracle.py \
  --sample-source "$SAMPLE_SOURCE" \
  --annotations data/streamingbench/questions_real.json \
  --video-dir data/streamingbench/videos \
  --out-dir "$OUT_DIR" \
  --qa-device cuda:0 \
  --context-time 70 \
  --history-search-chunks 64 \
  --candidate-pool 12 \
  --recent-window 6
