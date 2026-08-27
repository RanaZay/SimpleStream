#!/bin/bash
#SBATCH --job-name=sb_counting_event_ledger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
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

SB_ANNOTATIONS=${SB_ANNOTATIONS:-data/streamingbench/questions_real.json}
SB_VIDEO_DIR=${SB_VIDEO_DIR:-data/streamingbench/videos}
RECENT_RESULTS=${RECENT_RESULTS:-main_experiments/results/repro_recent_sampler/streamingbench_minicpmv46_current_recent6_exact6_d8}
PRISM_RESULTS=${PRISM_RESULTS:-reports/prism_retrieval_variants/streamingbench_full_prism_clip_mmr_candidate_override_guarded_rollback_g0p2995_controller}
OUT_DIR=${OUT_DIR:-reports/prism_event_ledger/streamingbench_counting}

export MINICPM_EVENT_LEDGER_CLIP_MODEL=${MINICPM_EVENT_LEDGER_CLIP_MODEL:-openai/clip-vit-base-patch32}
export MINICPM_EVENT_LEDGER_CLIP_DEVICE=${MINICPM_EVENT_LEDGER_CLIP_DEVICE:-cuda:0}
export MINICPM_EVENT_LEDGER_CLIP_CHANGE_WEIGHT=${MINICPM_EVENT_LEDGER_CLIP_CHANGE_WEIGHT:-0.70}
export MINICPM_EVENT_LEDGER_VISUAL_CHANGE_WEIGHT=${MINICPM_EVENT_LEDGER_VISUAL_CHANGE_WEIGHT:-0.30}
export MINICPM_EVENT_LEDGER_BOUNDARY_THRESHOLD=${MINICPM_EVENT_LEDGER_BOUNDARY_THRESHOLD:-0.35}
export MINICPM_EVENT_LEDGER_MIN_EVENT_SECONDS=${MINICPM_EVENT_LEDGER_MIN_EVENT_SECONDS:-1.0}
export MINICPM_EVENT_LEDGER_MAX_EVENT_SECONDS=${MINICPM_EVENT_LEDGER_MAX_EVENT_SECONDS:-5.0}

echo "=== ENV CHECK ==="
which python
python -V
python -c "import torch; print('torch=', torch.__version__); print('hip=', torch.version.hip); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "SB_ANNOTATIONS=$SB_ANNOTATIONS"
echo "SB_VIDEO_DIR=$SB_VIDEO_DIR"
echo "RECENT_RESULTS=$RECENT_RESULTS"
echo "PRISM_RESULTS=$PRISM_RESULTS"
echo "OUT_DIR=$OUT_DIR"
echo "MINICPM_EVENT_LEDGER_CLIP_MODEL=$MINICPM_EVENT_LEDGER_CLIP_MODEL"
echo "MINICPM_EVENT_LEDGER_CLIP_DEVICE=$MINICPM_EVENT_LEDGER_CLIP_DEVICE"
echo "=== END ENV CHECK ==="

python main_experiments/tools/analyze_streamingbench_event_ledger_counting.py \
  --annotations "$SB_ANNOTATIONS" \
  --video-dir "$SB_VIDEO_DIR" \
  --recent "$RECENT_RESULTS" \
  --prism "$PRISM_RESULTS" \
  --out-dir "$OUT_DIR"
