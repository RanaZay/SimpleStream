#!/bin/bash
#SBATCH --job-name=streambench_guarded_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --partition=faculty
#SBATCH --qos=fkqos
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH:-$REPO_ROOT/.conda/envs/stream35}/bin/python}"
export PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TMPDIR="${PRISM_TMPDIR:-/tmp/$USER/streambench_${SLURM_JOB_ID:-$$}}"
export MIOPEN_USER_DB_PATH="$TMPDIR/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$TMPDIR/miopen"
export PYTORCH_KERNEL_CACHE_PATH="$TMPDIR/torch_kernels"
mkdir -p logs "$TMPDIR/miopen-lockfiles" "$MIOPEN_USER_DB_PATH" "$PYTORCH_KERNEL_CACHE_PATH"
export MIOPEN_DISABLE_CACHE=0 PYTORCH_TUNABLEOP_ENABLED=0
export ATTN_IMPLEMENTATION=sdpa
export MINICPM_DOWNSAMPLE_MODE=16x MINICPM_MAX_SLICE_NUMS=1
export MINICPM_PROFILE_COMPONENTS=1 MINICPM_SEED=42 PYTHONHASHSEED=42
export HF_ENABLE_PARALLEL_LOADING=false HF_DEACTIVATE_ASYNC_LOAD=1
export STREAMBENCH_PRISM_MODE=progressive_sufficiency_memory_clip_mmr_candidate_override_guarded_rollback_exact_recent
export STREAMBENCH_PRISM_CLIP_MODE=candidate_override_guarded_rollback_exact_recent
export MINICPM_PSM_MAX_MEMORY_FRAMES=3
export MINICPM_PSM_HISTORY_SEARCH_CHUNKS=64 MINICPM_PSM_HISTORY_CANDIDATE_POOL=12
export MINICPM_PSM_SUFFICIENCY_THRESHOLD=0.62 MINICPM_PSM_CLIP_OVERRIDE_THRESHOLD=0.2995
export MINICPM_PSM_ARBITRATION_MIN_MARGIN=0.60 MINICPM_PSM_ARBITRATION_MAX_SUFFICIENCY_DROP=0.08
export MINICPM_PSM_TEMPORAL_BAND_MIN_SECONDS=3 MINICPM_PSM_TEMPORAL_BAND_MAX_SECONDS=30
export MINICPM_PSM_CANDIDATE_K1_DISAGREE_MAX_DISTANCE_SECONDS=10
export MINICPM_PSM_MMR_LAMBDA=0.80 MINICPM_PSM_EXACT_RECENT_PRESERVE_SOURCE_IDS=1
export MINICPM_PSM_PRINT_TRACE="${MINICPM_PSM_PRINT_TRACE:-0}"
ANNOTATIONS="${STREAMBENCH_V03_ANNOTATIONS:-$REPO_ROOT/data/streambench_v0_3/streaming_bench_v0.3.json}"
DATA_ROOT="${STREAMBENCH_V03_DATA_ROOT:-$REPO_ROOT/data/streambench_v0_3}"
OUT_DIR="${STREAMBENCH_V03_OUT_DIR:-$REPO_ROOT/reports/streambench_v0_3/prism_guarded_full_${SLURM_JOB_ID:-$$}}"
test -f "$ANNOTATIONS" || { echo "Missing annotations: $ANNOTATIONS" >&2; exit 1; }
echo "PRISM_MODE=$STREAMBENCH_PRISM_MODE"
echo "MINICPM_PSM_MAX_MEMORY_FRAMES=$MINICPM_PSM_MAX_MEMORY_FRAMES"
echo "ANNOTATIONS=$ANNOTATIONS"
echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "TMPDIR=$TMPDIR"
"$PYTHON_BIN" -c 'import torch; print("torch", torch.__version__, "HIP", torch.version.hip); assert torch.cuda.is_available(), "GPU unavailable"; print(torch.cuda.get_device_name(0))'
"$PYTHON_BIN" -u main_experiments/minicpm_v46/streambench_v03/eval_streambench_v03_smoke.py \
  --anno-path "$ANNOTATIONS" --data-root "$DATA_ROOT" \
  --methods prism --recent-window 6 --context-time 6 --qa-device cuda:0 \
  --max-videos "${STREAMBENCH_V03_MAX_VIDEOS:-0}" \
  --max-questions "${STREAMBENCH_V03_MAX_QUESTIONS:-0}" \
  --output-dir "$OUT_DIR"
