#!/bin/bash
#SBATCH --job-name=voi_calibration
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --partition=faculty
#SBATCH --qos=fkqos
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# PRISM-VoI calibration study (docs/prism_voi_design.md, step 1). BENCH=streamingbench or streambench; N questions (default 300).
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
: "${BENCH:?Set BENCH=streamingbench or BENCH=streambench}"
PY="${PYTHON_BIN:-$REPO_ROOT/.conda/envs/stream35/bin/python}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_home}"
export HF_ENABLE_PARALLEL_LOADING=false HF_PARALLEL_LOADING_WORKERS=1 HF_DEACTIVATE_ASYNC_LOAD=1
export TMPDIR="/tmp/$USER/voi_calibration_${SLURM_JOB_ID:-local}_$BENCH"
export MIOPEN_USER_DB_PATH="$TMPDIR/miopen" MIOPEN_CUSTOM_CACHE_DIR="$TMPDIR/miopen" PYTORCH_KERNEL_CACHE_PATH="$TMPDIR/torch_kernels"
mkdir -p logs "$TMPDIR/miopen" "$TMPDIR/torch_kernels"
exec "$PY" main_experiments/tools/voi_calibration.py --benchmark "$BENCH" --n "${N:-300}" \
    --output "reports/voi_calibration/${BENCH}_${SLURM_JOB_ID:-local}"
