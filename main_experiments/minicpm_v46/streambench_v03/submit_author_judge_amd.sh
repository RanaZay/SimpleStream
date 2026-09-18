#!/bin/bash
#SBATCH --job-name=streambench_llama3_judge
#SBATCH --partition=faculty
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=logs/streambench_llama3_judge-%j.out
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH:-$REPO_ROOT/.conda/envs/stream35}/bin/python}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_home}"
export HF_ENABLE_PARALLEL_LOADING=false HF_DEACTIVATE_ASYNC_LOAD=1
export TMPDIR="${TMPDIR:-/tmp/$USER/streambench_judge_${SLURM_JOB_ID:-local}}"
mkdir -p "$TMPDIR" logs
exec "$PYTHON_BIN" -m main_experiments.minicpm_v46.streambench_v03.evaluate_saved_answers "$@"
