#!/bin/bash
#SBATCH --job-name=streambench_v03_recent6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --qos=skqos
#SBATCH --partition=faculty
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"

export STREAMBENCH_V03_MAX_VIDEOS=0
export STREAMBENCH_V03_MAX_QUESTIONS=0
export STREAMBENCH_V03_METHODS=recent6
export STREAMBENCH_V03_OUT_DIR=${STREAMBENCH_V03_OUT_DIR:-reports/streambench_v0_3/recent6_full}

bash main_experiments/minicpm_v46/streambench_v03/submit_streambench_v03_smoke_amd.sh
