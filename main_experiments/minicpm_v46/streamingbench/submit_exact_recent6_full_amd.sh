#!/bin/bash
#SBATCH --job-name=sb_exact_recent6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --qos=skqos
#SBATCH --partition=faculty
#SBATCH --output=/vast/users/salman.khan/SimpleStream/logs/%x-%j.out

set -euo pipefail

export MINICPM_RECENT_SAMPLER=current_recent6

bash main_experiments/minicpm_v46/streamingbench/submit_recent_sampler_current6_8gpu_amd.sh
