#!/bin/bash
#SBATCH --job-name=pu_longvideobench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --partition=faculty
#SBATCH --qos=fkqos
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
BENCHMARK=longvideobench
DEFAULT_PROCESSES=4
export PRISM_UNIFIED=1  # Format-agnostic PRISM: one controller for MCQ and open-ended questions.
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/main_experiments/minicpm_v46/progressive_arbitration_amd_common.sh"
