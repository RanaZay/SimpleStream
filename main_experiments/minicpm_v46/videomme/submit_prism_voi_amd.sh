#!/bin/bash
#SBATCH --job-name=voi_videomme
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --partition=faculty
#SBATCH --qos=fkqos
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
BENCHMARK=videomme
DEFAULT_PROCESSES=4
export PRISM_VOI=1  # PRISM-VoI (lambda via PRISM_VOI_LAM; PRISM_VOI_TRACE=1 records every candidate for offline lambda replay).
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/main_experiments/minicpm_v46/progressive_arbitration_amd_common.sh"
