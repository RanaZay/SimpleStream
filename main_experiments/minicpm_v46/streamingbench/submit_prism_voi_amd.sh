#!/bin/bash
#SBATCH --job-name=voi_streamingbench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --partition=faculty
#SBATCH --qos=fkqos
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
BENCHMARK=streamingbench
DEFAULT_PROCESSES=4
export PRISM_VOI=1  # PRISM-VoI: memory frames kept only if they remove > lambda nats of semantic entropy (PRISM_VOI_LAM).
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/main_experiments/minicpm_v46/progressive_arbitration_amd_common.sh"
