#!/bin/bash
#SBATCH --job-name=oasis_ovo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --partition=faculty
#SBATCH --qos=fkqos
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Recent-6 + OASIS memory baseline; same node shape and launcher as the PRISM OVO run.
# Defaults to the OASIS paper's OVO protocol (Backward + Real-Time); export OVO_SPLITS to change it.
BENCHMARK=ovo
DEFAULT_PROCESSES=4
export PRISM_OASIS=1
if [[ "${1:-}" != "--check" && ! -d "${OASIS_EMBEDDING_MODEL:-}" ]]; then
    echo "Set OASIS_EMBEDDING_MODEL to a local Qwen3-Embedding-0.6B snapshot directory" >&2
    exit 2
fi
export OVO_SPLITS="${OVO_SPLITS:-backward,realtime}"
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/main_experiments/minicpm_v46/progressive_arbitration_amd_common.sh"
