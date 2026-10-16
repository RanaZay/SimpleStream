#!/bin/bash
#SBATCH --job-name=oasis_paper_streamingbench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --partition=faculty
#SBATCH --qos=fkqos
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# OASIS with the paper NowWindow (8 s at 2 fps, 16 frames) + 32 s buffer + event forest; same node shape and launcher as the PRISM StreamingBench run.
# OASIS_EMBEDDING_MODEL must point at a local Qwen3-Embedding-0.6B snapshot (see docs/oasis_baseline.md).
BENCHMARK=streamingbench
DEFAULT_PROCESSES=4
export PRISM_OASIS=paper
if [[ "${1:-}" != "--check" && ! -d "${OASIS_EMBEDDING_MODEL:-}" ]]; then
    echo "Set OASIS_EMBEDDING_MODEL to a local Qwen3-Embedding-0.6B snapshot directory" >&2
    exit 2
fi
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/main_experiments/minicpm_v46/progressive_arbitration_amd_common.sh"
