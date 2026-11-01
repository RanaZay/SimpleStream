#!/bin/bash
# Queue a launcher for the held nodes (node_worker.sh) instead of sbatch.
#   bash main_experiments/tools/node_submit.sh <gpus> "<VAR=value,VAR=value>" <launcher.sh> [name]
# Example:
#   bash main_experiments/tools/node_submit.sh 4 "PRISM_VOI_LAM=0.1,PRISM_VOI_TRACE=1" main_experiments/minicpm_v46/ovo/submit_prism_voi_amd.sh voi_ovo
# The run gets the GPU count it asks for; its result dir is named by a run id printed in the node log.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
gpus="$1"; vars="$2"; launcher="$3"; name="${4:-$(basename "$launcher" .sh)}"
[[ -f "$REPO_ROOT/$launcher" ]] || { echo "No launcher $launcher" >&2; exit 2; }
mkdir -p "$REPO_ROOT/node_queue/pending"
file="$REPO_ROOT/node_queue/pending/$(date +%Y%m%d_%H%M%S)_${name}.sh"
{
    echo "# gpus: $gpus"
    echo "export NUM_PROCESSES=$gpus PROGRESSIVE_ARBITRATION_TMP_ROOT=/tmp/\$USER"
    IFS=',' read -ra pairs <<< "$vars"
    for kv in "${pairs[@]}"; do [[ -n "$kv" ]] && printf 'export %q\n' "$kv"; done
    echo "exec bash $launcher"
} > "$file.tmp"
mv "$file.tmp" "$file"
echo "queued $(basename "$file") ($gpus GPUs)"
