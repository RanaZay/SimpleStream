#!/bin/bash
#SBATCH --job-name=SimpleStream_node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --partition=faculty
#SBATCH --qos=fkqos
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Holds one node and runs queued launchers on it, so held nodes are used without `srun` from the login shell
# (which is killed at logout). Every held node takes work from the same queue; queue runs with node_submit.sh.
#   node_queue/pending/*.sh   waiting (oldest first); first line "# gpus: N"
#   node_queue/running/<node job>/  claimed; node_queue/done/<node job>/  finished (exit code appended)
#   IDLE_EXIT  seconds with nothing running and nothing queued before the node is released (default 7200; 0 = never)
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
Q="$REPO_ROOT/node_queue"
mkdir -p "$Q/pending" "$Q/running/$SLURM_JOB_ID" "$Q/done/$SLURM_JOB_ID" logs
IFS=, read -ra slurm_gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"; NODE_GPUS=${#slurm_gpus[@]}
IDLE_EXIT="${IDLE_EXIT:-7200}"
declare -A PID_GPUS PID_FILE
FREE=$(seq -s ' ' 0 $((NODE_GPUS - 1)))
n=0; idle_since=$(date +%s)
echo "node $SLURM_JOB_ID on $(hostname): $NODE_GPUS GPUs, queue $Q"

while true; do
    for pid in "${!PID_GPUS[@]}"; do  # Reap finished runs and return their GPUs.
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"; code=$?
            FREE="$FREE ${PID_GPUS[$pid]//,/ }"
            echo "# exit code $code" >> "${PID_FILE[$pid]}"
            mv "${PID_FILE[$pid]}" "$Q/done/$SLURM_JOB_ID/"
            echo "$(date +%T) finished $(basename "${PID_FILE[$pid]}") exit $code"
            unset "PID_GPUS[$pid]" "PID_FILE[$pid]"
        fi
    done
    next=$(ls -1tr "$Q/pending"/*.sh 2>/dev/null | head -1)
    if [[ -n "$next" ]]; then
        need=$(sed -n '1s/^# gpus: *//p' "$next"); need=${need:-1}
        read -ra free <<< "$FREE"
        if (( ${#free[@]} >= need )); then
            mine="$Q/running/$SLURM_JOB_ID/$(basename "$next")"
            if mv "$next" "$mine" 2>/dev/null; then  # Atomic claim: another node may race for the same file.
                gpus=$(IFS=,; echo "${free[*]:0:need}"); FREE="${free[*]:need}"
                n=$((n + 1)); run_id=$((SLURM_JOB_ID * 100 + n))  # Numeric and unique: launchers name dirs and ports by it.
                log="logs/$(basename "$mine" .sh)-$run_id.out"
                SLURM_JOB_ID=$run_id CUDA_VISIBLE_DEVICES=$gpus HIP_VISIBLE_DEVICES=$gpus REPO_ROOT="$REPO_ROOT" \
                    bash "$mine" > "$log" 2>&1 &
                PID_GPUS[$!]=$gpus; PID_FILE[$!]=$mine
                echo "$(date +%T) started $(basename "$mine") as $run_id on GPUs $gpus -> $log"
                continue
            fi
        fi
    fi
    if (( ${#PID_GPUS[@]} > 0 )) || [[ -n "$next" ]]; then
        idle_since=$(date +%s)
    elif (( IDLE_EXIT > 0 && $(date +%s) - idle_since > IDLE_EXIT )); then
        echo "$(date +%T) idle for ${IDLE_EXIT}s; releasing the node"; exit 0
    fi
    sleep 20
done
