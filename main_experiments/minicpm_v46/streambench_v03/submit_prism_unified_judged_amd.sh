#!/bin/bash
#SBATCH --job-name=pu_streambench_judged
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --partition=faculty
#SBATCH --qos=fkqos
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
# Full StreamBench v0.3 run of format-agnostic PRISM, then the author LLaMA-3-8B judge in the same job.
# Judge prerequisites are checked before any inference, so a missing judge cannot waste the run.
#   JUDGE_MODEL        local Meta-Llama-3-8B-Instruct snapshot (default: first snapshot under $HF_HOME)
#   STREAMCHAT_SOURCE  pinned hmxiong/StreamChat checkout (default: $REPO_ROOT/.cache/streamchat_judge)
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
export REPO_ROOT HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_home}"
STREAMCHAT_SOURCE="${STREAMCHAT_SOURCE:-$REPO_ROOT/.cache/streamchat_judge}"
STREAMCHAT_COMMIT=f8bcd22ef68cb69f59bcae29ed9fe93f75c077be
if [[ -z "${JUDGE_MODEL:-}" ]]; then
    JUDGE_MODEL="$(ls -d "$HF_HOME"/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/* 2>/dev/null | head -1 || true)"
fi
if [[ ! -f "${JUDGE_MODEL:-/missing}/config.json" ]]; then
    echo "JUDGE_MODEL must be a local Meta-Llama-3-8B-Instruct snapshot (got '${JUDGE_MODEL:-}')" >&2
    exit 2
fi
if [[ "$(git -C "$STREAMCHAT_SOURCE" rev-parse HEAD 2>/dev/null)" != "$STREAMCHAT_COMMIT" ]]; then
    echo "STREAMCHAT_SOURCE must be hmxiong/StreamChat at $STREAMCHAT_COMMIT (got '$STREAMCHAT_SOURCE')" >&2
    exit 2
fi
echo "JUDGE_MODEL=$JUDGE_MODEL"
echo "STREAMCHAT_SOURCE=$STREAMCHAT_SOURCE"

# 1) Inference. The launcher execs the runner in a child shell, so control returns here afterwards.
bash main_experiments/minicpm_v46/streambench_v03/submit_prism_unified_amd.sh
RESULTS="reports/progressive_arbitration/streambench_v03/prism_unified_${SLURM_JOB_ID}/streambench_v0_3_smoke_results.json"
if [[ ! -f "$RESULTS" ]]; then
    echo "Inference finished without $RESULTS" >&2
    exit 1
fi

# 2) Judge. Offline evaluation only; it never enters the latency tables.
export TMPDIR="/tmp/$USER/streambench_judge_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
OUT="reports/streambench_judging/prism_unified_${SLURM_JOB_ID}"
bash main_experiments/minicpm_v46/streambench_v03/submit_author_judge_amd.sh \
    --results "$RESULTS" --method prism --source "$STREAMCHAT_SOURCE" \
    --judge-model "$JUDGE_MODEL" --output-dir "$OUT" --allow-inference-errors
echo "JUDGED_SUMMARY=$OUT/summary.md"
cat "$OUT/summary.md"
