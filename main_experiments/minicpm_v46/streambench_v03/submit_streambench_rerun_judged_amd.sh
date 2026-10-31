#!/bin/bash
#SBATCH --job-name=streambench_rerun_judged
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --partition=faculty
#SBATCH --qos=fkqos
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Rerun only the failed questions of an existing StreamBench run, merge them back, and judge the merged answers.
#   ORIGINAL   the existing run dir, e.g. reports/progressive_arbitration/streambench_v03/shallowstream_289846
#   The method is selected exactly as for the original run (PRISM_SHALLOWSTREAM=1, PRISM_UNIFIED=1, PRISM_OASIS=paper, ...).
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
PY="${PYTHON_BIN:-$REPO_ROOT/.conda/envs/stream35/bin/python}"
: "${ORIGINAL:?Set ORIGINAL to the StreamBench run dir to repair}"
"$PY" main_experiments/tools/prepare_rerun.py "$ORIGINAL"
export STREAMBENCH_ONLY_KEYS="$REPO_ROOT/$ORIGINAL/failed_keys.json"
if [[ "$("$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$STREAMBENCH_ONLY_KEYS")" == 0 ]]; then
    echo "No failed questions in $ORIGINAL"; exit 0
fi
bash main_experiments/minicpm_v46/streambench_v03/submit_progressive_arbitration_exact_recent_amd.sh || echo "Rerun exited nonzero; merging what was answered"
RERUN=$(ls -td reports/progressive_arbitration/streambench_v03/*_"${SLURM_JOB_ID}" | head -1)
MERGED="${ORIGINAL}_fixed_${SLURM_JOB_ID}"
"$PY" main_experiments/tools/merge_streambench_rerun.py "$ORIGINAL" "$RERUN" "$MERGED"
JUDGE_MODEL="${JUDGE_MODEL:-$(ls -d "$REPO_ROOT"/.hf_home/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/* | head -1)}"
METHOD_KEY=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['results'][0]['methods'][0]['method'])" "$MERGED/streambench_v0_3_smoke_results.json")
OUT="reports/streambench_judging/$(basename "$MERGED")"
export TMPDIR="/tmp/$USER/streambench_judge_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
bash main_experiments/minicpm_v46/streambench_v03/submit_author_judge_amd.sh --results "$MERGED/streambench_v0_3_smoke_results.json" \
    --method "$METHOD_KEY" --source "$REPO_ROOT/.cache/streamchat_judge" --judge-model "$JUDGE_MODEL" --output-dir "$OUT" --allow-inference-errors
echo "JUDGED_SUMMARY=$OUT/summary.md"; cat "$OUT/summary.md"
