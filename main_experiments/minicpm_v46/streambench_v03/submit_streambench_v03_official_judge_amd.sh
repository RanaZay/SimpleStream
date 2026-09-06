#!/bin/bash
#SBATCH --job-name=streambench_v03_judge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --qos=skqos
#SBATCH --partition=faculty
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"

source ~/.bashrc
conda activate "${CONDA_ENV_PATH:-stream35}"

export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export ROCM_HOME=${ROCM_HOME:-/opt/rocm}
export PATH="${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR="${TMPDIR:-/tmp/${USER}/streambench_judge_${SLURM_JOB_ID:-$$}}"
export MIOPEN_DISABLE_CACHE=${MIOPEN_DISABLE_CACHE:-0}
mkdir -p logs "$TMPDIR" "$TMPDIR/miopen-lockfiles"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}

INPUT_DIR="${INPUT_DIR:-reports/streambench_v0_3_official_inputs}"
OUT_DIR="${OUT_DIR:-reports/streambench_v0_3_official_judge}"
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
JUDGE_DTYPE="${JUDGE_DTYPE:-bfloat16}"
JUDGE_DEVICE="${JUDGE_DEVICE:-cuda}"
JUDGE_LIMIT="${JUDGE_LIMIT:-0}"

echo "=== ENV CHECK ==="
which python
python -V
python -c "import torch; print('torch=', torch.__version__); print('hip=', torch.version.hip); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
echo "INPUT_DIR=$INPUT_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "JUDGE_MODEL=$JUDGE_MODEL"
echo "JUDGE_LIMIT=$JUDGE_LIMIT"
echo "TMPDIR=$TMPDIR"
echo "=== END ENV CHECK ==="

mkdir -p "$OUT_DIR"

python main_experiments/minicpm_v46/streambench_v03/judge_streambench_v03_llama3.py \
  --input "$INPUT_DIR/streambench_v0_3_recent6_official_input.json" \
  --output "$OUT_DIR/streambench_v0_3_recent6_llama3_judged.jsonl" \
  --judge-model "$JUDGE_MODEL" \
  --device "$JUDGE_DEVICE" \
  --dtype "$JUDGE_DTYPE" \
  --limit "$JUDGE_LIMIT"

python main_experiments/minicpm_v46/streambench_v03/judge_streambench_v03_llama3.py \
  --input "$INPUT_DIR/streambench_v0_3_prism_official_input.json" \
  --output "$OUT_DIR/streambench_v0_3_prism_llama3_judged.jsonl" \
  --judge-model "$JUDGE_MODEL" \
  --device "$JUDGE_DEVICE" \
  --dtype "$JUDGE_DTYPE" \
  --limit "$JUDGE_LIMIT"

python main_experiments/minicpm_v46/streambench_v03/summarize_streambench_v03_official.py \
  --recent6 "$OUT_DIR/streambench_v0_3_recent6_llama3_judged.jsonl" \
  --prism "$OUT_DIR/streambench_v0_3_prism_llama3_judged.jsonl" \
  --out "$OUT_DIR/streambench_v0_3_official_summary.json"
