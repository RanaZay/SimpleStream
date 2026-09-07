#!/bin/bash
#SBATCH --job-name=lvb_recent6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --qos=fkqos
#SBATCH --partition=faculty
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/envs/stream35/bin/python}"

export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export ROCM_HOME=${ROCM_HOME:-/opt/rocm}
export PATH="${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [[ -n "${PRISM_TMPDIR:-}" ]]; then
    export TMPDIR="$PRISM_TMPDIR"
else
    export TMPDIR="/tmp/${USER}/lvb_recent6_${SLURM_JOB_ID:-$$}"
fi

export MIOPEN_DISABLE_CACHE=${MIOPEN_DISABLE_CACHE:-0}
export PYTORCH_TUNABLEOP_ENABLED=0
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
export MINICPM_DOWNSAMPLE_MODE=${MINICPM_DOWNSAMPLE_MODE:-16x}
export MINICPM_MAX_SLICE_NUMS=${MINICPM_MAX_SLICE_NUMS:-1}
export MINICPM_PROFILE_COMPONENTS=${MINICPM_PROFILE_COMPONENTS:-1}
export MINICPM_SERIALIZE_MODEL_LOAD=${MINICPM_SERIALIZE_MODEL_LOAD:-1}
export MINICPM_MODEL_LOAD_TIMEOUT=${MINICPM_MODEL_LOAD_TIMEOUT:-7200}
export HF_ENABLE_PARALLEL_LOADING=${HF_ENABLE_PARALLEL_LOADING:-false}
export HF_PARALLEL_LOADING_WORKERS=${HF_PARALLEL_LOADING_WORKERS:-1}
export HF_DEACTIVATE_ASYNC_LOAD=${HF_DEACTIVATE_ASYNC_LOAD:-1}
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-65536}
export MINICPM_SEED=${MINICPM_SEED:-42}
export PYTHONHASHSEED=${MINICPM_SEED}
export HF_HOME=${HF_HOME:-$REPO_ROOT/.hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}

mkdir -p logs "$TMPDIR" "$TMPDIR/miopen" "$TMPDIR/miopen-lockfiles" "$TMPDIR/torch_kernels"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$TMPDIR/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-$TMPDIR/miopen}"
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-$TMPDIR/torch_kernels}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}
export NUM_PROCESSES=${NUM_PROCESSES:-2}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-$((29600 + (${SLURM_JOB_ID:-0} % 1000)))}

SAMPLE_TAG="val"
if [[ -n "${LVB_MAX_SAMPLES:-${MAX_SAMPLES:-}}" ]]; then
    SAMPLE_TAG="n${LVB_MAX_SAMPLES:-${MAX_SAMPLES}}"
fi
RESULT_DIR="${LVB_RESULT_DIR:-$REPO_ROOT/main_experiments/results/repro_longvideobench/longvideobench_minicpmv46_recent6_${SAMPLE_TAG}_d8}"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi

echo "=== ENV CHECK ==="
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -V
"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__); print('hip=', torch.version.hip); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
echo "LVB_RESULT_DIR=$RESULT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "TMPDIR=$TMPDIR"
echo "MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH"
echo "=== END ENV CHECK ==="

COMMON_ARGS=(
    --data-root "${LVB_DATA_ROOT:-$REPO_ROOT/data/longvideobench}"
    --annotation-json "${LVB_ANNOTATION_JSON:-$REPO_ROOT/data/longvideobench/lvb_val.json}"
    --output-dir "$RESULT_DIR"
    --mode recent6
    --qa-model "${MINICPM_QA_MODEL:-openbmb/MiniCPM-V-4.6}"
    --recent-frames-only 6
    --chunk-duration 1.0
    --fps 1.0
    --max-qa-tokens 256
    --max-subtitle-chars "${LVB_MAX_SUBTITLE_CHARS:-0}"
)
if [[ -n "${LVB_MAX_SAMPLES:-${MAX_SAMPLES:-}}" ]]; then
    COMMON_ARGS+=(--max-samples "${LVB_MAX_SAMPLES:-${MAX_SAMPLES}}")
fi

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --multi_gpu \
    --mixed_precision bf16 \
    main_experiments/minicpm_v46/longvideobench/eval_longvideobench_dist.py "${COMMON_ARGS[@]}"
