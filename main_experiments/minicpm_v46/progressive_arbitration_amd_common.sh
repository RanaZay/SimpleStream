#!/bin/bash
# Sourced by the six isolated launchers. No existing launcher is modified.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH:-$REPO_ROOT/.conda/envs/stream35}/bin/python}"
export PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
export PATH="$ROCM_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_HOME/lib:$ROCM_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MINICPM_DOWNSAMPLE_MODE=16x MINICPM_MAX_SLICE_NUMS=1 ATTN_IMPLEMENTATION=sdpa
export MINICPM_PROFILE_COMPONENTS=1 MINICPM_SERIALIZE_MODEL_LOAD=1
export HF_ENABLE_PARALLEL_LOADING=false HF_PARALLEL_LOADING_WORKERS=1 HF_DEACTIVATE_ASYNC_LOAD=1
export MINICPM_SEED=42 PYTHONHASHSEED=42 PYTORCH_TUNABLEOP_ENABLED=0
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_home}"
export TMPDIR="/tmp/$USER/progressive_arbitration_${SLURM_JOB_ID:-check}_$BENCHMARK"
export MIOPEN_USER_DB_PATH="$TMPDIR/miopen" MIOPEN_CUSTOM_CACHE_DIR="$TMPDIR/miopen"
export PYTORCH_KERNEL_CACHE_PATH="$TMPDIR/torch_kernels"
export NUM_PROCESSES="${NUM_PROCESSES:-$DEFAULT_PROCESSES}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
mkdir -p logs "$TMPDIR/miopen" "$TMPDIR/torch_kernels"
if [[ "${1:-}" == "--check" ]]; then
    exec "$PYTHON_BIN" main_experiments/tools/run_progressive_arbitration.py "$BENCHMARK" --check
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "First run: bash $0 --check; then: sbatch $0" >&2
    exit 2
fi
"$PYTHON_BIN" -c 'import os, torch; print("torch",torch.__version__,"hip",torch.version.hip,"gpus",torch.cuda.device_count()); assert torch.version.hip, "Expected ROCm PyTorch on AMD"; assert torch.cuda.device_count() >= int(os.environ["NUM_PROCESSES"]), "Fewer visible GPUs than workers; check CUDA/HIP_VISIBLE_DEVICES"'
exec "$PYTHON_BIN" main_experiments/tools/run_progressive_arbitration.py "$BENCHMARK"
