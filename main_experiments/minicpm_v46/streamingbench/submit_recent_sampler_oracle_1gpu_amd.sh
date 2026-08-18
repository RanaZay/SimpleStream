#!/bin/bash
#SBATCH --job-name=sb100_recent_sampler_oracle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --qos=skqos
#SBATCH --partition=faculty
#SBATCH --output=/vast/users/salman.khan/SimpleStream/logs/%x-%j.out

set -euo pipefail
source ~/.bashrc
conda activate stream35

export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export ROCM_HOME=${ROCM_HOME:-/opt/rocm}
export PATH="${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MIOPEN_DISABLE_CACHE=1
export PYTORCH_TUNABLEOP_ENABLED=0
export MINICPM_SEED=${MINICPM_SEED:-42}
export PYTHONHASHSEED=${MINICPM_SEED}

REPO_ROOT=/vast/users/salman.khan/SimpleStream
cd "$REPO_ROOT"

mkdir -p logs .cache/miopen .cache/torch_kernels
export MIOPEN_USER_DB_PATH="$REPO_ROOT/.cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$REPO_ROOT/.cache/miopen"
export PYTORCH_KERNEL_CACHE_PATH="$REPO_ROOT/.cache/torch_kernels"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

SOURCE_RESULTS=${SOURCE_RESULTS:-main_experiments/results/repro_adaptive/streamingbench_minicpmv46_psm_microclip_temporal_microclip_recent6_h64_p12_ctx70_offsetsm1_0_1_limit100_d8}
SB_ANNOTATIONS=${SB_ANNOTATIONS:-data/streamingbench/questions_real.json}
SB_VIDEO_DIR=${SB_VIDEO_DIR:-data/streamingbench/videos}
OUT_DIR=${OUT_DIR:-reports/recent_sampler_diag/streamingbench_100}
QA_DEVICE=${QA_DEVICE:-cuda:0}
CURRENT_FPS=${CURRENT_FPS:-1.0}
CANDIDATE_FPS=${CANDIDATE_FPS:-4.0}

ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$OUT_DIR" "${OUT_DIR}.old_$ts" 2>/dev/null || true
fi

echo "=== ENV CHECK ==="
which python
python -V
python -c "import torch; print('torch=', torch.__version__); print('hip=', torch.version.hip); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "SOURCE_RESULTS=$SOURCE_RESULTS"
echo "SB_ANNOTATIONS=$SB_ANNOTATIONS"
echo "SB_VIDEO_DIR=$SB_VIDEO_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "QA_DEVICE=$QA_DEVICE"
echo "CURRENT_FPS=$CURRENT_FPS"
echo "CANDIDATE_FPS=$CANDIDATE_FPS"
echo "=== END ENV CHECK ==="

python main_experiments/tools/run_streamingbench_recent_sampler_oracle.py \
  --source-results "$SOURCE_RESULTS" \
  --annotations "$SB_ANNOTATIONS" \
  --video-dir "$SB_VIDEO_DIR" \
  --out-dir "$OUT_DIR" \
  --qa-device "$QA_DEVICE" \
  --current-fps "$CURRENT_FPS" \
  --candidate-fps "$CANDIDATE_FPS"
