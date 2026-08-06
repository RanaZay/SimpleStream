#!/bin/bash
#SBATCH --job-name=minicpmv46_sb_hybridclip_d8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --qos=skqos
#SBATCH --partition=faculty
#SBATCH --output=/vast/users/salman.khan/SimpleStream/logs/%x-%j.out

source ~/.bashrc
conda activate stream35

export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export RECENT_CLIP_FFMPEG=${RECENT_CLIP_FFMPEG:-/usr/bin/ffmpeg}
export ROCM_HOME=${ROCM_HOME:-/opt/rocm}
export PATH="${ROCM_HOME}/bin:/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_HOME}/lib64:${LD_LIBRARY_PATH}"
export MIOPEN_DISABLE_CACHE=1
export PYTORCH_TUNABLEOP_ENABLED=0
export MINICPM_SEED=${MINICPM_SEED:-42}
export PYTHONHASHSEED=${MINICPM_SEED}

REPO_ROOT=/vast/users/salman.khan/SimpleStream
cd "$REPO_ROOT" || exit 1
mkdir -p logs .cache/miopen .cache/torch_kernels
export MIOPEN_USER_DB_PATH="$REPO_ROOT/.cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$REPO_ROOT/.cache/miopen"
export PYTORCH_KERNEL_CACHE_PATH="$REPO_ROOT/.cache/torch_kernels"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export ATTN_IMPLEMENTATION=sdpa
export MINICPM_DOWNSAMPLE_MODE=16x
export MINICPM_MAX_SLICE_NUMS=1
export MINICPM_PROFILE_COMPONENTS=${MINICPM_PROFILE_COMPONENTS:-1}
export MINICPM_SERIALIZE_MODEL_LOAD=${MINICPM_SERIALIZE_MODEL_LOAD:-1}
export MINICPM_MODEL_LOAD_TIMEOUT=${MINICPM_MODEL_LOAD_TIMEOUT:-7200}
export HF_ENABLE_PARALLEL_LOADING=false
export HF_PARALLEL_LOADING_WORKERS=1
export HF_DEACTIVATE_ASYNC_LOAD=1
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-65536}

export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29911}
export RECENT_FRAMES_ONLY=6
export RECENT_CLIP_SECONDS=2

echo "[PREFLIGHT] RECENT_CLIP_FFMPEG=${RECENT_CLIP_FFMPEG}"
echo "[PREFLIGHT] PATH=${PATH}"
"${RECENT_CLIP_FFMPEG}" -version | head -1
python - <<'PY'
from pathlib import Path
from lib.minicpm.recent_clip import cut_recent_clip, resolve_ffmpeg_binary

video_dir = Path("data/streamingbench/videos")
video = next(video_dir.glob("*.mp4"))
out = Path("/tmp/minicpm_hybrid_recent_clip_preflight.mp4")
start, end = cut_recent_clip(
    video_path=video,
    clip_path=out,
    end_time_seconds=10.0,
    clip_seconds=2.0,
)
size = out.stat().st_size if out.exists() else 0
print(f"[PREFLIGHT] ffmpeg={resolve_ffmpeg_binary()}")
print(f"[PREFLIGHT] video={video}")
print(f"[PREFLIGHT] clip={out} start={start:.3f} end={end:.3f} size={size}")
if size <= 0:
    raise SystemExit("[PREFLIGHT] recent clip extraction produced an empty file")
PY

RESULT_DIR="$REPO_ROOT/main_experiments/results/repro_hybrid_clip/streamingbench_minicpmv46_hybrid_recent6_clip2_d8"
ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
    mv "$RESULT_DIR" "${RESULT_DIR}.old_$ts" 2>/dev/null || true
fi
export SB_RESULT_DIR="$RESULT_DIR"

bash main_experiments/minicpm_v46/streamingbench/run_hybrid_recent_clip.sh
