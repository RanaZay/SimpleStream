#!/usr/bin/env bash
set -euo pipefail

cd /tmp/mobilestream_repro/SimpleStream

PYTHON_BIN="${PYTHON_BIN:-/home/ashaker/miniconda3/envs/llava-ov-4b-clean/bin/python}"
RUN_ROOT="${RUN_ROOT:-reports/prism_retrieval_variants/local_ovo_full_prism_clip_mmr_evidence_contract_g0p2995_m0p60_d0p08_t3-30_c10_maxgpu}"
MAX_GPUS="${MAX_GPUS:-8}"
MIN_GPUS="${MIN_GPUS:-4}"
MIN_FREE_MEMORY_MB="${MIN_FREE_MEMORY_MB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$(dirname "${RUN_ROOT}")" logs

echo "RUN_ROOT=${RUN_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "Waiting for at least ${MIN_GPUS} free GPU(s), using up to ${MAX_GPUS}..."

while true; do
  mapfile -t FREE_GPUS < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      awk -F, -v limit="${MIN_FREE_MEMORY_MB}" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 < limit) print $1}'
  )
  if [[ "${#FREE_GPUS[@]}" -ge "${MIN_GPUS}" ]]; then
    break
  fi
  date '+%F %T waiting; free GPUs: '"${FREE_GPUS[*]:-none}"
  sleep "${POLL_SECONDS}"
done

NUM_PROCESSES="${#FREE_GPUS[@]}"
if [[ "${NUM_PROCESSES}" -gt "${MAX_GPUS}" ]]; then
  NUM_PROCESSES="${MAX_GPUS}"
fi
GPU_LIST="$(IFS=,; echo "${FREE_GPUS[*]:0:${NUM_PROCESSES}}")"

ts=$(date +%Y%m%d_%H%M%S)
if [[ "${RESUME:-0}" != "1" ]]; then
  mv "${RUN_ROOT}" "${RUN_ROOT}.old_${ts}" 2>/dev/null || true
fi
mkdir -p "${RUN_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export MINICPM_DOWNSAMPLE_MODE="${MINICPM_DOWNSAMPLE_MODE:-16x}"
export MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS:-1}"
export MINICPM_PROFILE_COMPONENTS="${MINICPM_PROFILE_COMPONENTS:-1}"
export MINICPM_SERIALIZE_MODEL_LOAD="${MINICPM_SERIALIZE_MODEL_LOAD:-1}"
export MINICPM_MODEL_LOAD_TIMEOUT="${MINICPM_MODEL_LOAD_TIMEOUT:-7200}"
export HF_ENABLE_PARALLEL_LOADING=false
export HF_PARALLEL_LOADING_WORKERS=1
export HF_DEACTIVATE_ASYNC_LOAD=1
export MINICPM_SEED="${MINICPM_SEED:-42}"
export PYTHONHASHSEED="${MINICPM_SEED}"
export HF_HOME="${HF_HOME:-/home/ashaker/.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:-dummy}"

export ADAPTIVE_MODE=progressive_sufficiency_memory_clip_mmr_evidence_contract
export PRISM_CLIP_MODE=evidence_contract
export ADAPTIVE_MIN_WINDOW=6
export ADAPTIVE_MID_WINDOW=6
export ADAPTIVE_MAX_WINDOW=6
export ADAPTIVE_MEMORY_ANCHORS=3
export ADAPTIVE_MEMORY_SEARCH_CHUNKS=64
export MINICPM_EXACT_RECENT_CANDIDATE_FPS="${MINICPM_EXACT_RECENT_CANDIDATE_FPS:-4.0}"
export MINICPM_PSM_HISTORY_SEARCH_CHUNKS="${MINICPM_PSM_HISTORY_SEARCH_CHUNKS:-64}"
export MINICPM_PSM_HISTORY_CANDIDATE_POOL="${MINICPM_PSM_HISTORY_CANDIDATE_POOL:-12}"
export MINICPM_PSM_MAX_MEMORY_FRAMES="${MINICPM_PSM_MAX_MEMORY_FRAMES:-3}"
export MINICPM_PSM_MIN_TEMPORAL_GAP="${MINICPM_PSM_MIN_TEMPORAL_GAP:-2}"
export MINICPM_PSM_SUFFICIENCY_THRESHOLD="${MINICPM_PSM_SUFFICIENCY_THRESHOLD:-0.62}"
export MINICPM_PSM_MIN_EVIDENCE_GAIN="${MINICPM_PSM_MIN_EVIDENCE_GAIN:-0.035}"
export MINICPM_PSM_NEGATIVE_GAIN_TOLERANCE="${MINICPM_PSM_NEGATIVE_GAIN_TOLERANCE:-0.02}"
export MINICPM_PSM_MARGIN_WEIGHT=0.50
export MINICPM_PSM_ENTROPY_WEIGHT=0.20
export MINICPM_PSM_VISUAL_SUPPORT_WEIGHT=0.30
export MINICPM_PSM_MMR_LAMBDA="${MINICPM_PSM_MMR_LAMBDA:-0.80}"
export MINICPM_PSM_CLIP_OVERRIDE_THRESHOLD="${MINICPM_PSM_CLIP_OVERRIDE_THRESHOLD:-0.2995}"
export MINICPM_PSM_ARBITRATION_MIN_MARGIN="${MINICPM_PSM_ARBITRATION_MIN_MARGIN:-0.60}"
export MINICPM_PSM_ARBITRATION_MAX_SUFFICIENCY_DROP="${MINICPM_PSM_ARBITRATION_MAX_SUFFICIENCY_DROP:-0.08}"
export MINICPM_PSM_TEMPORAL_BAND_MIN_SECONDS="${MINICPM_PSM_TEMPORAL_BAND_MIN_SECONDS:-3}"
export MINICPM_PSM_TEMPORAL_BAND_MAX_SECONDS="${MINICPM_PSM_TEMPORAL_BAND_MAX_SECONDS:-30}"
export MINICPM_PSM_CANDIDATE_K1_DISAGREE_MAX_DISTANCE_SECONDS="${MINICPM_PSM_CANDIDATE_K1_DISAGREE_MAX_DISTANCE_SECONDS:-10}"
export MINICPM_PSM_EXACT_RECENT_PRESERVE_SOURCE_IDS=0
export MINICPM_PSM_ASSERT_TEMPORAL_ALIGNMENT=1
export MINICPM_PSM_PRINT_TRACE="${MINICPM_PSM_PRINT_TRACE:-0}"
export OVO_RESULT_DIR="${RUN_ROOT}"

COMMON_ARGS=(
  --model_path "openbmb/MiniCPM-V-4.6"
  --anno_path "${OVO_ANNO_PATH:-/tmp/mobilestream_repro/SimpleStream/data/ovo_bench/ovo_bench_new.json}"
  --chunked_dir "${OVO_CHUNKED_DIR:-/tmp/mobilestream_repro/SimpleStream/data/ovo_bench/chunked_videos}"
  --result_dir "${OVO_RESULT_DIR}"
  --recent_frames_only "${ADAPTIVE_MAX_WINDOW}"
  --chunk_duration 1.0
  --fps 1.0
  --max_qa_tokens 256
  --adaptive-mode "${ADAPTIVE_MODE}"
  --adaptive-min-window "${ADAPTIVE_MIN_WINDOW}"
  --adaptive-mid-window "${ADAPTIVE_MID_WINDOW}"
  --adaptive-max-window "${ADAPTIVE_MAX_WINDOW}"
  --adaptive-dedup-threshold 4.0
  --adaptive-dedup-min-frames 4
  --adaptive-memory-anchors "${ADAPTIVE_MEMORY_ANCHORS}"
  --adaptive-memory-search-chunks "${ADAPTIVE_MEMORY_SEARCH_CHUNKS}"
)

if [[ -n "${MAX_SAMPLES_PER_SPLIT:-}" ]]; then
  COMMON_ARGS+=(--max_samples_per_split "${MAX_SAMPLES_PER_SPLIT}")
fi
if [[ -n "${MAX_SAMPLES_TOTAL:-}" ]]; then
  COMMON_ARGS+=(--max_samples_total "${MAX_SAMPLES_TOTAL}")
fi

{
  echo "=== ENV CHECK ==="
  date
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "NUM_PROCESSES=${NUM_PROCESSES}"
  echo "OVO_RESULT_DIR=${OVO_RESULT_DIR}"
  echo "ADAPTIVE_MODE=${ADAPTIVE_MODE}"
  echo "PRISM_CLIP_MODE=${PRISM_CLIP_MODE}"
  "${PYTHON_BIN}" -V
  "${PYTHON_BIN}" -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
  echo "=== END ENV CHECK ==="
} | tee "${RUN_ROOT}/env.log"

"${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29943}" \
  --multi_gpu \
  --mixed_precision bf16 \
  main_experiments/minicpm_v46/ovo/eval_prism_exact_recent.py "${COMMON_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/run.log"
