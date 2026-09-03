#!/usr/bin/env bash
set -euo pipefail

cd /tmp/mobilestream_repro/SimpleStream

PYTHON_BIN="${PYTHON_BIN:-/home/ashaker/miniconda3/envs/llava-ov-4b-clean/bin/python}"
CLIP_SNAPSHOT="${CLIP_SNAPSHOT:-/home/ashaker/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268}"
RUN_ROOT="${RUN_ROOT:-reports/prism_event_ledger/local_streamingbench_count_ledger_override_multigpu_v1}"
MAX_GPUS="${MAX_GPUS:-4}"
MIN_FREE_MEMORY_MB="${MIN_FREE_MEMORY_MB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

ANNOTATIONS="${ANNOTATIONS:-data/streamingbench/questions_real.json}"
VIDEO_DIR="${VIDEO_DIR:-data/streamingbench/videos}"
RECENT="${RECENT:-reports/local_hf_full_realtime_2500_fixed_options_6gpu/no_memory_exact_recent/streaming_bench_minicpmv46_results_20260901_151158.json}"
PRISM="${PRISM:-reports/prism_retrieval_variants/local_streamingbench_full_prism_clip_mmr_evidence_arbitration_temporal_consistency_g0p2995_m0p60_d0p08_t3-30_c10_6gpu/streaming_bench_minicpmv46_results_20260903_assembled_from_ranks.json}"

mkdir -p "${RUN_ROOT}"

echo "RUN_ROOT=${RUN_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CLIP_SNAPSHOT=${CLIP_SNAPSHOT}"
echo "Waiting for free GPUs..."

while true; do
  mapfile -t FREE_GPUS < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      awk -F, -v limit="${MIN_FREE_MEMORY_MB}" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 < limit) print $1}'
  )
  if [[ "${#FREE_GPUS[@]}" -gt 0 ]]; then
    break
  fi
  date '+%F %T no free GPU yet'
  sleep "${POLL_SECONDS}"
done

NUM_SHARDS="${#FREE_GPUS[@]}"
if [[ "${NUM_SHARDS}" -gt "${MAX_GPUS}" ]]; then
  NUM_SHARDS="${MAX_GPUS}"
fi

echo "Using ${NUM_SHARDS} GPU shard(s): ${FREE_GPUS[*]:0:${NUM_SHARDS}}"

PIDS=()
for ((SHARD=0; SHARD<NUM_SHARDS; SHARD++)); do
  GPU="${FREE_GPUS[$SHARD]}"
  SHARD_DIR="${RUN_ROOT}/shard_${SHARD}"
  mkdir -p "${SHARD_DIR}"
  echo "Launching shard ${SHARD}/${NUM_SHARDS} on physical GPU ${GPU}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    export HF_HOME="${HF_HOME:-/home/ashaker/.cache/huggingface}"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_TOKEN="${HF_TOKEN:-dummy}"
    export PYTHONNOUSERSITE=1
    "${PYTHON_BIN}" main_experiments/tools/analyze_streamingbench_count_ledger_override.py \
      --annotations "${ANNOTATIONS}" \
      --video-dir "${VIDEO_DIR}" \
      --recent "${RECENT}" \
      --prism "${PRISM}" \
      --clip-model "${CLIP_SNAPSHOT}" \
      --clip-device cuda:0 \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${SHARD}" \
      --out-dir "${SHARD_DIR}"
  ) > "${SHARD_DIR}/run.log" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "At least one shard failed. Logs:"
  find "${RUN_ROOT}" -path '*/run.log' -print
  exit 1
fi

export PYTHONNOUSERSITE=1
"${PYTHON_BIN}" main_experiments/tools/summarize_streamingbench_count_ledger_shards.py \
  --shard-root "${RUN_ROOT}" \
  --out-dir "${RUN_ROOT}/merged" | tee "${RUN_ROOT}/merge.log"

echo "Done: ${RUN_ROOT}/merged"
