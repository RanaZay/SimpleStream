#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

START_PORT="${START_PORT:-30100}"
RUN_GATE_E="${RUN_GATE_E:-0}"
RUN_OVO="${RUN_OVO:-1}"
RUN_STREAMINGBENCH="${RUN_STREAMINGBENCH:-1}"

CONFIGS=(
  "A 0.60 0.70 0.54 10"
  "B 0.60 0.72 0.54 10"
  "C 0.60 0.74 0.52 10"
  "D 0.60 0.74 0.54 10"
)

if [[ "${RUN_GATE_E}" == "1" ]]; then
  CONFIGS+=("E 0.62 0.72 0.54 10")
fi

if [[ "${RUN_STREAMINGBENCH}" == "1" && -z "${SB_ANNO_PATH:-}" ]]; then
  echo "[ERROR] RUN_STREAMINGBENCH=1 requires SB_ANNO_PATH for the fixed diagnostic subset." >&2
  echo "Build it first with main_experiments/tools/build_streamingbench_common_result_subset.py." >&2
  exit 2
fi

port="${START_PORT}"
for spec in "${CONFIGS[@]}"; do
  read -r name low high cand td <<<"${spec}"
  echo "[INFO] Submitting config ${name}: low=${low} high=${high} cand=${cand} td=${td}"
  if [[ "${RUN_OVO}" == "1" ]]; then
    MINICPM_PSM_GATE_TAU_LOW="${low}" \
    MINICPM_PSM_GATE_TAU_HIGH="${high}" \
    MINICPM_PSM_GATE_CANDIDATE_THRESHOLD="${cand}" \
    MINICPM_PSM_GATE_TEMPORAL_DISTANCE_THRESHOLD="${td}" \
    MAIN_PROCESS_PORT="${port}" \
    sbatch main_experiments/minicpm_v46/ovo/submit_progressive_sufficiency_memory_conservative_gate_8gpu_amd.sh
    port=$((port + 1))
  fi
  if [[ "${RUN_STREAMINGBENCH}" == "1" ]]; then
    SB_ANNO_PATH="${SB_ANNO_PATH}" \
    MINICPM_PSM_GATE_TAU_LOW="${low}" \
    MINICPM_PSM_GATE_TAU_HIGH="${high}" \
    MINICPM_PSM_GATE_CANDIDATE_THRESHOLD="${cand}" \
    MINICPM_PSM_GATE_TEMPORAL_DISTANCE_THRESHOLD="${td}" \
    MAIN_PROCESS_PORT="${port}" \
    sbatch main_experiments/minicpm_v46/streamingbench/submit_progressive_sufficiency_memory_conservative_gate_8gpu_amd.sh
    port=$((port + 1))
  fi
done
