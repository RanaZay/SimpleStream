#!/usr/bin/env bash
# Summarise the novelty comparison batch of 2026-11-11 (StreamScout, WRWS, signal swap, scan stopping, M ablation)
# against Recent-6 and PRISM-VoI. CPU only; reads result dirs and writes everything to reports/novelty_results/.
# Usage (from the repo root): bash main_experiments/tools/novelty_results.sh
set -u
OUT=reports/novelty_results
mkdir -p "$OUT"
PY=${PYTHON:-python}

# First result dir whose name ends in the job id (inference dirs; judging dirs for StreamBench).
run() { ls -d reports/progressive_arbitration/"$1"/*_"$2" 2>/dev/null | head -1; }
judged() { ls -d reports/streambench_judging/*_"$1" 2>/dev/null | head -1; }

tables() {  # benchmark, then label=job pairs
    local bench=$1; shift
    local dir=$bench; [ "$bench" = streambench ] && dir=streambench_v03
    local args=()
    for pair in "$@"; do
        local label=${pair%%=*} job=${pair#*=} d
        d=$(run "$dir" "$job"); [ -n "$d" ] && args+=(--run "$label=$d") || echo "missing run $label ($job)"
        if [ "$bench" = streambench ]; then
            d=$(judged "$job"); [ -n "$d" ] && args+=(--judged "$label=$d") || echo "missing judging $label ($job)"
        fi
    done
    $PY main_experiments/tools/benchmark_tables.py --benchmark "$bench" "${args[@]}" --output "$OUT/tables_$bench"
}

{
echo "# Run sizes (result rows per run)"
for j in 289855 349369 346000 396216 396217 396218 396402 289856 349368 346001 396219 396220 396221 396403 \
         193879 349370 342441 396222 396267 396268 396404; do
    d=$(ls -d reports/progressive_arbitration/*/*_"$j" 2>/dev/null | head -1)
    n=$([ -n "$d" ] && $PY -c 'import sys; from pathlib import Path; sys.path.insert(0, "."); from main_experiments.tools.summarize_progressive_arbitration import records; print(len(records(Path(sys.argv[1]))))' "$d" 2>/dev/null)
    echo "$j ${d:-MISSING} ${n:-?}"
done

echo; echo "# 1. Main comparison"
tables ovo "Recent-6=289855" "PRISM-VoI λ=0=349369" "PRISM-VoI λ=0.1=346000" "WRWS=396217" "StreamScout=396402" \
           "Scan stop=396218"
tables streamingbench "Recent-6=289856" "PRISM-VoI λ=0=349368" "PRISM-VoI λ=0.1=346001" "WRWS=396220" \
           "StreamScout=396403" "Scan stop=396221"
tables streambench "Recent-6=193879" "PRISM-VoI λ=0 (M=5)=349370" "PRISM-VoI λ=0.1=342441" "WRWS=396222" \
           "StreamScout=396404" "M=2=396267" "M=3=396268"

for pair in ovo=396216 streamingbench=396219; do
    bench=${pair%%=*}; d=$(run "$bench" "${pair#*=}")
    [ -z "$d" ] && { echo "missing trace $pair"; continue; }
    echo; echo "# 2. Decision-signal swap, $bench"; $PY main_experiments/tools/signal_swap.py --benchmark "$bench" --run "$d"
    echo; echo "# 3. Lower semantic entropy vs correctness, $bench"; $PY main_experiments/tools/voi_correctness.py --benchmark "$bench" --run "$d"
    echo; echo "# 4. Lambda sweep and always-add-k, $bench"; $PY main_experiments/tools/lambda_sweep.py --benchmark "$bench" --run "$d"
done
} 2>&1 | tee "$OUT/summary.md"
