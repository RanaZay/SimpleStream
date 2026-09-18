# StreamBench semantic judging on AMD

This evaluates saved answers; it does not rerun MiniCPM or read videos. Outputs:
`coverage.json`, `judge_input.json`, `judge/manifest.json`, `judge/judged.json`,
`summary.json`, and `summary.md`. The final summary includes overall and all
present subtask accuracies as percentages, using yes/no semantic judgments.
The separate 0..5 judge rating is not treated as accuracy.

## Prerequisites

- Pull the implementation before using these new commands.
- Run on an existing allocated GPU node, or submit the launcher with sbatch.
- Provide a fixed local **Meta-Llama-3-8B-Instruct** snapshot. Model access may
  require accepting Meta's license and authenticating with Hugging Face. Never
  put access tokens in launch scripts or shared logs.
- The Python environment needs torch, transformers, accelerate, and tqdm. The
  wrapper records their relevant versions. No GPU/backend changes are required
  to saved MiniCPM results.
- Keep the same judge model snapshot, package versions, and protocol for Recent-6
  and PRISM. Inherited checkpoint generation defaults are used by the original
  author script; do not describe this as deterministic just because the wrapper
  is pinned. The upstream script may need compatible Transformers versions.

The author script is not vendored or modified. Obtain a dedicated checkout:

```bash
git clone https://github.com/hmxiong/StreamChat.git /tmp/$USER/streamchat_judge
git -C /tmp/$USER/streamchat_judge checkout --detach f8bcd22ef68cb69f59bcae29ed9fe93f75c077be
```

## Run

Set RESULTS to the completed `streambench_v0_3_smoke_results.json` file, including
all 1838 recorded rows. Its historical filename does not imply a limited run.
Set JUDGE_MODEL to the existing model snapshot path. Use a fresh output directory
for every evaluation; existing directories are deliberately not overwritten.

```bash
cd /vast/users/fahad.khan/SimpleStream
export REPO_ROOT="$PWD"
export CONDA_ENV_PATH="$PWD/.conda/envs/stream35"
RESULTS="/absolute/path/to/streambench_v0_3_smoke_results.json"
JUDGE_MODEL="/absolute/path/to/Meta-Llama-3-8B-Instruct/snapshot"
OUT="reports/streambench_judging/recent6_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
set -o pipefail
bash main_experiments/minicpm_v46/streambench_v03/submit_author_judge_amd.sh \
  --results "$RESULTS" --method prism \
  --source /tmp/$USER/streamchat_judge \
  --judge-model "$JUDGE_MODEL" --output-dir "$OUT" \
  --allow-inference-errors \
  2>&1 | tee "logs/streambench_judge_$(date +%Y%m%d_%H%M%S).out"
```

The method argument selects the saved JSON key. The recent control launched
through the generic PRISM runner is stored under `prism`; its actual mode is
read from metadata and displayed as Recent-6. For older outputs actually saved
under `recent6`, use `--method recent6` instead.

## Eight inference failures

The default is to refuse scoring a run with inference failures. The explicit
flag above permits judging the 1830 valid predictions and computes
`correct / 1838`, counting the eight failures as wrong. It also reports
`correct / 1830` separately as a diagnostic and marks the result as not error-free.
This is our explicit failure-accounting policy, not a claim that the benchmark
authors specify how failed inference must be handled. Repairing those failures
and judging the complete valid set remains preferable for a clean comparison.

Judge output failures are different: malformed judgments, missing output rows,
duplicate IDs, or changed question/answer text cause evaluation to fail. They
are not silently scored as no and never excluded from the denominator.

## CPU preparation check

This does not load LLaMA or require a GPU:

```bash
./.conda/envs/stream35/bin/python -m \
  main_experiments.minicpm_v46.streambench_v03.evaluate_saved_answers \
  --results "$RESULTS" --method prism --prepare-only \
  --output-dir "reports/streambench_judging/preflight_$(date +%Y%m%d_%H%M%S)"
```

All judge computation is offline evaluation overhead and must be excluded from
the Recent-6/PRISM inference latency tables. No new semantic score exists until
the judge actually completes and the coverage audit passes.
