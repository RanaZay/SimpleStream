# Progressive Arbitration, frozen configuration v1

New mode:
`progressive_sufficiency_memory_clip_mmr_progressive_arbitration_exact_recent`

Branch: `prism-progressive-arbitration`. Old modes are not overwritten.
Core entry: `lib/minicpm/progressive_sufficiency.py::select_progressive_arbitration`.
Implementation: `lib/minicpm/progressive_arbitration.py`.

## Controller

Let K0 be the exact six frames returned by the existing corrected selector.
Recent images are never resized/reconstructed by the controller. Heterogeneous
recent geometry or fewer than six records raises an explicit error. Historical
images alone are resized to recent geometry when necessary.

Before CLIP/MMR, retain history with `end < recent_start` and
`3 <= recent_start-end <= 30`. Deduplicate exact images/source IDs, then take
the last 64 eligible chunks. One representative frame per chunk is a candidate.
This is pre-ranking filtering, NOT pre-decode filtering: the existing decoder
still materializes the available history first. That cost is reported separately.

The CLIP cache belongs to one QA-model instance and one video path at a time.
Changing video clears it. Keys hash image mode, dimensions and exact pixel bytes.
The cache stores normalized visual embeddings, never option scores. Text vectors
are also bounded. CLIP model initialization occurs outside query timers.

Evaluate K0, then rank eligible candidates with question-plus-option cosine
similarity and MMR (lambda 0.80, pool 12). Candidate-supported options are CLIP
predictions, not MiniCPM predictions. Non-overlap is enforced between candidates.

For each context, `S = .50*M + .20*E + .30*V`, where M is top-two option
probability margin, E is one minus normalized entropy, and V is normalized
maximum visual support. Existing MiniCPM option-logit/fallback scoring is reused.

Admission: `S0 < .62 OR (r1 >= .2995 AND y_h1 != y0)`.
Admission permits a test, not retention.

For each candidate, build chronological accepted history followed by unchanged R6.
Accept iff all conditions hold:

1. `M_j >= .60`.
2. `S_previous - S_j <= .08`.
3. If the new candidate is closer than 10 seconds and has a supported option,
   MiniCPM agrees with that option.
4. At depth 1: answer changes OR `S_j > S0 + .035`.
5. At later depths: `S_j >= S_previous + .035` OR answer changes with
   `M_j >= M_previous`.

Here previous means the last accepted state. Acceptance updates it; rejection
stops expansion without erasing accepted memory. Thus K1/K2 accepted and K3
rejected returns K2. Maximum history is three frames, maximum final context nine.
The final answer is generated afterward using the original prompt and frozen
MiniCPM-V-4.6. No training, label-derived choices, category gates or threshold tuning.

Open-ended/no-option samples explicitly use Exact Recent-6, as approved by the
user. `arbitration_supported=false` and the fallback stop reason distinguish them.
This applies to StreamBench v0.3 and any other prompt without parsed options.
It is NOT evidence that the MCQ controller supports open-ended arbitration.
StreamBench accuracy remains pending its official judge; lexical F1 is not accuracy.

## Timing contract

PRISM algorithmic latency excludes raw video I/O and video/frame decoding.
Timing begins once exact Recent-6 and historical candidate frames with temporal
metadata are available. Full-system latency is reported separately and includes
visual preparation/decode where measurable.

All major GPU timers synchronize through PyTorch, which also supports ROCm.
Two label-free MiniCPM warmup calls (option scoring and generation), plus CLIP
warmup, occur once per QA instance outside both query timers.

Per-query metadata is under `profile.progressive_arbitration` and
`profile.adaptive.timing` (the same values, not additive measurements).

- Visual preparation: `broad_history_decode_ms`, `recent6_decode_ms`,
  `total_visual_preparation_ms`. `video_io_ms=null` because the decoder fuses I/O
  and decoding; zero would falsely claim an independently measured cost.
- Retrieval: prefilter, image-cache lookup, uncached image encoding, text encoding,
  relevance, MMR. Candidate counts, cache hits/misses and reduction ratio are saved.
- K0..K3: option evaluation, visual support, sufficiency arithmetic, state total.
  Option evaluation includes its input preparation; `kN_model_forward_only_ms`
  additionally records the existing scorer's raw model-forward timer.
- Visual-support CLIP work belongs to the corresponding K-state, not retrieval
  again. Its support-cache counters are in the iteration record.
- Unexecuted K stages/arbitrations are null, never zero.
- Actual root-model forward counts and evaluated-state counts are separate:
  fallback scoring can execute more than one forward for a single state.
- Decisions: admission and each arbitration. Context: assembly, memory resizing,
  and final generation's input preparation. Existing decoder/selector materialization
  is visual preparation. Recent hashes/timestamp metadata remain available.
- Final generation: model.generate duration, TTFT, post-first-token duration,
  generated token count. Final wrapper overhead is charged to context, not silently
  dropped. TTFT is part of generation, not an additive extra cost.
- Selection total sums only disjoint child blocks. Algorithmic E2E adds final
  model generation. An independent wall timer validates that sum with tolerance
  `max(50 ms, 2% of measured algorithmic time)`, recording the discrepancy.
  This includes host bookkeeping tolerance, not hidden video decoding.
- Full-system timing includes visual preparation, excluding initialization/warmup.
  Peak allocated/reserved GPU memory and final visual tokens are recorded.

The controller still decodes broad history and the Recent-6 candidate path
separately. No claim of a decoding speedup is made. Future pre-decode pruning
must preserve sampler targets and decoded frame equivalence.

## AMD submission

Use the existing Fahad checkout root. Launchers use faculty/fkqos, one node,
ROCm PyTorch, the repository stream35 environment and per-job `/tmp` MIOpen caches.
Five distributed evaluators default to four GPUs; StreamBench v0.3 is serial and
requests one GPU. No full jobs are submitted by the implementation.

After the branch is published:

```bash
cd /vast/users/fahad.khan/SimpleStream
git fetch origin
git switch prism-progressive-arbitration
git pull --ff-only origin prism-progressive-arbitration
export REPO_ROOT="$PWD"
export CONDA_ENV_PATH="$PWD/.conda/envs/stream35"
mkdir -p logs
unset MAX_SAMPLES
```

Do not discard local edits if git refuses the branch switch. `--check` validates
annotation loading and every referenced video before submitting. The batch job
repeats validation. Direct `sbatch` alone cannot validate before resource allocation.

### OVO-Bench
```bash
bash main_experiments/minicpm_v46/ovo/submit_progressive_arbitration_exact_recent_amd.sh --check &&
sbatch main_experiments/minicpm_v46/ovo/submit_progressive_arbitration_exact_recent_amd.sh
```

### StreamingBench
```bash
bash main_experiments/minicpm_v46/streamingbench/submit_progressive_arbitration_exact_recent_amd.sh --check &&
sbatch main_experiments/minicpm_v46/streamingbench/submit_progressive_arbitration_exact_recent_amd.sh
```

### StreamBench v0.3 (logged open-ended fallback)
```bash
bash main_experiments/minicpm_v46/streambench_v03/submit_progressive_arbitration_exact_recent_amd.sh --check &&
sbatch main_experiments/minicpm_v46/streambench_v03/submit_progressive_arbitration_exact_recent_amd.sh
```

### Video-MME
```bash
bash main_experiments/minicpm_v46/videomme/submit_progressive_arbitration_exact_recent_amd.sh --check &&
sbatch main_experiments/minicpm_v46/videomme/submit_progressive_arbitration_exact_recent_amd.sh
```

### LongVideoBench
```bash
bash main_experiments/minicpm_v46/longvideobench/submit_progressive_arbitration_exact_recent_amd.sh --check &&
sbatch main_experiments/minicpm_v46/longvideobench/submit_progressive_arbitration_exact_recent_amd.sh
```

### EgoSchema-500
```bash
bash main_experiments/minicpm_v46/egoschema/submit_progressive_arbitration_exact_recent_amd.sh --check &&
sbatch main_experiments/minicpm_v46/egoschema/submit_progressive_arbitration_exact_recent_amd.sh
```

## Dataset paths and overrides

These are existing repo conventions, not confirmation that AMD files exist today.
All can be overridden with the named environment variable before `--check`/sbatch.

| Benchmark | Variables and defaults relative to REPO_ROOT |
|---|---|
| OVO | OVO_ANNO_PATH=data/ovo_bench/ovo_bench_new.json; OVO_CHUNKED_DIR=data/ovo_bench/chunked_videos |
| StreamingBench | STREAMINGBENCH_ANNO_PATH=data/streamingbench/questions_real.json; STREAMINGBENCH_VIDEO_DIR=data/streamingbench/videos |
| StreamBench v0.3 | STREAMBENCH_V03_ANNOTATIONS=data/streambench_v0_3/streaming_bench_v0.3.json; STREAMBENCH_V03_DATA_ROOT=data/streambench_v0_3 |
| Video-MME | VIDEOMME_ANNOTATION_PARQUET=data/video_mme/videomme/test-00000-of-00001.parquet; VIDEOMME_VIDEO_DIR=data/video_mme/videos |
| LongVideoBench | LVB_ANNOTATION_JSON=data/longvideobench/lvb_val.json; LVB_DATA_ROOT=data/longvideobench |
| EgoSchema | EGOSCHEMA_ANNOTATION_JSON=reports/egoschema_subset_annotations.json; EGOSCHEMA_VIDEO_DIR=data/egoschema/videos |

EgoSchema preflight requires 500 unique labeled videos. Set its video directory
to the actual extracted folder if nested. No hidden-test labels are fabricated.
Annotation timestamps are not silently clamped; the previously observed truncated
StreamBench videos remain a data-integrity issue requiring an explicit decision.

## Outputs and monitoring

Each run writes `reports/progressive_arbitration/<benchmark>/prism_<jobid>/`.
The six benchmark directory names are `ovo`, `streamingbench`, `streambench_v03`,
`videomme`, `longvideobench`, `egoschema`.
Controls use `recent6_<jobid>` instead. Existing directories are never overwritten.

Each contains run_config.json (frozen values, command, commit, whitelisted env),
run.log, evaluator results, job_timing.json and latency JSON/CSV/Markdown reports.
Per-rank results remain the source of latency statistics. Global throughput uses
subprocess wall time (including startup) and is labeled separately.

```bash
squeue -u fahad.khan
tail -f "$(ls -t logs/pa_ovo-*.out | head -n 1)"
tail -f "$(ls -t logs/pa_streamingbench-*.out | head -n 1)"
tail -f "$(ls -t logs/pa_streambench_v03-*.out | head -n 1)"
tail -f "$(ls -t logs/pa_videomme-*.out | head -n 1)"
tail -f "$(ls -t logs/pa_longvideobench-*.out | head -n 1)"
tail -f "$(ls -t logs/pa_egoschema-*.out | head -n 1)"
find reports/progressive_arbitration -maxdepth 4 -type f
```

## Matched baseline and analysis

Submit the same script with `PRISM_BASELINE=1` exported. This invokes the isolated
Exact Recent-6 control and bypasses sufficiency/retrieval. The same six-frame
selector, prompt, generation settings and data paths are retained. Old unmatched
baseline logs are not accepted as proof of exact-frame equivalence.

```bash
PRISM_BASELINE=1 sbatch --export=ALL main_experiments/minicpm_v46/ovo/submit_progressive_arbitration_exact_recent_amd.sh
```

After runs finish, pass actual output directories (the IDs below are shell
variables set to the job IDs returned by sbatch):

```bash
./.conda/envs/stream35/bin/python main_experiments/tools/summarize_progressive_arbitration.py \
  --run "ovo=reports/progressive_arbitration/ovo/prism_$PRISM_JOB" \
  --baseline "ovo=reports/progressive_arbitration/ovo/recent6_$BASELINE_JOB"
```

Repeat `--run benchmark=directory` and `--baseline benchmark=directory` for all
six benchmarks to produce latency_all_benchmarks.json/.csv/.md under
reports/progressive_arbitration/latency_summary. Pairing verifies question keys
and exact recent hashes. Unverified baselines are rejected rather than compared.
OVO accuracy retains its task/group macro averaging; errors stay in denominators.
StreamBench official accuracy is null until official judging is integrated.
No missing-run scores or baseline gains are invented.

Reports include selected and evaluated depth bins, conditional/amortized block
costs, percentiles, mean actual scoring forwards, cache rate, rescue/damage
with a paired control, task/duration groups and the two largest measured blocks.

## Verification limits

The deterministic integrated fixtures cover 24 paths across selected depths 0..3,
rejection preserving previous memory, stable-answer evidence gain, temporal
consistency and null unused timings. These are correctness tests, not ablations.

The real local 20-question StreamBench smoke completed with zero errors, all six
recent hashes preserved, six final frames and all accounting assertions passing.
All 20 used the approved no-options fallback. Therefore real MCQ K1/K2/K3
execution and AMD runtime have NOT yet been verified. Do not describe this as a
full real-data arbitration validation. No thresholds were tuned.

Initial smoke setup attempts found broken local StreamingBench video symlinks and
a CLIP cache path mismatch; neither produced valid model evaluation results.
The real smoke used the available StreamBench videos and an explicit cached CLIP
checkpoint. Raw setup failures are retained locally, not reported as accuracy.

An AMD MCQ smoke can use `MAX_SAMPLES=20` with one of the MCQ launchers before a
full job. This is a validation requirement, not a component ablation sequence.
