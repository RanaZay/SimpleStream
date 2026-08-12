# OVO Oracle vs Official Evaluator Reconciliation

This note documents the scoring protocol that must be used for publication-facing OVO numbers.

## Official OVO evaluator

Code path:

- `main_experiments/minicpm_v46/ovo/eval_baseline.py` runs MiniCPM and writes per-rank checkpoint JSONL.
- `main_experiments/qwen/evals/eval_qwen3vl_ovo.py::merge_shard_results` merges those checkpoints.
- `scoring/score_ovo_bench.py` contains the standalone official OVO scoring implementation.
- `lib/shared/recent_window.py::calculate_ovo_scores` mirrors the same scoring and aggregation used by the runner.

Record selection:

- Each raw checkpoint row has `_key = f"{task}:{id}"`.
- During official merge, the first row for a `_key` is kept.
- Later duplicate/incremental rows with the same `_key` are skipped.
- The final official JSON is grouped as `backward`, `realtime`, and `forward`.
- Forward tasks keep one parent record per annotation, with per-question entries in `test_info`.

Prediction extraction and scoring:

- Backward and real-time tasks: correct if `ground_truth` appears as a substring in `response`.
- `REC`: extract all digits from `response`, join them, and compare to `test_info.count`.
- `SSR` and `CRR`: `type=0` means No, `type=1` means Yes. Exact `N`/`Y` or response containing `No`/`Yes` is accepted.

Aggregation:

- Compute per-task accuracy.
- Average task accuracies inside each category:
  - Backward Avg
  - Real-Time Avg
  - Forward Avg
- Final Total Avg is the macro average of those three category averages.

## Old oracle mismatch

The old oracle summary reported:

- Recent-6 = `1563 / 3035 = 51.50`
- Oracle K=0..3 = `56.44`

That was not the official OVO protocol. It was a flat micro average over the oracle rows. The result directory can contain 3035 flattened/incremental records but only 1640 unique primary keys, with 1395 duplicate/incremental records. Official OVO scoring does not average those duplicate flat rows directly.

The corrected oracle tool now:

- Loads the final official grouped JSON if present.
- Otherwise reconstructs the official first-seen `_key = task:id` checkpoint merge.
- Scores K=0, fixed K=1, fixed K=2, fixed K=3, Oracle-K, and PRISM with the official task/category macro aggregation.
- Writes `official_protocol_reconciliation` into `oracle_summary.json`.
- Supports `--rescore-existing` to rescore existing oracle branch predictions without rerunning MiniCPM.

Publication rule:

- Report only the `official_protocol_reconciliation.official_ovo.*.total_avg` values.
- Do not mix the legacy flat `oracle_overall.recent6_accuracy` with official OVO benchmark scores.
