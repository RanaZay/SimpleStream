# Experiment Entry Points

This folder groups experiment code by model and benchmark. Current MiniCPM-V
4.6 experiments live under `minicpm_v46/`; Qwen reproduction baselines live
under `qwen/`.

Top-level `eval_*`, `run_*`, and `submit_*` files have been removed from this
folder. Use the model-specific folders below as the source of truth.

## Current MiniCPM-V 4.6 All-Frame Runs

OVO-Bench:

- `minicpm_v46/ovo/submit_baseline_8gpu_amd.sh`: baseline, all frames at 1 FPS.
- `minicpm_v46/ovo/submit_ctr_g50_8gpu_amd.sh`: MiniCPM + CTR, `G=50`, `tau=0.9`.
- `minicpm_v46/ovo/submit_streamingtom_g50_8gpu_amd.sh`: MiniCPM + CTR + OQM.
- `minicpm_v46/ovo/submit_timechat_8gpu_amd.sh`: MiniCPM + TimeChat-DTD.
  Set `TIMECHAT_RETENTION_RATIO` to `1.0`, `0.8`, or `0.4`.

StreamingBench:

- `minicpm_v46/streamingbench/submit_baseline_8gpu_amd.sh`: baseline, all frames at 1 FPS.
- `minicpm_v46/streamingbench/submit_ctr_g50_8gpu_amd.sh`: MiniCPM + CTR, `G=50`, `tau=0.9`.
- `minicpm_v46/streamingbench/submit_streamingtom_g50_8gpu_amd.sh`: MiniCPM + CTR + OQM.
- `minicpm_v46/streamingbench/submit_timechat_8gpu_amd.sh`: MiniCPM + TimeChat-DTD.
  Set `TIMECHAT_RETENTION_RATIO` to `1.0`, `0.8`, or `0.4`.

## Current Python Evaluators

OVO-Bench:

- `minicpm_v46/ovo/eval_baseline.py`
- `minicpm_v46/ovo/eval_ctr.py`
- `minicpm_v46/ovo/eval_streamingtom.py`
- `minicpm_v46/ovo/eval_timechat.py`

StreamingBench:

- `minicpm_v46/streamingbench/eval_baseline.py`
- `minicpm_v46/streamingbench/eval_baseline_dist.py`
- `minicpm_v46/streamingbench/eval_ctr.py`
- `minicpm_v46/streamingbench/eval_ctr_dist.py`
- `minicpm_v46/streamingbench/eval_streamingtom.py`
- `minicpm_v46/streamingbench/eval_streamingtom_dist.py`
- `minicpm_v46/streamingbench/eval_timechat.py`
- `minicpm_v46/streamingbench/eval_timechat_dist.py`

## Current Run Wrappers

- `minicpm_v46/ovo/run_baseline_allframes.sh`
- `minicpm_v46/ovo/run_ctr_g50.sh`
- `minicpm_v46/ovo/run_recent4.sh`
- `minicpm_v46/ovo/run_streamingtom_g50.sh`
- `minicpm_v46/ovo/run_timechat_retention.sh`
- `minicpm_v46/streamingbench/run_baseline_allframes.sh`
- `minicpm_v46/streamingbench/run_ctr_g50.sh`
- `minicpm_v46/streamingbench/run_recent4.sh`
- `minicpm_v46/streamingbench/run_streamingtom_g50.sh`
- `minicpm_v46/streamingbench/run_timechat_retention.sh`

## Qwen Baselines

Qwen files are grouped by role:

- `qwen/evals/`: Python evaluator entry points.
- `qwen/runs/`: local run wrappers.
- `qwen/submits/`: Slurm submit scripts.

These are older baseline/reproduction experiments, separate from the main
MiniCPM-V 4.6 StreamingTOM comparison path, but they are kept organized because
they are still useful references.

## Utilities

- `tools/summarize_profile_metrics.py`: summarizes result directories into
  latency, token, and GPU-memory tables.

## Legacy Baselines

The old CDAS/debug/probe files were removed from the active tree because they
were only used during exploration and are not part of the current experiments.
