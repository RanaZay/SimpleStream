# SimpleStream Recent-4 Reproduction

This note documents the safe reproduction setup for Table 1 style runs with
`recent_frames_only=4`, `chunk_duration=1.0`, and `fps=1.0`.

## Data Layout

The launchers intentionally use only benchmark data placed inside this clone:

```text
data/
  ovo_bench/
    ovo_bench_new.json
    chunked_videos/
      ...
  streamingbench/
    questions_real.json
    videos/
      ...
```

`data/` is ignored by git, so the benchmark files should be copied locally on
the machine where the runs will execute.

## Launchers

OVO-Bench:

```bash
PYTHON_BIN=/path/to/qwen3/env/bin/python \
  bash main_experiments/qwen/runs/run_repro_qwen3vl_ovo_recent4.sh

PYTHON_BIN=/path/to/qwen25/env/bin/python \
  bash main_experiments/qwen/runs/run_repro_qwen25vl_ovo_recent4.sh
```

StreamingBench:

```bash
PYTHON_BIN=/path/to/qwen3/env/bin/python \
  bash main_experiments/qwen/runs/run_repro_qwen3vl_streamingbench_recent4.sh

PYTHON_BIN=/path/to/qwen25/env/bin/python \
  bash main_experiments/qwen/runs/run_repro_qwen25vl_streamingbench_recent4.sh
```

The Qwen3-VL and Qwen2.5-VL runs require separate environments because the
repository pins different `transformers` and `accelerate` versions for each.

## Safety

Each launcher resolves the benchmark paths and refuses to run if the data is not
under this repository's `data/` directory. This avoids accidentally using another
user's benchmark copy on a shared machine.
