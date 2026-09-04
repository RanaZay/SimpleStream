# PRISM: Selective Evidence for Efficient Streaming Video Understanding

This repository contains the PRISM paper-release implementation built on top of
SimpleStream and MiniCPM-V-4.6. The active code path keeps the same efficient
recent-frame backbone as SimpleStream, then selectively admits a small amount of
historical evidence only when the current visual context is insufficient and the
retrieved evidence passes lightweight consistency checks.

The repository has been cleaned for paper experiments. Exploratory launchers and
discontinued variants have been removed from the active tree so the main
benchmark paths are easier to audit and rerun.

## Selected Method

The selected method is:

```text
progressive_sufficiency_memory_clip_mmr_evidence_contract
```

PRISM uses:

- exact Recent-6 as the default current context
- timestamp-correct historical candidate eligibility
- CLIP-MMR retrieval over older valid frames
- candidate override with `gamma=0.2995`
- evidence arbitration with margin/drop checks
- temporal evidence band of `3-30` seconds
- K1 disagreement guard within `10` seconds
- at most `3` historical frames

Core implementation files:

```text
lib/minicpm/adaptive.py
lib/minicpm/progressive_sufficiency.py
lib/minicpm/prism_retrieval_variants.py
lib/shared/recent_window.py
```

## Environment

The MiniCPM experiments use the `stream35` environment on the AMD cluster:

```bash
source ~/.bashrc
conda activate stream35
```

Final PRISM settings:

```bash
PRISM_CLIP_MODE=evidence_contract
MINICPM_PSM_CLIP_OVERRIDE_THRESHOLD=0.2995
MINICPM_PSM_ARBITRATION_MIN_MARGIN=0.60
MINICPM_PSM_ARBITRATION_MAX_SUFFICIENCY_DROP=0.08
MINICPM_PSM_TEMPORAL_BAND_MIN_SECONDS=3
MINICPM_PSM_TEMPORAL_BAND_MAX_SECONDS=30
MINICPM_PSM_CANDIDATE_K1_DISAGREE_MAX_DISTANCE_SECONDS=10
MINICPM_PSM_MAX_MEMORY_FRAMES=3
MINICPM_PSM_HISTORY_SEARCH_CHUNKS=64
MINICPM_PSM_HISTORY_CANDIDATE_POOL=12
MINICPM_PSM_MMR_LAMBDA=0.80
```

## Data Layout

Place benchmark data under `data/`:

```text
data/
  streamingbench/
    questions_real.json
    videos/
  ovo_bench/
    ovo_bench_new.json
    chunked_videos/
  egoschema/
    ...
  streambench_v0_3/
    streaming_bench_v0.3.json
    Ego/
    Movie/
    WebVideo/
```

`data/` is ignored by git.

## StreamingBench

Corrected exact Recent-6 baseline:

```bash
sbatch main_experiments/minicpm_v46/streamingbench/submit_exact_recent6_full_amd.sh
```

Final PRISM:

```bash
sbatch main_experiments/minicpm_v46/streamingbench/submit_prism_evidence_contract_full_amd.sh
```

Current validated full StreamingBench results:

| Method | Correct | Accuracy | Avg Frames | Avg Vision Tokens | TTFT | E2E |
|---|---:|---:|---:|---:|---:|---:|
| Corrected exact Recent-6 | 1903/2499 | 76.15 | 6.00 | 395.7 | 0.505s | 17.313s |
| PRISM evidence contract | 1925/2499 | 77.03 | 6.08 | 401.4 | 0.521s | 19.166s |

Reconstructed readable logs are kept in `logs/` when available:

```text
logs/RECONSTRUCTED_streamingbench_full_exact_recent6_76p15_20260901.out
logs/RECONSTRUCTED_streamingbench_full_prism_evidence_contract_77p03_20260903.out
```

## OVO-Bench

Exact Recent-6 baseline:

```bash
sbatch main_experiments/minicpm_v46/ovo/submit_prism_exact_current6_8gpu_amd.sh
```

Final PRISM:

```bash
sbatch main_experiments/minicpm_v46/ovo/submit_prism_guarded_exact_recent_8gpu_amd.sh
```

Before interpreting OVO as a final paper result, verify the Slurm log includes:

```text
ADAPTIVE_MODE=progressive_sufficiency_memory_clip_mmr_evidence_contract
PRISM_CLIP_MODE=evidence_contract
```

## EgoSchema500

Exact Recent-6 baseline:

```bash
sbatch main_experiments/minicpm_v46/egoschema/submit_recent6_8gpu_amd.sh
```

Final PRISM:

```bash
sbatch main_experiments/minicpm_v46/egoschema/submit_prism_guarded_exact_recent_8gpu_amd.sh
```

## StreamBench-v0.3

StreamBench-v0.3 is open-ended, while the current PRISM evidence-contract
controller is designed around MCQ option evidence. Use these runs as
generalization diagnostics:

```bash
sbatch main_experiments/minicpm_v46/streambench_v03/submit_streambench_v03_full_recent6_amd.sh
sbatch main_experiments/minicpm_v46/streambench_v03/submit_streambench_v03_full_prism_amd.sh
```

The StreamBench-v0.3 report includes:

```text
OS | LM | SM | CI | KG | SF | Avg
```

## Active Directory Map

```text
lib/minicpm/                         MiniCPM wrappers and PRISM logic
lib/shared/                          shared decoding/scoring utilities
main_experiments/minicpm_v46/         benchmark evaluators and Slurm launchers
main_experiments/qwen/                Qwen reproduction baselines
scoring/                              benchmark scoring helpers
efficiency/                           standalone efficiency tools
archive/                              archived exploratory material
```

## Notes

The original SimpleStream project showed that a small recent-frame window is a
strong streaming-video baseline. PRISM keeps that principle intact and adds a
small, selective evidence mechanism instead of uniformly increasing historical
context.

For the original SimpleStream paper, see arXiv `2604.02317`.
