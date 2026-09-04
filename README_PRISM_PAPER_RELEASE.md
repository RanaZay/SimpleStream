# PRISM Paper Release Guide

This repository contains the selected PRISM implementation and the exact Recent-6 baseline used for the paper experiments.

## Selected Method

**PRISM evidence contract** is the final selected method.

Core implementation files:

- `lib/minicpm/adaptive.py`
- `lib/minicpm/progressive_sufficiency.py`
- `lib/minicpm/prism_retrieval_variants.py`

The selected controller uses:

- exact Recent-6 as K0
- timestamp-correct historical candidate eligibility
- CLIP-MMR retrieval
- candidate override with `gamma=0.2995`
- evidence arbitration with margin/drop checks
- temporal band `3-30` seconds
- candidate/K1 disagreement guard within `10` seconds

Key environment values:

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

## Main StreamingBench Commands

Exact Recent-6 baseline:

```bash
sbatch main_experiments/minicpm_v46/streamingbench/submit_exact_recent6_full_amd.sh
```

Final PRISM:

```bash
sbatch main_experiments/minicpm_v46/streamingbench/submit_prism_evidence_contract_full_amd.sh
```

Generic PRISM launcher, if manual overrides are needed:

```bash
PRISM_CLIP_MODE=evidence_contract \
MINICPM_PSM_CLIP_OVERRIDE_THRESHOLD=0.2995 \
MINICPM_PSM_ARBITRATION_MIN_MARGIN=0.60 \
MINICPM_PSM_ARBITRATION_MAX_SUFFICIENCY_DROP=0.08 \
MINICPM_PSM_TEMPORAL_BAND_MIN_SECONDS=3 \
MINICPM_PSM_TEMPORAL_BAND_MAX_SECONDS=30 \
MINICPM_PSM_CANDIDATE_K1_DISAGREE_MAX_DISTANCE_SECONDS=10 \
sbatch main_experiments/minicpm_v46/streamingbench/submit_prism_clip_retrieval_full_amd.sh
```

## OVO-Bench Commands

Exact Recent-6 baseline:

```bash
sbatch main_experiments/minicpm_v46/ovo/submit_prism_exact_current6_8gpu_amd.sh
```

Final PRISM:

```bash
sbatch main_experiments/minicpm_v46/ovo/submit_prism_guarded_exact_recent_8gpu_amd.sh
```

## EgoSchema500 Commands

Exact Recent-6 baseline:

```bash
sbatch main_experiments/minicpm_v46/egoschema/submit_recent6_8gpu_amd.sh
```

Final PRISM:

```bash
sbatch main_experiments/minicpm_v46/egoschema/submit_prism_guarded_exact_recent_8gpu_amd.sh
```

## StreamBench-v0.3 Diagnostic Commands

StreamBench-v0.3 is open-ended, while PRISM evidence contract is MCQ-oriented.
Use this only as a generalization diagnostic.

```bash
sbatch main_experiments/minicpm_v46/streambench_v03/submit_streambench_v03_full_recent6_amd.sh
sbatch main_experiments/minicpm_v46/streambench_v03/submit_streambench_v03_full_prism_amd.sh
```

The output table reports:

```text
OS | LM | SM | CI | KG | SF | Avg
```

## Current Full StreamingBench Result

| Method | Correct | Accuracy | Avg Frames | Avg Vision Tokens | TTFT | E2E |
|---|---:|---:|---:|---:|---:|---:|
| Corrected exact Recent-6 | 1903/2499 | 76.15 | 6.00 | 395.7 | 0.505s | 17.313s |
| PRISM evidence contract | 1925/2499 | 77.03 | 6.08 | 401.4 | 0.521s | 19.166s |

## Archive

Exploratory launchers and discontinued variants are under:

```text
archive/exploratory_methods_2026_09/
```

They are retained for ablation provenance but are not part of the main method.
