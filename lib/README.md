# Library Modules

This folder contains reusable model wrappers and StreamingTOM components.
Current MiniCPM code is grouped under `minicpm/`, Qwen wrappers are grouped
under `qwen/`, shared evaluation utilities are grouped under `shared/`, and
current StreamingTOM components are grouped under `streamingtom/`, and
TimeChat-Online style token dropping is grouped under `timechat/`.

## Current MiniCPM-V 4.6 Wrappers

- `minicpm/baseline.py`: baseline MiniCPM-V 4.6 wrapper used for
  recent-window and all-frame evaluation.
- `minicpm/ctr.py`: MiniCPM-V 4.6 wrapper with CTR visual-token
  compression.
- `minicpm/streamingtom.py`: MiniCPM-V 4.6 wrapper with CTR plus
  OQM.
- `minicpm/timechat.py`: MiniCPM-V 4.6 wrapper with TimeChat-Online
  feature-level Differential Token Drop.

## StreamingTOM Components

- `streamingtom/ctr.py`: Causal Temporal Reduction implementation.
- `streamingtom/oqm.py`: Online Quantized Memory implementation.

## TimeChat-Online Components

- `timechat/dtd.py`: feature-level Differential Token Drop implementation
  with fixed visual-token retention ratios for 100%, 80%, and 40% runs.

## Shared Utilities

- `shared/recent_window.py`: common recent-window dataclasses, video decoding,
  prompt helpers, scoring, and distributed result utilities.

## Qwen Wrappers

- `qwen/qwen25.py`: Qwen2.5-VL recent-window evaluator wrapper.
- `qwen/qwen3.py`: Qwen3-VL recent-window evaluator wrapper.
- `qwen/qwen35.py`: Qwen3.5-VL recent-window evaluator wrapper.
- `qwen/qwen35_default.py`: Qwen3.5-VL default-decoding variant.
- `qwen/qwen35_thinking.py`: Qwen3.5-VL thinking-mode variant.
- `qwen/exact_recent_decoder.py`: exact recent-frame decoding helper.

## Other Model Wrappers

- `cdas_sampler.py`: dormant support for the old CDAS idea. Active all-frame
  MiniCPM/CTR/StreamingTOM runs do not enable it.
