from __future__ import annotations

import os

import torch

from lib.recent_window_eval_qwen3 import (
    RecentWindowQAModel as _Qwen3RecentWindowQAModel,
    evaluate_ovo_backward_realtime,
    evaluate_ovo_forward,
    print_ovo_results,
    query_recent_window,
)


class RecentWindowQAModel(_Qwen3RecentWindowQAModel):
    """Qwen3.5 compatibility wrapper for the SimpleStream Qwen3 path.

    Newer Qwen3.5 Transformers code can return a BaseModelOutputWithPooling
    object from get_image_features instead of the tensor/tuple returned by the
    Qwen3-VL release used in the SimpleStream paper.
    """

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            attn_implementation=attn_implementation or os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        )

    def _flatten_vision_features(self, features):
        if hasattr(features, "last_hidden_state") and isinstance(features.last_hidden_state, torch.Tensor):
            tensor = features.last_hidden_state
        elif hasattr(features, "pooler_output") and isinstance(features.pooler_output, torch.Tensor):
            tensor = features.pooler_output
        else:
            return super()._flatten_vision_features(features)

        if tensor.dim() == 3 and tensor.shape[0] == 1:
            return tensor.squeeze(0)
        if tensor.dim() > 2:
            return tensor.reshape(-1, tensor.shape[-1])
        return tensor
