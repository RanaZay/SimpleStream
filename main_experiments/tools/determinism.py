from __future__ import annotations

import os
import random


def configure_determinism(seed: int | str | None = None) -> int:
    """Set best-effort deterministic behavior for evaluation runs."""
    if seed is None:
        seed = os.environ.get("MINICPM_SEED", "42")
    seed_int = int(seed)

    os.environ.setdefault("PYTHONHASHSEED", str(seed_int))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed_int)

    try:
        import numpy as np

        np.random.seed(seed_int)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed_int)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_int)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    try:
        from transformers import set_seed

        set_seed(seed_int)
    except Exception:
        pass

    return seed_int
