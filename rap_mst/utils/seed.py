"""Reproducibility helpers.

The seed used for a run is saved into that run's config so any experiment can be
reproduced bit-for-bit (as far as the hardware allows). Determinism is opt-in via
the config because fully deterministic cuDNN kernels are slower and, on a 4 GB
GPU, throughput matters.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    Parameters
    ----------
    seed:
        Global seed value; also recorded in the run config.
    deterministic:
        When True, request deterministic cuDNN algorithms and disable the
        autotuner. When False, enable ``cudnn.benchmark`` for speed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required so deterministic CuBLAS GEMMs are actually deterministic on
        # CUDA >= 10.2; also silences the per-op UserWarning that otherwise
        # prints every epoch. Must be set before the CuBLAS handle is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # `warn_only` keeps ops without a deterministic implementation from
        # crashing training while still surfacing them.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for reproducible augmentation per worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
