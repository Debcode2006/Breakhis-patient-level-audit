"""Checkpoint saving / loading with full training-state for resume.

A checkpoint bundles everything needed to (a) resume training exactly and (b)
run inference later: model weights, optimizer, scheduler, AMP scaler, epoch,
best-metric bookkeeping, the RNG states, and the config the run used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False: our checkpoints intentionally contain the config dict
    # and RNG state, not just tensors.
    return torch.load(path, map_location=map_location, weights_only=False)


def build_state(
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a checkpoint state dict."""
    state: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if extra:
        state.update(extra)
    return state
