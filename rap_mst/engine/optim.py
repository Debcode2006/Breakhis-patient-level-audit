"""Optimizer and LR scheduler construction from config."""

from __future__ import annotations

from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, _LRScheduler


def build_optimizer(cfg, model: torch.nn.Module) -> Optimizer:
    ocfg = cfg.optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    name = ocfg.name.lower()

    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=ocfg.lr, weight_decay=ocfg.weight_decay,
            betas=tuple(getattr(ocfg, "betas", (0.9, 0.999))),
        )
    if name == "adam":
        return torch.optim.Adam(params, lr=ocfg.lr, weight_decay=ocfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=ocfg.lr, momentum=getattr(ocfg, "momentum", 0.9),
            weight_decay=ocfg.weight_decay, nesterov=getattr(ocfg, "nesterov", True),
        )
    raise ValueError(f"Unknown optimizer: {ocfg.name}")


def build_scheduler(cfg, optimizer: Optimizer, steps_per_epoch: int) -> Optional[_LRScheduler]:
    scfg = getattr(cfg, "scheduler", None)
    if scfg is None or not getattr(scfg, "name", None) or scfg.name.lower() == "none":
        return None

    name = scfg.name.lower()
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs, eta_min=getattr(scfg, "min_lr", 1e-6)
        )
    if name == "step":
        return StepLR(optimizer, step_size=scfg.step_size, gamma=getattr(scfg, "gamma", 0.1))
    raise ValueError(f"Unknown scheduler: {scfg.name}")
