"""Config-driven loss construction."""

from __future__ import annotations

import torch

from rap_mst.losses.combined import CombinedLoss


def build_loss(cfg, class_weights: torch.Tensor | None = None) -> CombinedLoss:
    """Build the composite loss described by ``cfg.loss``.

    ``class_weights`` (from the datamodule) is applied to CrossEntropy only when
    ``cfg.loss.use_class_weights`` is set.
    """
    lcfg = cfg.loss
    weights = class_weights if getattr(lcfg, "use_class_weights", False) else None
    return CombinedLoss(
        ce_weight=lcfg.ce_weight,
        supcon_weight=lcfg.supcon_weight,
        class_weights=weights,
        supcon_temperature=getattr(lcfg, "supcon_temperature", 0.07),
    )
