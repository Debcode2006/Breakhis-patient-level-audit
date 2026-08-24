"""Composite loss that linearly combines named loss terms.

Keeping each objective independent and summing them with configurable weights is
what makes the loss extensible: a future retrieval or prototype loss is added by
registering one more term, without rewriting the training step. The combined loss
returns both the scalar total and a per-term breakdown for logging.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from rap_mst.losses.supcon import SupConLoss


class CombinedLoss(nn.Module):
    """Weighted sum of CrossEntropy and (optionally) SupCon.

    Parameters
    ----------
    ce_weight:
        Weight on the CrossEntropy term (0 disables it).
    supcon_weight:
        Weight on the SupCon term (0 disables it).
    class_weights:
        Optional per-class weights for CrossEntropy (handles the 24:58 imbalance).
    supcon_temperature:
        Temperature for the SupCon term.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        supcon_weight: float = 0.0,
        class_weights: torch.Tensor | None = None,
        supcon_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.supcon_weight = supcon_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.supcon = SupConLoss(temperature=supcon_temperature)

    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return ``(total_loss, {term_name: value})``.

        ``outputs`` is the dict produced by :class:`RAPMSTModel.forward`.
        """
        total = outputs["logits"].new_zeros(())
        parts: Dict[str, float] = {}

        if self.ce_weight > 0:
            ce = self.ce(outputs["logits"], labels)
            total = total + self.ce_weight * ce
            parts["ce"] = float(ce.detach())

        if self.supcon_weight > 0:
            if outputs.get("projections") is None:
                raise ValueError(
                    "SupCon weight > 0 but model produced no projections. "
                    "Enable model.projection for this experiment."
                )
            sc = self.supcon(outputs["projections"], labels)
            total = total + self.supcon_weight * sc
            parts["supcon"] = float(sc.detach())

        parts["total"] = float(total.detach())
        return total, parts
