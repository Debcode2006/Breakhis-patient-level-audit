"""Classification head.

A configurable linear or single-hidden-layer MLP mapping the fused feature to
class logits. Kept deliberately thin so the representation-learning burden stays
on the backbone / projection modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Linear (default) or MLP classifier over the fused feature."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
