"""Projection head for supervised contrastive learning.

A small MLP that maps the fused feature to a lower-dimensional, L2-normalized
embedding on which the SupCon loss operates. It is independent of the classifier
so contrastive and classification objectives can coexist (Experiment 3) or the
head can be dropped entirely (Experiments 1 and 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head (SimCLR / SupCon style), returning unit-norm vectors."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 128,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.normalize:
            z = F.normalize(z, dim=1)
        return z
