"""Feature fusion: multi-scale pyramid -> single feature vector.

Sits between the FPN and the downstream vector-consuming modules (magnification
embedding, projection head, classifier). It global-average-pools each pyramid
level to a vector and combines the per-level vectors into one representation.

Keeping fusion in its own module -- rather than folding it into the FPN or the
classifier -- means the pooling strategy can evolve (e.g. attention pooling for a
future reasoning module) without touching anything else, and the pooled vector
stays the canonical representation that Retrieval Memory / Prototype Learning will
key on.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """Global-pool each pyramid level and combine into a single vector.

    Parameters
    ----------
    in_channels_list:
        Channel count of each incoming map (all equal to the FPN ``out_channels``
        when the FPN is enabled; a single-element list when it is disabled).
    mode:
        ``"concat"`` -> output dim = sum of channels (levels kept distinct);
        ``"mean"`` / ``"sum"`` -> output dim = shared channel count (requires all
        levels to share the same channel dim, which the FPN guarantees).
    """

    def __init__(self, in_channels_list: Sequence[int], mode: str = "concat") -> None:
        super().__init__()
        if mode not in ("concat", "mean", "sum"):
            raise ValueError(f"Unknown fusion mode: {mode!r}")
        self.mode = mode
        self.in_channels_list = list(in_channels_list)
        self.pool = nn.AdaptiveAvgPool2d(1)

        if mode == "concat":
            self.output_dim = int(sum(self.in_channels_list))
        else:
            if len(set(self.in_channels_list)) != 1:
                raise ValueError(
                    f"fusion mode {mode!r} requires equal channels across levels, "
                    f"got {self.in_channels_list}."
                )
            self.output_dim = int(self.in_channels_list[0])

    def forward(self, maps: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(maps) != len(self.in_channels_list):
            raise ValueError(
                f"FeatureFusion expected {len(self.in_channels_list)} maps, got {len(maps)}."
            )
        vecs: List[torch.Tensor] = [self.pool(m).flatten(1) for m in maps]  # each [B, C_i]
        if self.mode == "concat":
            return torch.cat(vecs, dim=1)
        stacked = torch.stack(vecs, dim=0)  # [L, B, C]
        return stacked.sum(0) if self.mode == "sum" else stacked.mean(0)
