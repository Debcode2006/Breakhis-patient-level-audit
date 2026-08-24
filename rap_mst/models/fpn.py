"""Feature Pyramid Network (FPN).

Aggregates the hierarchical multi-scale feature maps from the Swin backbone into
a semantically-rich pyramid with a **unified channel dimension**. This is the
default feature aggregator for every experiment in the repository.

Design (Lin et al., 2017):
    * lateral 1x1 convs project each stage to ``out_channels``;
    * a top-down pathway upsamples coarser levels and adds them into finer ones,
      injecting high-level semantics into high-resolution maps;
    * 3x3 output convs smooth the merged maps (anti-aliasing the nearest-neighbor
      upsampling).

The FPN is intentionally independent of the backbone and of the pooling step: it
consumes a list of NCHW maps and returns a list of NCHW maps at the same spatial
resolutions, each with ``out_channels`` channels. Downstream modules
(FeatureFusion now; Retrieval Memory / Prototype Learning later) can consume this
pyramid without any change to the backbone or FPN.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Top-down Feature Pyramid Network over hierarchical backbone maps.

    Parameters
    ----------
    in_channels_list:
        Channel count of each input stage, in fine -> coarse order
        (e.g. ``[96, 192, 384, 768]``).
    out_channels:
        Unified channel dimension of every pyramid level.
    norm:
        If True, apply GroupNorm after each conv (stabilizes training when the
        backbone is pretrained but the FPN is trained from scratch).
    """

    def __init__(
        self,
        in_channels_list: Sequence[int],
        out_channels: int = 256,
        norm: bool = True,
    ) -> None:
        super().__init__()
        if len(in_channels_list) < 1:
            raise ValueError("FPN needs at least one input level.")
        self.out_channels = out_channels
        self.num_levels = len(in_channels_list)

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(self._conv(in_ch, out_channels, kernel_size=1, norm=norm))
            self.output_convs.append(self._conv(out_channels, out_channels, kernel_size=3, norm=norm))

    @staticmethod
    def _conv(in_ch: int, out_ch: int, kernel_size: int, norm: bool) -> nn.Module:
        padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=not norm)
        ]
        if norm:
            # GroupNorm(32, C) is the standard choice for detection FPNs and is
            # batch-size independent -- important given the small 4 GB batch.
            num_groups = 32 if out_ch % 32 == 0 else 8 if out_ch % 8 == 0 else 1
            layers.append(nn.GroupNorm(num_groups, out_ch))
        return nn.Sequential(*layers)

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(feats) != self.num_levels:
            raise ValueError(
                f"FPN expected {self.num_levels} feature maps, got {len(feats)}."
            )

        # 1x1 lateral projections.
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, feats)]

        # Top-down pathway: coarsest (last) -> finest (first).
        for i in range(self.num_levels - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 3x3 smoothing to produce the final pyramid.
        return [out_conv(lat) for out_conv, lat in zip(self.output_convs, laterals)]
