"""Hierarchical Swin Transformer backbone (multi-scale feature extractor).

The backbone is a pure, head-free extractor that returns the **list of
hierarchical stage feature maps** produced by Swin -- e.g. for
``swin_tiny_patch4_window7_224`` at 224px:

    stage 0: [B,  96, 56, 56]   (stride  4)
    stage 1: [B, 192, 28, 28]   (stride  8)
    stage 2: [B, 384, 14, 14]   (stride 16)
    stage 3: [B, 768,  7,  7]   (stride 32)

These maps are consumed by the FPN (`models/fpn.py`), which aggregates them
before any pooling happens. Pooling / vectorization is deliberately NOT done here
-- that is the FeatureFusion module's job -- so the backbone stays a clean,
swappable feature source and the FPN logic lives outside it.

timm's Swin ``features_only`` output can be channels-last (NHWC); this wrapper
normalizes every map to NCHW using the known per-stage channel counts, so the
rest of the pipeline can assume standard conv layout regardless of timm version.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import timm
import torch
import torch.nn as nn


class SwinBackbone(nn.Module):
    """Multi-scale Swin feature extractor.

    Parameters
    ----------
    variant:
        Any hierarchical timm Swin model name, e.g. ``swin_tiny_patch4_window7_224``.
    pretrained:
        Load ImageNet-pretrained weights.
    drop_rate / drop_path_rate:
        Regularization passed through to timm.
    out_indices:
        Which Swin stages to expose (default all four). Order is fine -> coarse
        (increasing stride / decreasing resolution).

    Attributes
    ----------
    feature_channels:
        Per-stage channel counts, aligned with ``out_indices`` order.
    feature_reductions:
        Per-stage stride (spatial reduction) factors.
    """

    def __init__(
        self,
        variant: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        out_indices: Sequence[int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        self.out_indices: Tuple[int, ...] = tuple(out_indices)
        # features_only exposes the hierarchical stage maps instead of a pooled
        # classifier output. num_classes / global_pool are invalid here.
        self.model = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.feature_channels: List[int] = list(self.model.feature_info.channels())
        self.feature_reductions: List[int] = list(self.model.feature_info.reduction())

    def _to_nchw(self, feat: torch.Tensor, channels: int) -> torch.Tensor:
        """Normalize a stage output to NCHW.

        timm's Swin may emit NHWC ``[B, H, W, C]``. We detect this by comparing
        the last dim against the known channel count (channel counts like
        96/192/384/768 never collide with spatial dims at supported resolutions).
        """
        if feat.dim() != 4:
            raise ValueError(f"Expected 4D stage feature, got shape {tuple(feat.shape)}")
        if feat.shape[1] != channels and feat.shape[-1] == channels:
            feat = feat.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        return feat

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.model(x)
        return [self._to_nchw(f, c) for f, c in zip(feats, self.feature_channels)]
