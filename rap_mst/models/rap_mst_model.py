"""RAP-MST model assembler.

A *single* model class assembles all experiments from independent, optional
components rather than hard-coding separate architectures:

    image
      |
    SwinBackbone          (always)  -> list of hierarchical stage maps
      |
    FPN                   (default; optional)  -> unified multi-scale pyramid
      |
    FeatureFusion         (always)  -> single pooled feature vector
      |
    MagnificationEmbedding (optional -- Experiments 2, 3)
      |
      +--> ProjectionHead  (optional -- Experiment 3, for SupCon)
      |
    ClassificationHead    (always)

The forward pass returns a dict of named outputs so the training loop and every
future module can pick what it needs:

    logits        classification logits                          [B, num_classes]
    features      fused vector after FPN + fusion (pre-mag)       [B, fused_dim]
    embeddings    feature after magnification fusion              [B, embed_dim]
    projections   L2-normalized SupCon embedding (or None)        [B, proj_dim]
    fpn_features  list of multi-scale FPN maps (or None)          [B, C, Hi, Wi]

``features`` -- **not** ``embeddings`` -- is the canonical *vector* representation
the Retrieval Memory indexes and queries (``docs/retrieval.md`` D1): when the
magnification embedding is enabled, ``embeddings`` carries a 64-d lookup-table block
that is identical for every image at the same zoom, which makes cosine retrieval
99.6-100% magnification-locked. ``features`` is that vector with the block stripped,
and it equals ``embeddings`` whenever magnification is disabled (as on exp3n, the
retrieval base encoder), so one code path covers every experiment.
``fpn_features`` additionally exposes the *spatial* multi-scale pyramid for future
dense / region-level modules. Adding such a module means consuming this dict --
the Retrieval Memory did exactly that and required no change to this class.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from rap_mst.models.backbone import SwinBackbone
from rap_mst.models.classifier import ClassificationHead
from rap_mst.models.fpn import FPN
from rap_mst.models.fusion import FeatureFusion
from rap_mst.models.magnification import MagnificationEmbedding
from rap_mst.models.projection import ProjectionHead


class RAPMSTModel(nn.Module):
    """Modular assembly of backbone + FPN + fusion + optional heads."""

    def __init__(
        self,
        backbone: SwinBackbone,
        fusion: FeatureFusion,
        classifier: ClassificationHead,
        fpn: Optional[FPN] = None,
        magnification: Optional[MagnificationEmbedding] = None,
        projection: Optional[ProjectionHead] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.fusion = fusion
        self.magnification = magnification
        self.projection = projection
        self.classifier = classifier

    @property
    def uses_fpn(self) -> bool:
        return self.fpn is not None

    @property
    def uses_magnification(self) -> bool:
        return self.magnification is not None

    @property
    def uses_projection(self) -> bool:
        return self.projection is not None

    def forward(self, image: torch.Tensor, mag_index: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        stage_maps = self.backbone(image)  # list of NCHW maps (fine -> coarse)

        if self.fpn is not None:
            pyramid: Optional[List[torch.Tensor]] = self.fpn(stage_maps)
            fused = self.fusion(pyramid)
        else:
            # FPN disabled: fall back to pooling the deepest (most semantic) stage.
            pyramid = None
            fused = self.fusion([stage_maps[-1]])

        if self.magnification is not None:
            if mag_index is None:
                raise ValueError("Model uses magnification embedding but mag_index was not provided.")
            embeddings = self.magnification(fused, mag_index)
        else:
            embeddings = fused

        projections = self.projection(embeddings) if self.projection is not None else None
        logits = self.classifier(embeddings)

        return {
            "logits": logits,
            "features": fused,
            "embeddings": embeddings,
            "projections": projections,
            "fpn_features": pyramid,
        }
