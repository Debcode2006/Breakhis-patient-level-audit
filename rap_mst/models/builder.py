"""Config-driven model construction.

`build_model` reads the config and assembles a :class:`RAPMSTModel` with exactly
the components an experiment needs, tracking the running feature dimension so the
classifier / projection heads are always sized correctly:

    backbone stage channels
        -> FPN (unifies to fpn.out_channels)         [if model.fpn.enabled]
        -> FeatureFusion (pools pyramid -> vector)   -> fused_dim
        -> (+ magnification embedding)               -> fused_dim
        -> projection / classifier sized to fused_dim

Which components are enabled is derived from explicit flags under ``model`` set by
the experiment presets, so switching experiments is purely a config change. The
FPN is on by default for every experiment.
"""

from __future__ import annotations

from typing import List

from rap_mst.models.backbone import SwinBackbone
from rap_mst.models.classifier import ClassificationHead
from rap_mst.models.fpn import FPN
from rap_mst.models.fusion import FeatureFusion
from rap_mst.models.magnification import MagnificationEmbedding
from rap_mst.models.projection import ProjectionHead
from rap_mst.models.rap_mst_model import RAPMSTModel


def build_model(cfg) -> RAPMSTModel:
    """Instantiate the model described by ``cfg.model``."""
    mcfg = cfg.model

    # --- Backbone: hierarchical multi-scale maps ------------------------- #
    out_indices = tuple(getattr(mcfg.backbone, "out_indices", (0, 1, 2, 3)))
    backbone = SwinBackbone(
        variant=mcfg.backbone.variant,
        pretrained=mcfg.backbone.pretrained,
        drop_rate=getattr(mcfg.backbone, "drop_rate", 0.0),
        drop_path_rate=getattr(mcfg.backbone, "drop_path_rate", 0.1),
        out_indices=out_indices,
    )
    stage_channels: List[int] = backbone.feature_channels

    # --- FPN (default) + fusion channel bookkeeping ---------------------- #
    fpn = None
    if mcfg.fpn.enabled:
        fpn = FPN(
            in_channels_list=stage_channels,
            out_channels=mcfg.fpn.out_channels,
            norm=getattr(mcfg.fpn, "norm", True),
        )
        fusion_channels = [mcfg.fpn.out_channels] * len(stage_channels)
    else:
        # No FPN: fusion pools only the deepest stage (baseline Swin behavior).
        fusion_channels = [stage_channels[-1]]

    fusion = FeatureFusion(fusion_channels, mode=mcfg.fusion.mode)
    fused_dim = fusion.output_dim

    # --- Optional magnification embedding -------------------------------- #
    magnification = None
    if mcfg.magnification.enabled:
        magnification = MagnificationEmbedding(
            feature_dim=fused_dim,
            embed_dim=mcfg.magnification.embed_dim,
            fusion=mcfg.magnification.fusion,
        )
        fused_dim = magnification.output_dim

    # --- Optional projection head (SupCon) ------------------------------- #
    projection = None
    if mcfg.projection.enabled:
        projection = ProjectionHead(
            in_dim=fused_dim,
            hidden_dim=mcfg.projection.hidden_dim,
            out_dim=mcfg.projection.out_dim,
            normalize=getattr(mcfg.projection, "normalize", True),
        )

    # --- Classifier ------------------------------------------------------ #
    classifier = ClassificationHead(
        in_dim=fused_dim,
        num_classes=mcfg.classifier.num_classes,
        hidden_dim=getattr(mcfg.classifier, "hidden_dim", None),
        dropout=getattr(mcfg.classifier, "dropout", 0.0),
    )

    return RAPMSTModel(
        backbone=backbone,
        fusion=fusion,
        classifier=classifier,
        fpn=fpn,
        magnification=magnification,
        projection=projection,
    )
