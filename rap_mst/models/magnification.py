"""Optional magnification-embedding module.

The unified model sees all four magnifications; this module lets it *condition*
on which magnification an image came from. It maps the magnification index to a
learned embedding and fuses it with the backbone feature, either by
concatenation (default, most expressive) or by projecting the embedding and
adding it (keeps the feature dimension unchanged).

Toggled entirely by config: Experiment 1 omits it; Experiments 2 and 3 enable it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rap_mst.constants import NUM_MAGNIFICATIONS


class MagnificationEmbedding(nn.Module):
    """Learned per-magnification embedding fused into the feature vector.

    Parameters
    ----------
    feature_dim:
        Dimension of the incoming backbone feature.
    embed_dim:
        Size of the magnification embedding.
    fusion:
        ``"concat"`` -> output dim = ``feature_dim + embed_dim``.
        ``"add"``    -> embedding is projected to ``feature_dim`` and added;
                        output dim = ``feature_dim``.
    num_magnifications:
        Vocabulary size for the embedding table (defaults to the 4 BreaKHis mags).
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int = 64,
        fusion: str = "concat",
        num_magnifications: int = NUM_MAGNIFICATIONS,
    ) -> None:
        super().__init__()
        if fusion not in ("concat", "add"):
            raise ValueError(f"Unknown fusion mode: {fusion!r}")
        self.fusion = fusion
        self.embedding = nn.Embedding(num_magnifications, embed_dim)

        if fusion == "concat":
            self.output_dim = feature_dim + embed_dim
            self.proj = None
        else:  # add
            self.output_dim = feature_dim
            self.proj = nn.Linear(embed_dim, feature_dim)

    def forward(self, features: torch.Tensor, mag_index: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(mag_index)  # [B, embed_dim]
        if self.fusion == "concat":
            return torch.cat([features, emb], dim=1)
        return features + self.proj(emb)
