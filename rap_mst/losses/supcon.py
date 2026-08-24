"""Supervised Contrastive Loss (Khosla et al., 2020).

Operates on the L2-normalized projection-head embeddings. For each anchor, all
other samples sharing its label are positives; the loss pulls positives together
and pushes negatives apart in the embedding space.

The implementation accepts either single-view features ``[B, D]`` (positives come
from other same-label images in the batch) or multi-view features
``[B, n_views, D]`` for the two-crop pipeline, so enabling explicit two-view
augmentation later needs no loss change.

Numerical precision
-------------------
The whole computation is forced to fp32 via an inner ``autocast(enabled=False)``
region. This is deliberate and matches the reference implementation: the trainer
runs the forward pass (and therefore this loss) inside a mixed-precision autocast
context, so the incoming projections are fp16. Computing the similarity matrix and
its ``exp``/``log`` in fp16 quantizes the cosine-similarity logits and can flush
hard-negative exponentials to zero, silently degrading the contrastive gradient.
Upcasting to fp32 here removes that failure mode regardless of the surrounding
autocast state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss with label-defined positives.

    Parameters
    ----------
    temperature:
        Softmax temperature scaling the similarity logits.
    base_temperature:
        Normalization constant from the original paper (kept configurable).
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out_dtype = features.dtype

        # Force fp32 for the entire contrastive computation (see module docstring).
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.float()

            # Normalize shape to [B, n_views, D].
            if features.dim() == 2:
                features = features.unsqueeze(1)
            if features.dim() != 3:
                raise ValueError(f"features must be [B, D] or [B, n_views, D], got {tuple(features.shape)}")

            # Re-normalize defensively so the loss owns its unit-norm invariant
            # rather than trusting an upstream flag. Idempotent when the projection
            # head already normalized, and done in fp32 after the upcast so the
            # temperature scaling operates on true cosine similarities.
            features = F.normalize(features, dim=-1)

            device = features.device
            batch_size, n_views, _ = features.shape
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("labels length must match batch size.")

            # Degenerate batch: no cross-sample pair exists.
            if batch_size * n_views <= 1:
                return (features.sum() * 0.0).to(out_dtype)

            # Positive mask over the flattened (B * n_views) set.
            mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]
            contrast = torch.cat(torch.unbind(features, dim=1), dim=0)  # [B*n_views, D]
            anchor = contrast
            anchor_count = n_views

            # Similarity logits with numerical stabilization.
            logits = torch.matmul(anchor, contrast.T) / self.temperature
            logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

            mask = mask.repeat(anchor_count, n_views)  # tile to [B*nv, B*nv]
            # Remove self-comparisons.
            logits_mask = torch.ones_like(mask)
            logits_mask.fill_diagonal_(0)
            mask = mask * logits_mask

            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

            pos_per_anchor = mask.sum(dim=1)
            # Anchors with no positive in the batch contribute nothing.
            valid = pos_per_anchor > 0
            mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / pos_per_anchor[valid]

            if mean_log_prob_pos.numel() == 0:
                return (features.sum() * 0.0).to(out_dtype)  # keeps graph connected, zero loss

            loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()

        return loss.to(out_dtype)
