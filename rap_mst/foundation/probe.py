"""The probe head and its fit loop (Stage F2).

This is deliberately the smallest thing that can answer the question. The baseline
claim is "*frozen* pathology foundation model + a linear head", so the head is a
``Dropout -> Linear(D -> 2)`` -- the same shape as ``ClassificationHead`` with
``hidden_dim: null``, which is what exp1-exp3n use. ``head: mlp`` adds one hidden
layer, the "optionally a small MLP" variant of the same experiment
(paper S3.8).

Everything about the fit that could make the comparison unfair is pinned to what
the Swin ladder actually did, not to what would flatter the baseline:

* selection metric ``accuracy`` (image-level validation accuracy) with the same
  early-stopping semantics as :class:`rap_mst.engine.trainer.Trainer`;
* inverse-frequency class weights on CrossEntropy (``loss.use_class_weights``);
* AdamW + cosine to ``min_lr``;
* patient-level metrics reported alongside image-level, from the same
  ``rap_mst.utils.metrics`` functions.

The one thing it does *not* share is the training loop, because there is nothing to
share: with the encoder frozen and the features cached, a fold is a
[~4.5k x 768] tensor that lives on the GPU and trains in seconds. Re-using
``Trainer`` would mean re-encoding 4.5k images per epoch to change ~1.5k
parameters.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rap_mst.utils.metrics import compute_metrics, patient_level_metrics


# --------------------------------------------------------------------------- #
# Feature standardisation -- fitted on TRAIN rows only
# --------------------------------------------------------------------------- #
class Standardizer:
    """Per-dimension z-score whose statistics come from one split only.

    Kept as an explicit object (rather than a couple of tensors) so it can be
    stored in the checkpoint and re-applied verbatim at test time: a probe scored
    with statistics refitted on the test set would be a leak that no accuracy
    number would reveal.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, fitted_on: Dict[str, Any] | None = None):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.fitted_on: Dict[str, Any] = dict(fitted_on or {})

    @classmethod
    def fit(cls, features: np.ndarray, fitted_on: Dict[str, Any] | None = None) -> "Standardizer":
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)  # a dead dimension stays dead, not inf
        return cls(mean, std, fitted_on)

    @classmethod
    def identity(cls, dim: int) -> "Standardizer":
        return cls(np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32),
                   {"mode": "identity"})

    def apply(self, features: np.ndarray) -> np.ndarray:
        return ((np.asarray(features, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "fitted_on": self.fitted_on}

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "Standardizer":
        return cls(state["mean"], state["std"], state.get("fitted_on"))


# --------------------------------------------------------------------------- #
# Head
# --------------------------------------------------------------------------- #
class ProbeHead(nn.Module):
    """``linear``: Dropout -> Linear.  ``mlp``: Dropout -> Linear -> ReLU -> Linear."""

    def __init__(self, in_dim: int, num_classes: int = 2, head: str = "linear",
                 hidden_dim: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        head = head.strip().lower()
        if head not in ("linear", "mlp"):
            raise ValueError(f"foundation.probe.head must be 'linear' or 'mlp', got {head!r}.")
        self.head = head
        self.in_dim = in_dim
        layers = [nn.Dropout(dropout)] if dropout > 0 else []
        if head == "mlp":
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            layers += [nn.Linear(hidden_dim, num_classes)]
        else:
            layers += [nn.Linear(in_dim, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def describe(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{self.head} head over {self.in_dim}-d features ({n:,} parameters)"


# --------------------------------------------------------------------------- #
# Class weights -- identical formula to BreaKHisDataModule.class_weights()
# --------------------------------------------------------------------------- #
def class_weights_from(labels: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes)
    total = int(counts.sum())
    return torch.tensor(
        [total / (num_classes * max(int(c), 1)) for c in counts], dtype=torch.float32
    )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(head: nn.Module, features: torch.Tensor) -> np.ndarray:
    """P(malignant) for every row. Batched only for memory, not for speed."""
    head.eval()
    out = []
    for start in range(0, features.shape[0], 8192):
        logits = head(features[start:start + 8192])
        out.append(F.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def score(labels: np.ndarray, probs: np.ndarray, patient_ids: np.ndarray,
          threshold: float = 0.5) -> Dict[str, float]:
    """Image-level + patient-level metrics, from the project's shared functions."""
    preds = (probs >= threshold).astype(int)
    metrics = compute_metrics(labels, preds, probs)
    metrics.update(patient_level_metrics(patient_ids, labels, probs, threshold=threshold))
    return metrics


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def fit_probe(
    head: nn.Module,
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    *,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler: str = "cosine",
    min_lr: float = 1e-6,
    monitor: str = "accuracy",
    patience: int = 20,
    use_class_weights: bool = True,
    seed: int = 42,
    on_epoch=None,
) -> Tuple[Dict[str, Any], list]:
    """Train ``head`` on cached features; return ``(best, history)``.

    ``best`` carries the best epoch's ``state_dict`` (a deep CPU copy, so later
    epochs cannot mutate it), its epoch index, the monitored value and the full
    validation metrics at that epoch -- i.e. everything Stage F2 needs to write a
    checkpoint that behaves like the Swin runs' ``best.pt``.

    ``on_epoch(epoch, row)`` is called once per epoch with the CSV row, so the
    caller owns logging (console / CSV / TensorBoard) and this stays pure.
    """
    head = head.to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    xtr = torch.from_numpy(train["x"]).to(device)
    ytr = torch.from_numpy(np.asarray(train["label"], dtype=np.int64)).to(device)
    xva = torch.from_numpy(val["x"]).to(device)
    y_val = np.asarray(val["label"], dtype=int)
    pid_val = np.asarray(val["patient_id"])

    weights = class_weights_from(train["label"]).to(device) if use_class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = None
    if str(scheduler).lower() == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1),
                                                           eta_min=min_lr)

    n = xtr.shape[0]
    best: Dict[str, Any] = {"epoch": None, "monitor": monitor, "value": -math.inf,
                            "state_dict": None, "val_metrics": {}}
    history: list = []
    no_improve = 0

    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, generator=generator).to(device)
        total, batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(xtr[idx]), ytr[idx])
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        lr_now = optimizer.param_groups[0]["lr"]
        if sched is not None:
            sched.step()

        probs = predict(head, xva)
        val_metrics = score(y_val, probs, pid_val)
        if monitor not in val_metrics:
            raise KeyError(
                f"Monitored metric '{monitor}' not in val metrics {sorted(val_metrics)}."
            )

        row = {"epoch": epoch, "lr": lr_now, "train_ce": total / max(batches, 1)}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        if on_epoch is not None:
            on_epoch(epoch, row)

        current = float(val_metrics[monitor])
        if current > best["value"]:
            best.update({
                "epoch": epoch,
                "value": current,
                "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "val_metrics": dict(val_metrics),
            })
            no_improve = 0
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                best["stopped_early_at"] = epoch
                break

    if best["state_dict"] is not None:
        head.load_state_dict(best["state_dict"])
    return best, history


def guard_probe_health(best: Dict[str, Any], train_labels: np.ndarray,
                       warn) -> None:
    """Raise the two banners that mean 'this number is not what it looks like'.

    * a probe pinned at the majority-class rate has not learned anything, and on a
      ~70%-malignant fold that still *reads* as a plausible accuracy;
    * a non-finite monitored value means the fit diverged.
    """
    value = float(best.get("value", float("nan")))
    if not math.isfinite(value):
        warn("The probe's monitored metric is not finite",
             f"best {best.get('monitor')} = {value}",
             "the fit diverged; do not report this fold.")
        return
    counts = np.bincount(np.asarray(train_labels, dtype=int), minlength=2)
    majority = float(counts.max() / max(counts.sum(), 1))
    if value <= majority + 1e-6:
        warn("The probe did not beat the majority-class rate",
             f"best val {best.get('monitor')} = {value:.4f} vs train majority rate {majority:.4f}",
             "the head is predicting one class; check the features, not the metric.")
