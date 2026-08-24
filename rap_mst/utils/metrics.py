"""Classification metrics for the binary benign/malignant task.

Implemented with NumPy + scikit-learn only so metric computation never depends on
the training loop. Patient-level aggregation lives here too because the research
protocol reports both image-level and patient-level performance.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
    probs: Sequence[float] | None = None,
) -> Dict[str, float]:
    """Compute standard binary-classification metrics.

    Parameters
    ----------
    labels, preds:
        Ground-truth and predicted class indices (0 = benign, 1 = malignant).
    probs:
        Optional probability of the positive (malignant) class; enables AUC.
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

    # Specificity / sensitivity from the confusion matrix (fixed label order).
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0

    if probs is not None and len(np.unique(labels)) > 1:
        metrics["auc"] = float(roc_auc_score(labels, np.asarray(probs)))
    else:
        metrics["auc"] = float("nan")

    return metrics


def candidate_thresholds(probs: np.ndarray) -> np.ndarray:
    """Midpoints between consecutive unique probs + a dense grid.

    Midpoints are the only points where the accuracy step function can change, so
    they give the exact optimum; the dense grid guarantees 0.5 and evenly spaced
    reference points are always evaluated.
    """
    probs = np.asarray(probs, dtype=float)
    uniq = np.unique(probs)
    mids = (uniq[:-1] + uniq[1:]) / 2.0 if uniq.size > 1 else np.array([0.5])
    grid = np.linspace(0.01, 0.99, 197)  # step 0.005
    cand = np.unique(np.concatenate([mids, grid, [0.5]]))
    return cand[(cand > 0.0) & (cand < 1.0)]


def accuracy_at(labels: np.ndarray, probs: np.ndarray, thr: float) -> float:
    return float(((np.asarray(probs) >= thr).astype(int) == np.asarray(labels)).mean())


def best_accuracy_threshold(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Threshold maximising accuracy; ties broken toward the maximal-plateau centre.

    A robust generalising cut sits in the *middle* of the widest interval that
    achieves the best accuracy, not at its noisy edge -- so among all thresholds
    that hit the maximum we take the median. Also returns Youden's J threshold for
    context. Shared by ``scripts/threshold_calibration.py`` (where it originated)
    and by the retrieval gate's pooled out-of-fold calibration.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=float)
    cand = candidate_thresholds(probs)
    accs = np.array([accuracy_at(labels, probs, t) for t in cand])
    best = accs.max()
    winners = cand[accs >= best - 1e-12]
    thr = float(np.median(winners))

    pos = labels == 1
    neg = ~pos
    if int(pos.sum()) and int(neg.sum()):
        j = np.array([(probs[pos] >= t).mean() + (probs[neg] < t).mean() - 1.0 for t in cand])
        youden_thr = float(np.median(cand[j >= j.max() - 1e-12]))
    else:
        youden_thr = 0.5

    return {
        "threshold": thr,
        "val_accuracy": float(best),
        "acc_at_0.5": accuracy_at(labels, probs, 0.5),
        "plateau_lo": float(winners.min()),
        "plateau_hi": float(winners.max()),
        "youden_threshold": youden_thr,
    }


def patient_level_metrics(
    patient_ids: Sequence[str],
    labels: Sequence[int],
    probs: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Aggregate image predictions to patient predictions, then score.

    A patient's malignancy probability is the mean over their images (mean-pooling
    across magnifications, consistent with the unified single-model design). This
    is the patient-level recognition rate used throughout the BreaKHis literature.
    """
    patient_ids = np.asarray(patient_ids)
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    uniq = np.unique(patient_ids)
    p_labels: List[int] = []
    p_preds: List[int] = []
    p_probs: List[float] = []
    for pid in uniq:
        mask = patient_ids == pid
        mean_prob = float(probs[mask].mean())
        p_probs.append(mean_prob)
        p_preds.append(int(mean_prob >= threshold))
        p_labels.append(int(labels[mask][0]))  # patient label is constant

    out = compute_metrics(p_labels, p_preds, p_probs)
    return {f"patient_{k}": v for k, v in out.items()}
