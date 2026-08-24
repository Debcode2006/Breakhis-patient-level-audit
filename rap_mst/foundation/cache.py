"""The frozen-feature cache (Stage F1's output, Stages F2/F3's input).

ONE ``.npz`` per encoder holding every BreaKHis image encoded once:

    key         [N, D] float32   pooled foundation-model feature (raw, not normed)
    label       [N]    int64     0 = benign, 1 = malignant
    patient_id  [N]    str       SOB_<cls>_<type>-<year>-<slide>
    subtype     [N]    str       the eight tumour subtypes (diagnostics)
    magnification [N]  int64     40 / 100 / 200 / 400
    image_path  [N]    str
    meta               json      encoder provenance + transform description

Why one file for all 82 patients is not leakage
-----------------------------------------------
The encoder is frozen, and pretrained on TCGA/PAIP -- it never sees a BreaKHis
label and no gradient flows back into it. What *is* fitted from labels (the
standardiser, the head) is fitted on a fold's TRAIN rows only, and
:meth:`FeatureCache.rows_for` + the probe's assertions are what enforce that. The
cache is therefore the analogue of the dataset itself, not of a memory bank.

Provenance is checked on load, in the same spirit as ``MemoryBank``: a cache built
with a different encoder than the config asks for **raises** rather than quietly
producing a number for the wrong row of the paper's table.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


class FeatureCache:
    """Frozen features for every image, addressable by patient."""

    def __init__(
        self,
        key: np.ndarray,
        label: np.ndarray,
        patient_id: np.ndarray,
        subtype: np.ndarray,
        magnification: np.ndarray,
        image_path: np.ndarray,
        meta: Dict[str, Any] | None = None,
    ) -> None:
        self.key = np.asarray(key, dtype=np.float32)
        self.label = np.asarray(label, dtype=np.int64)
        self.patient_id = np.asarray(patient_id).astype(str)
        self.subtype = np.asarray(subtype).astype(str)
        self.magnification = np.asarray(magnification, dtype=np.int64)
        self.image_path = np.asarray(image_path).astype(str)
        self.meta: Dict[str, Any] = dict(meta or {})

        n = self.key.shape[0]
        for name in ("label", "patient_id", "subtype", "magnification", "image_path"):
            got = getattr(self, name).shape[0]
            if got != n:
                raise ValueError(f"FeatureCache column '{name}' has {got} rows, expected {n}.")

    # ------------------------------------------------------------------ #
    # Shape / introspection
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self.key.shape[0])

    @property
    def dim(self) -> int:
        return int(self.key.shape[1])

    def unique_patients(self) -> List[str]:
        return sorted(set(self.patient_id.tolist()))

    def summary(self) -> Dict[str, Any]:
        norms = np.linalg.norm(self.key, axis=1)
        mags, counts = np.unique(self.magnification, return_counts=True)
        return {
            "rows": len(self),
            "dim": self.dim,
            "patients": len(self.unique_patients()),
            "benign_rows": int((self.label == 0).sum()),
            "malignant_rows": int((self.label == 1).sum()),
            "mag_counts": {int(m): int(c) for m, c in zip(mags, counts)},
            "feature_norm_min": float(norms.min()),
            "feature_norm_max": float(norms.max()),
            "feature_absmax": float(np.abs(self.key).max()),
            "n_nonfinite": int((~np.isfinite(self.key)).sum()),
        }

    # ------------------------------------------------------------------ #
    # Row selection -- the ONLY way a split is materialised
    # ------------------------------------------------------------------ #
    def rows_for(self, patients: Sequence[str], what: str = "split") -> np.ndarray:
        """Indices of every row belonging to ``patients``, in cache order.

        Raises if a requested patient has no rows: a silently-empty split would
        make a fold train on 4/5 of its data and never say so.
        """
        wanted = set(patients)
        mask = np.isin(self.patient_id, list(wanted))
        idx = np.nonzero(mask)[0]
        found = set(self.patient_id[idx].tolist())
        if wanted - found:
            raise AssertionError(
                f"{len(wanted - found)} {what} patient(s) have no rows in the feature "
                f"cache: {sorted(wanted - found)[:5]}. Re-run Stage F1 -- the cache is "
                "stale with respect to the splits file."
            )
        return idx

    def view(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        """A dict of columns restricted to ``idx`` (no copy of the cache object)."""
        return {
            "key": self.key[idx],
            "label": self.label[idx],
            "patient_id": self.patient_id[idx],
            "subtype": self.subtype[idx],
            "magnification": self.magnification[idx],
            "image_path": self.image_path[idx],
        }

    # ------------------------------------------------------------------ #
    # Guards
    # ------------------------------------------------------------------ #
    @staticmethod
    def assert_disjoint(*groups: Iterable[str], names: Sequence[str] = ()) -> None:
        """Assert the given patient groups do not intersect. Never work around it."""
        sets = [set(g) for g in groups]
        labels = list(names) + [f"group{i}" for i in range(len(names), len(sets))]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                overlap = sets[i] & sets[j]
                if overlap:
                    raise AssertionError(
                        f"LEAKAGE: {len(overlap)} patient(s) in both {labels[i]} and "
                        f"{labels[j]}: {sorted(overlap)[:5]}"
                    )

    def assert_encoder(self, encoder: str) -> None:
        """Refuse a cache built with a different encoder than the config asks for."""
        built = str(self.meta.get("encoder", "?"))
        if built != encoder:
            raise SystemExit(
                f"Feature cache was built with encoder '{built}', but the config asks "
                f"for '{encoder}'. Re-run Stage F1 for '{encoder}' rather than mixing "
                "encoders across the table's rows."
            )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            key=self.key,
            label=self.label,
            patient_id=self.patient_id,
            subtype=self.subtype,
            magnification=self.magnification,
            image_path=self.image_path,
            meta=np.array(json.dumps(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeatureCache":
        path = Path(path)
        if not path.exists():
            raise SystemExit(
                f"Feature cache not found: {path}\n"
                "Run Stage F1 first: python scripts/extract_foundation_features.py"
            )
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            return cls(
                key=data["key"],
                label=data["label"],
                patient_id=data["patient_id"],
                subtype=data["subtype"],
                magnification=data["magnification"],
                image_path=data["image_path"],
                meta=meta,
            )


# --------------------------------------------------------------------------- #
# Splits -> cache rows (the single place a fold is materialised)
# --------------------------------------------------------------------------- #
def split_views(cache: FeatureCache, splits: Dict[str, Any], fold: int,
                include_test: bool = True) -> Dict[str, Dict[str, Any]]:
    """Materialise fold ``fold``'s train / val (/ test) rows, asserting disjointness.

    Both Stage F2 and Stage F3 go through here, so the leakage guard is one
    implementation rather than two that can drift. The assertion is on the
    *patient* lists (the protocol's unit), before a single feature row is touched.
    """
    from rap_mst.data.splits import get_fold

    fold_rec = get_fold(splits, fold)
    groups = {
        "train": list(fold_rec["train_patients"]),
        "val": list(fold_rec["val_patients"]),
        "test": list(splits["test_patients"]),
    }
    FeatureCache.assert_disjoint(*groups.values(), names=list(groups))

    wanted = ["train", "val"] + (["test"] if include_test else [])
    out: Dict[str, Dict[str, Any]] = {}
    for name in wanted:
        idx = cache.rows_for(groups[name], what=name)
        view = cache.view(idx)
        view["patients"] = groups[name]
        view["rows"] = int(idx.size)
        out[name] = view
    return out


def write_predictions_csv(path: str | Path, view: Dict[str, Any], probs: np.ndarray,
                          threshold: float = 0.5) -> Path:
    """Per-image predictions in the EXACT column layout ``scripts/test.py`` writes.

    Byte-compatible on purpose: ``diagnose_folds.py`` and ``threshold_calibration.py``
    read this file positionally-by-name, so the foundation baseline drops into those
    analyses with no special case.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = (np.asarray(probs) >= threshold).astype(int)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "patient_id", "magnification", "label", "pred",
                         "prob_malignant"])
        for i in range(len(probs)):
            writer.writerow([
                view["image_path"][i], view["patient_id"][i], int(view["magnification"][i]),
                int(view["label"][i]), int(preds[i]), f"{float(probs[i]):.6f}",
            ])
    return path
