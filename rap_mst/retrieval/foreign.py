"""Foreign-encoder keys: present an *external* frozen feature cache as a fold cache.

Why this exists
---------------
``docs/results/retrieval_key_ablation.md`` §6 isolated the one quantity that responds to
anything in the whole retrieval study: **error-decorrelation from the parametric
head**, and it responds only to *encoder identity*. The control that showed it
(``--cross-encoder exp1``) swapped in keys from a second **Swin** run, which the
ablation script can read because both caches come from its own Stage-1 encode.

A pathology foundation model is the same experiment with a much larger dose of
encoder diversity, but its features live in a different artefact: the Stage-F1
``FeatureCache`` (``rap_mst/foundation/cache.py``) -- one file, all 82 patients,
768-d, ordered by the global dataset scan rather than by fold. This module is the
adapter between the two, and nothing else:

    bank / query keys  <- CTransPath (encoder A, foreign)
    p_param, labels    <- exp3n's own head (encoder B, unchanged)

Everything downstream -- ``MemoryBank``, the two-level ranking, the per-patient
cap, the softmax vote, ``FusionGate`` -- is the **production** class, untouched.

The alignment, which is the only place a bug could hide
-------------------------------------------------------
The ablation cache stores ``patient_id`` / ``label`` / ``mag_index`` per row but
**not** ``image_path``, so a silent misalignment between the two feature spaces
would produce a beautifully plausible null. Three defences, in order:

1. :func:`protocol_rows` rebuilds a fold/split's row order by calling the *same
   production objects the cache was written from* (``scan_dataset`` ->
   ``BreaKHisDataset`` under the fold's patient list). It does not re-implement
   the ordering; it re-executes it.
2. :meth:`ForeignKeyCache.assert_aligned_with` checks that reconstruction against
   the reference cache's own columns, element-wise, and **raises**.
3. The join into the foundation cache is then by ``image_path`` -- an exact
   string key, every row required to resolve.

Leakage is unchanged and unweakened: the foundation cache holds every patient,
but a fold's bank is filled only from that fold's TRAIN rows, ``assert_disjoint``
still runs, and the query still blocks its own patient. The encoder never saw a
BreaKHis label (``rap_mst/foundation/cache.py`` explains why the single file is
the analogue of the dataset, not of a memory bank).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from rap_mst.constants import MAG_TO_IDX
from rap_mst.data.breakhis import Sample, scan_dataset, subtype_from_patient_id
from rap_mst.data.dataset import BreaKHisDataset
from rap_mst.data.splits import get_fold
from rap_mst.foundation.cache import FeatureCache
from rap_mst.retrieval.keys import extract_key_from_pooled

#: Which patient list each split name draws from. ``bank`` is the fold's TRAIN
#: patients under eval transforms -- exactly ``BreaKHisDataModule.setup_bank``.
SPLIT_PATIENTS = {"bank": "train_patients", "val": "val_patients"}


# --------------------------------------------------------------------------- #
# Row order, re-executed rather than re-implemented
# --------------------------------------------------------------------------- #
def protocol_rows(samples: Sequence[Sample], splits: Dict, fold: int,
                  split: str, magnifications: Optional[Sequence[int]] = None) -> List[Sample]:
    """The ordered :class:`Sample` list a fold/split's cache rows were written from.

    Goes through :class:`BreaKHisDataset` itself, so the filtering predicate and
    the ordering are the production ones by construction. Both loaders that wrote
    the caches use ``shuffle=False`` and ``drop_last=False``
    (``BreaKHisDataModule.bank_dataloader`` / ``val_dataloader``), so dataset order
    *is* cache order.
    """
    if split not in SPLIT_PATIENTS:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_PATIENTS)}.")
    patients = get_fold(splits, fold)[SPLIT_PATIENTS[split]]
    ds = BreaKHisDataset(
        samples=samples,
        patient_ids=patients,
        magnifications=magnifications,
        transform=None,          # metadata only; no image is ever opened here
    )
    return list(ds.samples)


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #
class ForeignKeyCache:
    """A :class:`FeatureCache` viewed as one fold's ``(bank, val)`` key source.

    Duck-typed against the ablation harness's ``FoldCache``: it answers
    ``keys(split, spec)``, ``col(split, name)`` and carries ``fold`` / ``device`` /
    ``clf_dir`` / ``meta``, which is the whole interface ``build_bank``,
    ``query_val`` and ``route_all_diagnostics`` consume.

    The key spec language is deliberately restricted. A foundation encoder emits
    one pooled vector, so there is no FPN pyramid and no SupCon projection to key
    on; ``features`` is the only valid part. ``key_transform`` is unaffected -- it
    is fitted by the bank on its own rows and works on any key space.
    """

    #: Reported as the encoder identity in results rows / bank provenance.
    def __init__(self, cache: FeatureCache, samples: Sequence[Sample], splits: Dict,
                 fold: int, device: torch.device,
                 magnifications: Optional[Sequence[int]] = None,
                 encoder: str = "?") -> None:
        self.fold = int(fold)
        self.device = device
        self.encoder = str(encoder)
        self.clf_dir = None      # no linear head in this space -> 'drop_dirs' is undefined
        self.meta = {"experiment": encoder, "encoder": encoder, "fold": self.fold,
                     "source": "foundation FeatureCache", **{
                         k: cache.meta.get(k) for k in ("hub_id", "feature_dim", "transform")
                         if k in cache.meta}}

        row_of = {p: i for i, p in enumerate(cache.image_path.tolist())}
        self._rows: Dict[str, np.ndarray] = {}
        self._samples: Dict[str, List[Sample]] = {}
        for split in SPLIT_PATIENTS:
            rows = protocol_rows(samples, splits, fold, split, magnifications)
            missing = [s.image_path for s in rows if s.image_path not in row_of]
            if missing:
                raise AssertionError(
                    f"fold {fold}/{split}: {len(missing)} image(s) have no row in the "
                    f"foundation feature cache, e.g. {missing[:3]}. The cache is stale "
                    "with respect to the splits/dataset -- re-run Stage F1."
                )
            self._samples[split] = rows
            self._rows[split] = np.asarray([row_of[s.image_path] for s in rows], dtype=np.int64)

        self._cache = cache
        self._cols: Dict[str, Dict[str, np.ndarray]] = {}
        for split, idx in self._rows.items():
            self._cols[split] = {
                "label": cache.label[idx].astype(np.int64),
                "patient_id": cache.patient_id[idx],
                "subtype": np.asarray([subtype_from_patient_id(p) for p in cache.patient_id[idx]]),
                "mag_index": np.asarray([MAG_TO_IDX[int(m)] for m in cache.magnification[idx]],
                                        dtype=np.int64),
                "image_path": cache.image_path[idx],
            }
            # The cache's own subtype column must agree with the one derived from
            # the patient id -- the bank stores the derived one everywhere else.
            if not np.array_equal(self._cols[split]["subtype"], cache.subtype[idx]):
                raise AssertionError(
                    f"fold {fold}/{split}: foundation cache 'subtype' column disagrees with "
                    "subtype_from_patient_id. One of the two parsers has drifted."
                )

    # ------------------------------------------------------------------ #
    def col(self, split: str, name: str) -> np.ndarray:
        return self._cols[split][name]

    def keys(self, split: str, spec: str) -> torch.Tensor:
        """``[N, 768]`` foreign keys for ``spec`` (``features`` only, see the class doc)."""
        vec = torch.as_tensor(self._cache.key[self._rows[split]], dtype=torch.float32,
                              device=self.device)
        try:
            return extract_key_from_pooled(spec, {"features": vec, "embeddings": vec}, {})
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"Retrieval key {spec!r} is not available on a foundation encoder: it emits a "
                f"single pooled {vec.shape[1]}-d vector, so there is no FPN pyramid and no "
                "SupCon projection. Use 'features' (optionally with a key_transform)."
            ) from exc

    def p_param(self, split: str) -> np.ndarray:  # pragma: no cover - never the p_param source
        raise NotImplementedError(
            "A foreign key cache never supplies p_param: the parametric probability must "
            "come from the base encoder's own head, which is the entire point of the "
            "cross-encoder design."
        )

    # ------------------------------------------------------------------ #
    def assert_aligned_with(self, reference, splits: Sequence[str] = ("bank", "val")) -> Dict:
        """Row-for-row agreement with the base encoder's fold cache. Raises on any drift.

        A misalignment here silently pairs one image's ``p_param`` with another
        image's neighbourhood and yields a plausible, wrong null -- so this checks
        every metadata column the two caches share, element-wise, and reports what
        it verified so the run log carries the evidence.
        """
        report: Dict[str, Dict] = {}
        for split in splits:
            for name in ("patient_id", "label", "mag_index"):
                mine, theirs = self.col(split, name), reference.col(split, name)
                if mine.shape != theirs.shape:
                    raise AssertionError(
                        f"fold {self.fold}/{split}: foreign cache has {mine.shape[0]} rows, "
                        f"base cache has {theirs.shape[0]}. Re-cache both from the same splits file."
                    )
                bad = int((np.asarray(mine) != np.asarray(theirs)).sum())
                if bad:
                    first = int(np.nonzero(np.asarray(mine) != np.asarray(theirs))[0][0])
                    raise AssertionError(
                        f"fold {self.fold}/{split}: '{name}' differs in {bad} row(s), first at "
                        f"index {first} ({mine[first]!r} vs {theirs[first]!r}). The two feature "
                        "spaces are NOT row-aligned; every downstream number would be garbage."
                    )
            report[split] = {
                "rows": int(self.col(split, "label").shape[0]),
                "patients": int(len(set(self.col(split, "patient_id").tolist()))),
                "columns_verified": ["patient_id", "label", "mag_index"],
                "joined_on": "image_path",
            }
        return report


# --------------------------------------------------------------------------- #
def load_foreign_caches(cache_path: str | Path, cfg, splits: Dict, folds: Sequence[int],
                        device: torch.device, encoder: str) -> Dict[int, ForeignKeyCache]:
    """One :class:`ForeignKeyCache` per fold from a single Stage-F1 cache file."""
    cache = FeatureCache.load(cache_path)
    cache.assert_encoder(encoder)
    samples = scan_dataset(cfg.data.dataset_root)
    mags = getattr(cfg.data, "magnifications", None)
    mags = list(mags) if mags else None
    return {f: ForeignKeyCache(cache, samples, splits, f, device, mags, encoder) for f in folds}
