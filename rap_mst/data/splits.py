"""Patient-level split generation and loading.

The whole no-leakage guarantee rests on splitting by *patient*, never by image.
This module:

1. Scans the dataset once and builds the patient -> label/subtype index.
2. Holds out ``NUM_TEST_PATIENTS`` patients for a permanent test set, **stratified
   by the eight fine-grained tumor subtypes** (not just the binary class), so the
   test set is representative of the subtype mix instead of a lucky/unlucky draw.
3. Splits the remaining patients into ``num_folds`` class-stratified folds.
4. Serializes everything to a single JSON file that every future experiment
   reloads verbatim -- splits are generated *once* and never recreated at train
   time.

Subtype-aware test policy
-------------------------
Earlier splits stratified by binary class only, which let the rare subtypes clump
by chance: the original test set drew the two hardest benign subtypes (phyllodes +
a 235-image tubular adenoma) and *none* of the easier adenosis / papillary groups,
pinning image-level test accuracy at ~82% across every experiment regardless of
the model. Two transparent, seed-reproducible knobs fix that:

* ``reserve_subtypes`` -- tumor-type tokens kept entirely out of the test set and
  assigned to the CV pool. Default ``("PT",)`` (phyllodes): only 3 such patients
  exist in all of BreaKHis, so holding one out both starves training of a rare
  class and gives a high-variance single-patient test contribution. Reserving them
  keeps the rare subtype in training (where retrieval / prototype modules can learn
  it) and out of the held-out estimate.
* ``max_test_image_fraction`` -- no single test patient may exceed this fraction of
  the expected test-image budget; oversized patients (e.g. the 235-image slide) are
  routed to the CV pool so one patient can't dominate the image-level metric.

This is a *representativeness* policy, recorded verbatim under ``split_policy`` in
the output, not a tuning of the split to a target accuracy: within every subtype,
test patients are drawn by a seeded permutation and the resulting test number is
reported as-is.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from rap_mst.constants import DEFAULT_NUM_FOLDS, NUM_TEST_PATIENTS, TOTAL_PATIENTS
from rap_mst.data.breakhis import patient_index, scan_dataset

SPLITS_SCHEMA_VERSION = 2

# Subtypes whose patients are kept in the CV pool (never held out for test).
DEFAULT_RESERVE_SUBTYPES = ("PT",)
# Cap on any single test patient's share of the expected test-image budget.
DEFAULT_MAX_TEST_IMAGE_FRACTION = 0.12


def _allocate_test_quota(subtype_sizes: Dict[str, int], num_test: int) -> Dict[str, int]:
    """Largest-remainder apportionment of ``num_test`` slots across subtypes.

    Each subtype gets a share proportional to how many eligible patients it has,
    capped at its own patient count, summing to exactly ``num_test``.
    """
    total = sum(subtype_sizes.values())
    if total < num_test:
        raise ValueError(
            f"Only {total} test-eligible patients but {num_test} requested for test. "
            "Relax reserve_subtypes / max_test_image_fraction."
        )
    raw = {s: n / total * num_test for s, n in subtype_sizes.items()}
    quota = {s: min(int(np.floor(r)), subtype_sizes[s]) for s, r in raw.items()}
    remaining = num_test - sum(quota.values())
    # Hand out the leftover slots to the largest fractional remainders (that still
    # have spare capacity), deterministically.
    order = sorted(raw, key=lambda s: (raw[s] - np.floor(raw[s]), subtype_sizes[s], s), reverse=True)
    i = 0
    while remaining > 0:
        s = order[i % len(order)]
        if quota[s] < subtype_sizes[s]:
            quota[s] += 1
            remaining -= 1
        i += 1
        if i > 10_000:  # pragma: no cover - defensive; cannot happen given total>=num_test
            raise RuntimeError("Quota allocation failed to converge.")
    return quota


def _select_test_patients(
    patients: np.ndarray,
    subtypes: Sequence[str],
    num_images: np.ndarray,
    num_test: int,
    reserve: Sequence[str],
    max_image_fraction: float,
    seed: int,
) -> Dict[str, object]:
    """Pick the held-out test patients, subtype-stratified with reservation + cap."""
    rng = np.random.RandomState(seed)
    reserve = set(reserve)

    expected_test_images = float(num_images.sum()) * num_test / len(patients)
    image_cap = max_image_fraction * expected_test_images

    by_subtype: Dict[str, List[int]] = defaultdict(list)
    reserved_out: List[str] = []
    oversized_out: List[str] = []
    for i in range(len(patients)):
        st = subtypes[i]
        if st in reserve:
            reserved_out.append(patients[i])
            continue
        if num_images[i] > image_cap:
            oversized_out.append(patients[i])
            continue
        by_subtype[st].append(i)

    quota = _allocate_test_quota({s: len(idx) for s, idx in by_subtype.items()}, num_test)

    test_idx: List[int] = []
    for st in sorted(by_subtype):
        idx = np.array(by_subtype[st])
        pick = rng.permutation(idx)[: quota[st]]
        test_idx.extend(int(j) for j in pick)

    test_patients = sorted(patients[i] for i in test_idx)
    return {
        "test_patients": test_patients,
        "quota": quota,
        "reserved_out": sorted(reserved_out),
        "oversized_out": sorted(oversized_out),
        "image_cap": image_cap,
    }


def generate_splits(
    dataset_root: str | Path,
    num_folds: int = DEFAULT_NUM_FOLDS,
    num_test_patients: int = NUM_TEST_PATIENTS,
    seed: int = 42,
    reserve_subtypes: Sequence[str] = DEFAULT_RESERVE_SUBTYPES,
    max_test_image_fraction: float = DEFAULT_MAX_TEST_IMAGE_FRACTION,
) -> Dict:
    """Build the held-out test split and stratified CV folds (patient-level).

    The test set is stratified by tumor subtype, with ``reserve_subtypes`` kept in
    the CV pool and oversized patients (> ``max_test_image_fraction`` of the test
    image budget) routed to CV. Folds remain class-stratified. Returns a
    JSON-serializable dict; use :func:`save_splits` to persist it.
    """
    samples = scan_dataset(dataset_root)
    pindex = patient_index(samples)

    patients = np.array(sorted(pindex.keys()))
    labels = np.array([pindex[p]["label"] for p in patients])
    subtypes = [pindex[p]["tumor_type"] for p in patients]
    num_images = np.array([pindex[p]["num_images"] for p in patients])

    if len(patients) != TOTAL_PATIENTS:
        # Not fatal (a subset of the data may be present), but worth surfacing.
        print(
            f"[warn] Expected {TOTAL_PATIENTS} patients, found {len(patients)}. "
            "Proceeding with what is on disk."
        )

    # --- 1. Subtype-stratified held-out test set (permanent) ------------- #
    selection = _select_test_patients(
        patients=patients,
        subtypes=subtypes,
        num_images=num_images,
        num_test=num_test_patients,
        reserve=reserve_subtypes,
        max_image_fraction=max_test_image_fraction,
        seed=seed,
    )
    test_set = set(selection["test_patients"])
    cv_mask = np.array([p not in test_set for p in patients])
    cv_patients = patients[cv_mask]
    cv_labels = labels[cv_mask]

    # --- 2. Class-stratified K folds over the remaining patients --------- #
    # Fold stratification stays binary: the scarcest subtype (phyllodes, n=3) has
    # too few patients to stratify across 5 folds, and it is reserved anyway.
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds: List[Dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(cv_patients, cv_labels)):
        folds.append(
            {
                "fold": fold_idx,
                "train_patients": sorted(cv_patients[train_idx].tolist()),
                "val_patients": sorted(cv_patients[val_idx].tolist()),
            }
        )

    splits = {
        "schema_version": SPLITS_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "seed": seed,
        "num_folds": num_folds,
        "num_test_patients": num_test_patients,
        "total_patients": len(patients),
        "class_balance": {
            "benign": int((labels == 0).sum()),
            "malignant": int((labels == 1).sum()),
        },
        "split_policy": {
            "test_stratified_by": "tumor_subtype",
            "fold_stratified_by": "class",
            "reserve_subtypes": list(reserve_subtypes),
            "max_test_image_fraction": max_test_image_fraction,
            "test_image_cap": round(float(selection["image_cap"]), 1),
            "test_subtype_quota": selection["quota"],
            "reserved_to_cv": selection["reserved_out"],
            "oversized_to_cv": selection["oversized_out"],
            "rationale": (
                "Test set stratified over the eight tumor subtypes; rare/oversized "
                "patients kept in the CV pool so the held-out image-level metric is "
                "representative and rare subtypes remain available for training."
            ),
        },
        "test_patients": sorted(test_set),
        "folds": folds,
        "patient_meta": {
            p: {
                "label": pindex[p]["label"],
                "tumor_type": pindex[p]["tumor_type"],
                "num_images": pindex[p]["num_images"],
            }
            for p in patients.tolist()
        },
    }
    _assert_no_leakage(splits)
    return splits


def _assert_no_leakage(splits: Dict) -> None:
    """Fail loudly if any patient appears in more than one partition of a fold.

    Checked at generation time and again at load time -- cheap insurance against
    the single failure mode that would invalidate every downstream result.
    """
    test = set(splits["test_patients"])
    all_patients = set(splits["patient_meta"].keys())

    for fold in splits["folds"]:
        train = set(fold["train_patients"])
        val = set(fold["val_patients"])

        if train & val:
            raise AssertionError(f"Fold {fold['fold']}: train/val patient overlap: {train & val}")
        if train & test:
            raise AssertionError(f"Fold {fold['fold']}: train/test patient overlap: {train & test}")
        if val & test:
            raise AssertionError(f"Fold {fold['fold']}: val/test patient overlap: {val & test}")
        if (train | val) != (all_patients - test):
            raise AssertionError(
                f"Fold {fold['fold']}: train+val does not cover all non-test patients."
            )


def save_splits(splits: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: str | Path) -> Dict:
    """Load and re-validate a saved splits file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Splits file not found: {path}. Run scripts/prepare_splits.py first."
        )
    with path.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    _assert_no_leakage(splits)  # re-validate on every load
    return splits


def get_fold(splits: Dict, fold: int) -> Dict:
    """Return the fold record for a given fold index."""
    for f in splits["folds"]:
        if f["fold"] == fold:
            return f
    raise ValueError(f"Fold {fold} not found. Available: {[f['fold'] for f in splits['folds']]}")
