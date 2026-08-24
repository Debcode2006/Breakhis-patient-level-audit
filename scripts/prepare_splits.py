"""Generate the permanent patient-level splits.

Run this ONCE. It scans BreaKHis, holds out 16 patients for the final test set,
builds 5 stratified patient-level CV folds over the remaining 66, verifies there
is no patient leakage, and writes everything to a single JSON that every future
experiment reloads verbatim.

Example
-------
    python scripts/prepare_splits.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the `rap_mst` package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter  # noqa: E402

from rap_mst.constants import DEFAULT_NUM_FOLDS, NUM_TEST_PATIENTS  # noqa: E402
from rap_mst.data.splits import (  # noqa: E402
    DEFAULT_MAX_TEST_IMAGE_FRACTION,
    DEFAULT_RESERVE_SUBTYPES,
    generate_splits,
    save_splits,
)
from rap_mst.utils.config import load_config  # noqa: E402


def _subtype(pid: str) -> str:
    """B_PT / M_DC style key from a patient id (SOB_<cls>_<type>-...)."""
    parts = pid.split("_")
    return f"{parts[1]}_{parts[2].split('-')[0]}"


def _compose(patient_ids, meta) -> str:
    """Human-readable subtype composition (patients + images) for a split."""
    by = Counter(_subtype(p) for p in patient_ids)
    imgs = Counter()
    for p in patient_ids:
        imgs[_subtype(p)] += meta[p]["num_images"]
    return ", ".join(f"{s}:{by[s]}p/{imgs[s]}i" for s in sorted(by))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate patient-level BreaKHis splits.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset-root", default=None, help="Override dataset root")
    parser.add_argument("--out", default=None, help="Override output splits path")
    parser.add_argument("--num-folds", type=int, default=DEFAULT_NUM_FOLDS)
    parser.add_argument("--num-test-patients", type=int, default=NUM_TEST_PATIENTS)
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument(
        "--reserve-subtypes", nargs="*", default=list(DEFAULT_RESERVE_SUBTYPES),
        help="Tumor-type tokens kept out of the test set (assigned to CV pool). "
             f"Default: {list(DEFAULT_RESERVE_SUBTYPES)}. Pass with no values to reserve none.",
    )
    parser.add_argument(
        "--max-test-image-fraction", type=float, default=DEFAULT_MAX_TEST_IMAGE_FRACTION,
        help="Cap on any single test patient's share of the expected test-image budget.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing splits file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = args.dataset_root or cfg.data.dataset_root
    out_path = Path(args.out or cfg.data.splits_path)
    seed = args.seed if args.seed is not None else cfg.seed

    if out_path.exists() and not args.force:
        raise SystemExit(
            f"Splits already exist at {out_path}. Splits are meant to be generated once. "
            "Pass --force to regenerate (this invalidates prior experiments)."
        )

    print(f"Scanning dataset at: {dataset_root}")
    splits = generate_splits(
        dataset_root=dataset_root,
        num_folds=args.num_folds,
        num_test_patients=args.num_test_patients,
        seed=seed,
        reserve_subtypes=tuple(args.reserve_subtypes),
        max_test_image_fraction=args.max_test_image_fraction,
    )
    save_splits(splits, out_path)

    # Report.
    cb = splits["class_balance"]
    meta = splits["patient_meta"]
    pol = splits["split_policy"]
    print("\n=== Splits generated (no patient leakage verified) ===")
    print(f"  Total patients      : {splits['total_patients']} "
          f"(benign={cb['benign']}, malignant={cb['malignant']})")
    print(f"  Test stratified by  : {pol['test_stratified_by']} | folds by {pol['fold_stratified_by']}")
    print(f"  Reserved -> CV      : {pol['reserve_subtypes']} "
          f"(patients: {', '.join(pol['reserved_to_cv']) or 'none'})")
    print(f"  Oversized -> CV     : {', '.join(pol['oversized_to_cv']) or 'none'} "
          f"(image cap={pol['test_image_cap']:.0f})")
    print(f"  Held-out test       : {len(splits['test_patients'])} patients")
    print(f"    subtype mix       : {_compose(splits['test_patients'], meta)}")
    print(f"  CV folds            : {splits['num_folds']}")
    for fold in splits["folds"]:
        print(f"    fold {fold['fold']}: "
              f"train={len(fold['train_patients'])}, val={len(fold['val_patients'])} "
              f"| val mix: {_compose(fold['val_patients'], meta)}")
    print(f"  Seed                : {splits['seed']}")
    print(f"  Saved to            : {out_path.resolve()}")


if __name__ == "__main__":
    main()
