"""DataModule: builds train/val/test dataloaders for a given fold.

Encapsulates the full data path so `train.py` / `test.py` never touch split
internals. It scans the dataset exactly once and reuses the parsed samples for
every split, which matters on spinning disks with ~8k files.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from rap_mst.constants import TOTAL_PATIENTS
from rap_mst.data.breakhis import scan_dataset
from rap_mst.data.collate import breakhis_collate
from rap_mst.data.dataset import BreaKHisDataset
from rap_mst.data.splits import get_fold, load_splits
from rap_mst.data.transforms import build_transforms
from rap_mst.utils.seed import seed_worker


class BreaKHisDataModule:
    """Owns datasets and dataloaders for one cross-validation fold."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.splits = load_splits(cfg.data.splits_path)

        # Optional magnification whitelist; None => all four (unified model).
        mags = getattr(cfg.data, "magnifications", None)
        self.magnifications = list(mags) if mags else None

        # Scan the filesystem once; all splits reuse these parsed samples.
        self._samples = scan_dataset(cfg.data.dataset_root)

        self.train_set: Optional[BreaKHisDataset] = None
        self.val_set: Optional[BreaKHisDataset] = None
        self.test_set: Optional[BreaKHisDataset] = None
        self.bank_set: Optional[BreaKHisDataset] = None  # retrieval bank (eval transforms)
        self.all_set: Optional[BreaKHisDataset] = None   # every protocol patient (eval transforms)

        # Set by setup_fold(); exposed for the dataset report / leakage guard.
        self.fold: Optional[int] = None
        self.fold_rec: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def setup_fold(self, fold: int) -> None:
        """Build train/val datasets for the requested fold."""
        fold_rec = get_fold(self.splits, fold)
        self.fold = fold
        self.fold_rec = fold_rec
        self.train_set = BreaKHisDataset(
            samples=self._samples,
            patient_ids=fold_rec["train_patients"],
            magnifications=self.magnifications,
            transform=build_transforms(self.cfg, train=True),
            two_view=bool(getattr(self.cfg.data, "two_view", False)),
        )
        self.val_set = BreaKHisDataset(
            samples=self._samples,
            patient_ids=fold_rec["val_patients"],
            magnifications=self.magnifications,
            transform=build_transforms(self.cfg, train=False),
        )

    def setup_bank(self, fold: int) -> None:
        """Deterministic view of a fold's TRAIN patients, for the retrieval bank.

        Deliberately *not* ``train_set``: the memory bank must be encoded with the
        frozen encoder under **eval** transforms (no augmentation), in order, with
        every image present. ``train_dataloader()`` shuffles and drops the last
        partial batch, which would silently omit images from the bank.
        """
        fold_rec = get_fold(self.splits, fold)
        self.fold = fold
        self.fold_rec = fold_rec
        self.bank_set = BreaKHisDataset(
            samples=self._samples,
            patient_ids=fold_rec["train_patients"],
            magnifications=self.magnifications,
            transform=build_transforms(self.cfg, train=False),
        )

    def setup_all(self) -> None:
        """Deterministic view of **every protocol patient**, for frozen encoding.

        Used by the foundation-model baseline's Stage F1, which encodes the whole
        dataset once with a frozen, externally-pretrained encoder
        (``rap_mst/foundation/cache.py`` explains why that leaks nothing).

        The patient list is the union taken from the *splits file* -- not "every
        image under ``dataset_root``" -- so an out-of-protocol image sitting in the
        dataset directory can never enter the cache, and the count is asserted.
        """
        patients = set(self.splits["test_patients"])
        for fold_rec in self.splits["folds"]:
            patients |= set(fold_rec["train_patients"]) | set(fold_rec["val_patients"])
        if len(patients) != TOTAL_PATIENTS:
            raise AssertionError(
                f"Splits file covers {len(patients)} patients, expected {TOTAL_PATIENTS}. "
                "Regenerate with scripts/prepare_splits.py."
            )
        self.all_set = BreaKHisDataset(
            samples=self._samples,
            patient_ids=sorted(patients),
            magnifications=self.magnifications,
            transform=build_transforms(self.cfg, train=False),
        )

    def setup_test(self) -> None:
        """Build the held-out 16-patient test dataset."""
        self.test_set = BreaKHisDataset(
            samples=self._samples,
            patient_ids=self.splits["test_patients"],
            magnifications=self.magnifications,
            transform=build_transforms(self.cfg, train=False),
        )

    # ------------------------------------------------------------------ #
    # Dataloaders
    # ------------------------------------------------------------------ #
    def _loader(self, dataset: BreaKHisDataset, shuffle: bool, sampler=None) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        return DataLoader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=shuffle,  # drop last only when training (stable BN / SupCon)
            collate_fn=breakhis_collate,
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=self.cfg.data.num_workers > 0,
        )

    # ------------------------------------------------------------------ #
    # Split introspection (used by the dataset report / leakage guard)
    # ------------------------------------------------------------------ #
    def train_set_patient_ids(self):
        assert self.fold_rec is not None, "Call setup_fold() first."
        return self.fold_rec["train_patients"]

    def val_set_patient_ids(self):
        assert self.fold_rec is not None, "Call setup_fold() first."
        return self.fold_rec["val_patients"]

    def train_dataloader(self) -> DataLoader:
        assert self.train_set is not None, "Call setup_fold() first."
        sampler = None
        if getattr(self.cfg.data, "balanced_sampler", False):
            sampler = self._make_balanced_sampler(self.train_set)
        return self._loader(self.train_set, shuffle=True, sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        assert self.val_set is not None, "Call setup_fold() first."
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_set is not None, "Call setup_test() first."
        return self._loader(self.test_set, shuffle=False)

    def bank_dataloader(self) -> DataLoader:
        """Every training image of the fold, once, in order, un-augmented."""
        assert self.bank_set is not None, "Call setup_bank(fold) first."
        return self._loader(self.bank_set, shuffle=False)

    def all_dataloader(self) -> DataLoader:
        """Every protocol image, once, in order, un-augmented."""
        assert self.all_set is not None, "Call setup_all() first."
        return self._loader(self.all_set, shuffle=False)

    # ------------------------------------------------------------------ #
    # Class imbalance handling
    # ------------------------------------------------------------------ #
    def _make_balanced_sampler(self, dataset: BreaKHisDataset) -> WeightedRandomSampler:
        """Oversample the minority (benign) class to balance the 24:58 ratio."""
        counts = dataset.class_counts()
        class_weight = {c: 1.0 / n for c, n in counts.items()}
        weights = [class_weight[s.label] for s in dataset.samples]
        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for a weighted CrossEntropy loss."""
        assert self.train_set is not None, "Call setup_fold() first."
        counts = self.train_set.class_counts()
        total = sum(counts.values())
        num_classes = len(counts)
        weights = torch.tensor(
            [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)],
            dtype=torch.float32,
        )
        return weights
