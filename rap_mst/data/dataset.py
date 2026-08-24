"""BreaKHis PyTorch Dataset.

The dataset is constructed from an explicit list of patient ids (produced by the
split loader) so it is impossible to accidentally include a held-out patient.
Each item returns the image tensor plus rich metadata; future retrieval and
prototype modules can consume ``patient_id`` / ``mag_index`` / ``image_path``
without any dataset changes.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from rap_mst.data.breakhis import Sample, scan_dataset


class BreaKHisDataset(Dataset):
    """Image-level dataset restricted to a given set of patients.

    Parameters
    ----------
    samples:
        Pre-scanned samples (pass to avoid re-scanning the filesystem for every
        split). If ``None``, the dataset scans ``dataset_root`` itself.
    dataset_root:
        Root used only when ``samples`` is not provided.
    patient_ids:
        Restrict to these patients. ``None`` keeps every sample (used for
        debugging / stats only -- training always passes an explicit set).
    magnifications:
        Optional whitelist of raw magnification values (e.g. ``[40, 100]``).
        ``None`` keeps all four -- the unified-model default.
    transform:
        Callable applied to the PIL image.
    two_view:
        When ``True``, apply ``transform`` twice to the same source image and
        return a second augmented crop under ``image2``. This yields the two
        correlated views SupCon needs to form a guaranteed positive pair per
        anchor (the [B, 2, D] contrastive setup). Only meaningful with a
        stochastic training transform; leave ``False`` for val/test.
    """

    def __init__(
        self,
        samples: Optional[Sequence[Sample]] = None,
        dataset_root: Optional[str] = None,
        patient_ids: Optional[Sequence[str]] = None,
        magnifications: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
        two_view: bool = False,
    ) -> None:
        if samples is None:
            if dataset_root is None:
                raise ValueError("Provide either `samples` or `dataset_root`.")
            samples = scan_dataset(dataset_root)

        patient_filter = set(patient_ids) if patient_ids is not None else None
        mag_filter = set(magnifications) if magnifications is not None else None

        self.samples: List[Sample] = [
            s
            for s in samples
            if (patient_filter is None or s.patient_id in patient_filter)
            and (mag_filter is None or s.magnification in mag_filter)
        ]
        if not self.samples:
            raise RuntimeError(
                "BreaKHisDataset is empty after filtering. Check patient_ids / magnifications."
            )
        self.transform = transform
        self.two_view = two_view

    def __len__(self) -> int:
        return len(self.samples)

    def class_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for s in self.samples:
            counts[s.label] = counts.get(s.label, 0) + 1
        return counts

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        source = Image.open(s.image_path).convert("RGB")
        image = self.transform(source) if self.transform is not None else source

        item = {
            "image": image,
            "label": torch.tensor(s.label, dtype=torch.long),
            "mag_index": torch.tensor(s.mag_index, dtype=torch.long),
            # Metadata kept as Python types; collated into lists (see collate.py).
            "patient_id": s.patient_id,
            "magnification": s.magnification,
            "image_path": s.image_path,
        }

        # Second correlated view (SupCon two-crop). Re-applying the stochastic
        # transform to the same source produces an independent augmentation; the
        # trainer stacks the two projections into the [B, 2, D] contrastive tensor.
        if self.two_view and self.transform is not None:
            item["image2"] = self.transform(source)

        return item
