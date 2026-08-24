"""Custom collate function.

Tensor fields (image, label, mag_index) are stacked; metadata fields
(patient_id, magnification, image_path) are kept as plain Python lists so they
survive batching and remain available to metrics and future retrieval modules.
"""

from __future__ import annotations

from typing import Dict, List

import torch

_TENSOR_KEYS = ("image", "label", "mag_index")
_LIST_KEYS = ("patient_id", "magnification", "image_path")


def breakhis_collate(batch: List[Dict]) -> Dict:
    out: Dict = {}
    for key in _TENSOR_KEYS:
        out[key] = torch.stack([item[key] for item in batch])
    for key in _LIST_KEYS:
        out[key] = [item[key] for item in batch]
    # Optional SupCon second view (present only when two_view is enabled).
    if "image2" in batch[0]:
        out["image2"] = torch.stack([item["image2"] for item in batch])
    return out
