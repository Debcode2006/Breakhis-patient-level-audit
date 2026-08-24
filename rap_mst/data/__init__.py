"""Data pipeline: BreaKHis parsing, patient-level splits, dataset and datamodule.

Torch-heavy modules (``dataset``, ``datamodule``) are imported lazily so that
torch-free tooling -- notably ``scripts/prepare_splits.py`` -- can use the
filesystem parser and split generator without importing PyTorch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from rap_mst.data.breakhis import Sample, parse_filename, scan_dataset

__all__ = [
    "Sample",
    "parse_filename",
    "scan_dataset",
    "BreaKHisDataset",
    "BreaKHisDataModule",
]

_LAZY = {
    "BreaKHisDataset": ("rap_mst.data.dataset", "BreaKHisDataset"),
    "BreaKHisDataModule": ("rap_mst.data.datamodule", "BreaKHisDataModule"),
}


def __getattr__(name: str):  # PEP 562 lazy attribute loading
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # help static analysers without triggering torch at import
    from rap_mst.data.dataset import BreaKHisDataset
    from rap_mst.data.datamodule import BreaKHisDataModule
