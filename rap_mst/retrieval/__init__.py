"""Retrieval Memory module (v1) -- non-parametric memory + a ~100-parameter gate.

Design and evidence: ``docs/retrieval.md`` (decisions D1-D9) and ``docs/retrieval.md``
(magnification verdict, unified-bank measurement, operator handbook). Built on the
**exp3n** encoder -- exp3 minus the magnification embedding -- whose SupCon space
is not magnification-locked and whose 1024-d ``features`` vector *is* the space
SupCon optimises (D9 / Part 4).

    bank.py     MemoryBank   ONE store: build / save / load / shard / view / query
    memory.py   RetrievalMemory + RetrievalRunner (two views, never one ranking)
    gate.py     FusionGate   learned blend weights, fitted on pooled OOF val
    builder.py  build_retrieval(cfg) -- config-driven, mirrors build_model(cfg)
    foreign.py  ForeignKeyCache -- keys from an *external* frozen encoder's feature
                cache (the cross-encoder screen), row-aligned by image_path

Torch-heavy submodules are imported lazily (same pattern as ``rap_mst/data``) so
importing the package never costs a torch import on its own.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "BankEntry",
    "MemoryBank",
    "RetrievalResult",
    "RetrievalMemory",
    "RetrievalRunner",
    "LevelConfig",
    "FusionGate",
    "fit_gate",
    "build_retrieval",
    "build_memory",
    "resolve_bank_path",
    "resolve_gate_path",
    "ForeignKeyCache",
    "load_foreign_caches",
]

_LAZY = {
    "BankEntry": ("rap_mst.retrieval.bank", "BankEntry"),
    "MemoryBank": ("rap_mst.retrieval.bank", "MemoryBank"),
    "RetrievalResult": ("rap_mst.retrieval.bank", "RetrievalResult"),
    "RetrievalMemory": ("rap_mst.retrieval.memory", "RetrievalMemory"),
    "RetrievalRunner": ("rap_mst.retrieval.memory", "RetrievalRunner"),
    "LevelConfig": ("rap_mst.retrieval.memory", "LevelConfig"),
    "FusionGate": ("rap_mst.retrieval.gate", "FusionGate"),
    "fit_gate": ("rap_mst.retrieval.gate", "fit_gate"),
    "build_retrieval": ("rap_mst.retrieval.builder", "build_retrieval"),
    "build_memory": ("rap_mst.retrieval.builder", "build_memory"),
    "resolve_bank_path": ("rap_mst.retrieval.builder", "resolve_bank_path"),
    "resolve_gate_path": ("rap_mst.retrieval.builder", "resolve_gate_path"),
    "ForeignKeyCache": ("rap_mst.retrieval.foreign", "ForeignKeyCache"),
    "load_foreign_caches": ("rap_mst.retrieval.foreign", "load_foreign_caches"),
}


def __getattr__(name: str):  # PEP 562 lazy attribute loading
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # help static analysers without triggering torch at import
    from rap_mst.retrieval.bank import BankEntry, MemoryBank, RetrievalResult
    from rap_mst.retrieval.builder import (
        build_memory,
        build_retrieval,
        resolve_bank_path,
        resolve_gate_path,
    )
    from rap_mst.retrieval.foreign import ForeignKeyCache, load_foreign_caches
    from rap_mst.retrieval.gate import FusionGate, fit_gate
    from rap_mst.retrieval.memory import LevelConfig, RetrievalMemory, RetrievalRunner
