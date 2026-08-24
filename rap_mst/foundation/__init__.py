"""Frozen pathology foundation-model baseline (the paper).

The point of this package is a **single extra row in the paper's main table**: what
does a modern, pathology-pretrained, *frozen* encoder plus a linear head score
under exactly the protocol exp1-exp3n were scored under? A reviewer in 2026 will
ask it, so it is answered before it is asked (paper S3.8).

Design, deliberately mirroring the Retrieval Memory's two-stage discipline
(frozen encoder -> cached store -> tiny fitted head), so nothing here couples to
the trainer, the backbone or the losses:

    Stage F1  encoders.py + cache.py   encode all ~7.9k images ONCE, no gradients,
                                       eval transforms, one .npz
    Stage F2  probe.py                 per fold, fit a ~1.5k-parameter head on the
                                       fold's TRAIN rows of that cache
    Stage F3  (scripts/test_linear_probe.py)  score the permanent 16-patient test
                                       set, writing the same files scripts/test.py
                                       writes so every downstream analysis reads it

Leakage, and why one shared cache is safe
-----------------------------------------
The encoder is frozen and was pretrained on TCGA/PAIP -- it never sees a BreaKHis
label, and no gradient flows from BreaKHis into it -- so caching all 82 patients in
one file leaks nothing. Everything that *is* fitted from labels (the feature
standardiser and the head) is fitted on a fold's TRAIN rows only, and both the
probe and the test script assert train/val/test patient disjointness before they
use a single row.

Torch-heavy modules are imported lazily (as in ``rap_mst.data``) so importing this
package does not drag timm in.
"""

from __future__ import annotations

__all__ = [
    "ConvStem",
    "FeatureCache",
    "ProbeHead",
    "Standardizer",
    "build_foundation_encoder",
    "build_probe_head",
    "foundation_cfg",
    "resolve_cache_path",
]


def __getattr__(name):  # pragma: no cover - thin lazy-import shim
    if name in ("ConvStem", "build_foundation_encoder"):
        from rap_mst.foundation import encoders

        return getattr(encoders, name)
    if name == "FeatureCache":
        from rap_mst.foundation.cache import FeatureCache

        return FeatureCache
    if name in ("ProbeHead", "Standardizer"):
        from rap_mst.foundation import probe

        return getattr(probe, name)
    if name in ("build_probe_head", "foundation_cfg", "resolve_cache_path"):
        from rap_mst.foundation import builder

        return getattr(builder, name)
    raise AttributeError(name)
