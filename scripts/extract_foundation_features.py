"""Stage F1 -- encode every BreaKHis image once with a frozen foundation model.

What this does, in words
------------------------
Download / load the configured pathology foundation model (default: **CTransPath**,
ungated, ~27.5 M parameters), freeze it, and run all ~7.9k protocol images through
it **once** under the project's *eval* transforms -- the exact same
``Resize -> ToTensor -> Normalize`` pipeline the memory bank encodes with, and the
same one exp1-exp3n are tested with. Write the pooled feature vectors, plus label /
patient / subtype / magnification / path, into ONE ``.npz``.

Nothing here is fitted, so nothing here can leak: the encoder saw TCGA and PAIP,
never BreaKHis, and no gradient flows back into it. Everything that *is* fitted
from labels happens in Stage F2, on a fold's TRAIN rows only.

Why the project's eval transform and not CTransPath's own
---------------------------------------------------------
CTransPath's timm config asks for bicubic resize with ``crop_pct=0.9``; this repo
resizes to 224x224 bilinearly with no crop. We use **the repo's**, because the
baseline's job is to occupy one row of a table whose other rows were produced that
way -- a row measured under different preprocessing would not be the controlled
comparison paper S3.8 asks for. Both are ImageNet-normalised, which is what
CTransPath was trained with, so this is a resize-policy difference and it is
recorded in the cache's ``meta.transform``.

Examples
--------
    python scripts/extract_foundation_features.py
    python scripts/extract_foundation_features.py --encoder ctranspath
    python scripts/extract_foundation_features.py --set foundation.batch_size=16
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rap_mst.data.breakhis import subtype_from_patient_id  # noqa: E402
from rap_mst.data.datamodule import BreaKHisDataModule  # noqa: E402
from rap_mst.foundation.builder import foundation_cfg, resolve_cache_path  # noqa: E402
from rap_mst.foundation.cache import FeatureCache  # noqa: E402
from rap_mst.foundation.encoders import build_foundation_encoder  # noqa: E402
from rap_mst.utils import reporting  # noqa: E402
from rap_mst.utils.config import apply_overrides, load_config, parse_set_overrides  # noqa: E402
from rap_mst.utils.logging_utils import get_logger  # noqa: E402
from rap_mst.utils.seed import seed_everything  # noqa: E402


def transform_description(cfg) -> str:
    return (f"Resize(({cfg.data.image_size}, {cfg.data.image_size})) -> ToTensor -> "
            "Normalize(ImageNet mean/std)  [rap_mst.data.transforms, train=False]")


@torch.no_grad()
def encode_all(model, loader, device, amp: bool, logger) -> dict:
    """Run the frozen encoder over the whole dataset, collecting features + metadata."""
    autocast = torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda")
    keys, labels, pids, mags, paths = [], [], [], [], []
    for batch in tqdm(loader, desc="encoding", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with autocast:
            feats = model(images)
        if feats.ndim != 2:
            raise RuntimeError(
                f"Encoder returned {tuple(feats.shape)}; the baseline needs a pooled "
                "[B, D] vector. Check `foundation.encoders.<name>.pool`."
            )
        keys.append(feats.float().cpu().numpy())
        labels.extend(batch["label"].tolist())
        pids.extend(batch["patient_id"])
        mags.extend(batch["magnification"])
        paths.extend(batch["image_path"])
    return {
        "key": np.concatenate(keys, axis=0).astype(np.float32),
        "label": np.asarray(labels, dtype=np.int64),
        "patient_id": np.asarray(pids),
        "subtype": np.asarray([subtype_from_patient_id(p) for p in pids]),
        "magnification": np.asarray(mags, dtype=np.int64),
        "image_path": np.asarray(paths),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F1: cache frozen foundation-model features.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--encoder", default=None,
                    help="Override foundation.encoder (default: from the config).")
    ap.add_argument("--out", default=None, help="Explicit .npz path (default: foundation.cache_path).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing cache instead of refusing.")
    ap.add_argument("--set", action="append", default=[],
                    help="Override any config key: --set foundation.batch_size=16 (repeatable).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    overrides = parse_set_overrides(args.set)
    if args.encoder:
        overrides["foundation.encoder"] = args.encoder
    cfg = apply_overrides(cfg, overrides)
    fcfg = foundation_cfg(cfg)
    encoder_name = str(fcfg.encoder).strip().lower()

    # Encoding is deterministic, but seed anyway so the config's promise holds.
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(args.out) if args.out else resolve_cache_path(cfg, encoder_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger("rap_mst.extract_foundation_features",
                        out_path.parent / "extract_foundation_features.log")

    if out_path.exists() and not args.force:
        raise SystemExit(
            f"Feature cache already exists: {out_path}\n"
            "It is deterministic, so re-running changes nothing -- pass --force to rebuild."
        )

    # Batch size for the encode pass is its own knob: no gradients are stored, so
    # it can exceed the training batch without risking OOM.
    batch_size = int(getattr(fcfg, "batch_size", cfg.data.batch_size))
    cfg = apply_overrides(cfg, {"data.batch_size": batch_size})

    env = reporting.collect_env(device)
    reporting.section(logger, "STAGE F1 -- FOUNDATION FEATURE EXTRACTION", [
        reporting.kv("encoder", encoder_name),
        reporting.kv("output", str(out_path)),
        reporting.kv("image size / batch", f"{cfg.data.image_size} / {batch_size}"),
        reporting.kv("transform", transform_description(cfg)),
        reporting.kv("magnifications", getattr(cfg.data, "magnifications", None) or "all four"),
        reporting.kv("seed / deterministic", f"{cfg.seed} / {cfg.deterministic}"),
        reporting._rule(),
        reporting.kv("device", f"{env['device']}  ({env['gpu_name']})"),
        reporting.kv("torch / timm", f"{env['torch']} / {env['timm']}"),
    ])

    # --- frozen encoder ---------------------------------------------------- #
    logger.info("Loading the frozen encoder (downloads on first use) ...")
    model, meta = build_foundation_encoder(cfg, encoder_name)
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable:
        raise AssertionError(f"{trainable} encoder parameters are still trainable; it must be frozen.")
    reporting.section(logger, "FROZEN ENCODER", [
        reporting.kv("hub id", meta["hub_id"]),
        reporting.kv("patch embed", meta["embed_layer"]),
        reporting.kv("pooling", meta["pool"]),
        reporting.kv("feature dim", meta["feature_dim"]),
        reporting.kv("parameters", f"{meta['parameters']:,} ({meta['parameters'] / 1e6:.2f}M), all frozen"),
        reporting.kv("weights verified", f"{meta['verified_identical']}/{meta['checkpoint_tensors']} "
                                         "checkpoint tensors identical in the model ✓"),
    ])

    # --- data: every protocol patient, eval transforms, in order ------------ #
    dm = BreaKHisDataModule(cfg)
    dm.setup_all()
    logger.info(f"Encoding {len(dm.all_set)} images ...")

    amp = bool(getattr(fcfg, "amp", True)) and device.type == "cuda"
    columns = encode_all(model, dm.all_dataloader(), device, amp, logger)

    cache = FeatureCache(**columns, meta={
        **meta,
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(cfg.data.dataset_root),
        "splits_path": str(cfg.data.splits_path),
        "image_size": int(cfg.data.image_size),
        "transform": transform_description(cfg),
        "amp": amp,
        "seed": int(cfg.seed),
        "torch": env["torch"],
    })

    # --- sanity table -- raises rather than shipping a bad cache ------------ #
    summary = cache.summary()
    if summary["n_nonfinite"]:
        raise AssertionError(f"{summary['n_nonfinite']} non-finite values in the features.")
    if summary["rows"] != len(dm.all_set):
        raise AssertionError(f"Encoded {summary['rows']} rows for {len(dm.all_set)} images.")
    if summary["patients"] != len(dm.splits["test_patients"]) + 66:
        raise AssertionError(f"Cache covers {summary['patients']} patients, expected 82.")
    if len(summary["mag_counts"]) != 4:
        reporting.warn_banner(logger, "Not all four magnifications are present",
                              f"counts: {summary['mag_counts']}")
    # A collapsed encoder (all rows near-identical) would still produce a plausible
    # probe accuracy on a 70%-malignant set, so check the features actually vary.
    spread = float(np.std(cache.key, axis=0).mean())
    if spread < 1e-4:
        raise AssertionError(f"Features are near-constant (mean per-dim std {spread:.2e}).")

    reporting.section(logger, "CACHE SANITY", [
        reporting.kv("rows", f"{summary['rows']}  ({summary['patients']} patients)"),
        reporting.kv("feature dim", summary["dim"]),
        reporting.kv("benign / malignant", f"{summary['benign_rows']} / {summary['malignant_rows']}"),
        reporting.kv("magnification counts", summary["mag_counts"]),
        reporting.kv("L2 norm range", f"{summary['feature_norm_min']:.3f} .. {summary['feature_norm_max']:.3f}"),
        reporting.kv("mean per-dim std", f"{spread:.4f}  (non-degenerate ✓)"),
        reporting.kv("non-finite values", "0 ✓"),
    ])

    cache.save(out_path)
    logger.info(f"Saved feature cache -> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
