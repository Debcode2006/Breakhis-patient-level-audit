"""Stage B1 -- build the retrieval memory bank for one (or every) fold.

What this does, in words
------------------------
Load fold *k*'s frozen ``best.pt``. Run every **training** image of that fold
through it. Take ``out["features"]`` -- the 1024-d vector *before* the
magnification block (``docs/retrieval.md`` D1) -- L2-normalise it, and write it to a
table along with the image's label, subtype, patient id and magnification. Then
group those rows by patient, average, re-normalise, and append them as ``slide``
rows (``docs/retrieval.md`` Part V). Save ONE ``.npz`` per fold.

The rule that matters more than anything else here
--------------------------------------------------
Fold *k*'s bank contains **only fold *k*'s training patients**. Not its validation
patients. Never the 16 test patients. If a validation patient is in the bank, the
gate fitted in Stage B2 is fitted on retrieval results that saw the answer, and
every number after that is worthless. This is asserted at build time (below) and
again at query time (``MemoryBank.query`` / ``assert_disjoint``). If an assertion
fires, do not work around it.

The encoder is *frozen*: no gradients, no augmentation (eval transforms), no
shuffling, no dropped last batch -- ``dm.bank_dataloader()`` exists precisely so
the bank cannot silently miss images.

Examples
--------
    python scripts/build_memory_bank.py --experiment exp3n              # all 5 folds
    python scripts/build_memory_bank.py --experiment exp3n --fold 0
    python scripts/build_memory_bank.py --experiment exp3n --set retrieval.key=projections
    python scripts/build_memory_bank.py --experiment exp3n \
        --set retrieval.key=features --set retrieval.key_transform=whiten:128

``retrieval.key`` accepts the spec language in ``rap_mst/retrieval/keys.py``
(forward-dict entries *and* FPN poolings) and ``retrieval.key_transform`` the
fitted reshaping in ``rap_mst/retrieval/transform.py``; both are written into the
bank and asserted at load, so a bank can never be queried with the wrong key.
``docs/results/retrieval_key_ablation.md`` is the measured comparison of those choices.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rap_mst.data.datamodule import BreaKHisDataModule  # noqa: E402
from rap_mst.data.splits import get_fold  # noqa: E402
from rap_mst.experiments import (  # noqa: E402
    assert_not_foundation, encoder_experiment, experiment_names,
)
from rap_mst.models.builder import build_model  # noqa: E402
from rap_mst.retrieval.bank import MemoryBank  # noqa: E402
from rap_mst.retrieval.builder import resolve_bank_path, retrieval_cfg  # noqa: E402
from rap_mst.retrieval.keys import parse_key_spec  # noqa: E402
from rap_mst.retrieval.transform import classifier_direction, parse_transform_spec  # noqa: E402
from rap_mst.utils import reporting  # noqa: E402
from rap_mst.utils.checkpoint import load_checkpoint  # noqa: E402
from rap_mst.utils.config import (  # noqa: E402
    apply_overrides, load_config, merge_section, parse_set_overrides,
)
from rap_mst.utils.logging_utils import get_logger  # noqa: E402
from rap_mst.utils.runs import available_folds, find_fold_run  # noqa: E402
from rap_mst.utils.seed import seed_everything  # noqa: E402


def resolve_experiment(cli_value, base_cfg) -> str:
    """CLI wins; otherwise the config's declared base encoder (``retrieval.base_experiment``)."""
    name = cli_value or str(getattr(retrieval_cfg(base_cfg), "base_experiment", "exp3n"))
    if name not in experiment_names():
        raise SystemExit(
            f"Unknown experiment '{name}'. Set retrieval.base_experiment to one of "
            f"{experiment_names()} or pass --experiment."
        )
    return name


def build_fold(args, base_cfg, experiment: str, encoder: str, fold: int,
               device: torch.device, logger) -> dict:
    """Encode one fold's training patients into a bank and save it."""
    log_root = Path(base_cfg.logging.log_root)
    run_dir = find_fold_run(encoder, log_root, fold)
    ckpt_path = run_dir / "checkpoints" / "best.pt"

    # Model/data config comes from the run that produced the weights; the
    # `retrieval:` block from the current config/config.yaml (it did not exist
    # when these checkpoints were trained).
    cfg = load_config(str(run_dir / "config.yaml"))
    cfg = merge_section(cfg, "retrieval", base_cfg)
    cfg = apply_overrides(cfg, parse_set_overrides(args.set))
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    key = retrieval_cfg(cfg).key
    key_transform = str(getattr(retrieval_cfg(cfg), "key_transform", "none") or "none")
    parse_key_spec(key)                 # fail on a typo before a 4-minute encode
    parse_transform_spec(key_transform)
    out_path = Path(args.out).joinpath(f"bank_fold{fold}.npz") if args.out \
        else resolve_bank_path(cfg, experiment, fold)

    reporting.section(logger, f"STAGE B1 -- MEMORY BANK  fold {fold}", [
        reporting.kv("experiment", f"{experiment}  (encoder: {encoder})"),
        reporting.kv("run dir", str(run_dir)),
        reporting.kv("checkpoint", str(ckpt_path)),
        reporting.kv("key (D1)", key),
        reporting.kv("key_transform", key_transform),
        reporting.kv("output", str(out_path)),
        reporting.kv("image size / batch", f"{cfg.data.image_size} / {cfg.data.batch_size}"),
        reporting.kv("magnifications", getattr(cfg.data, "magnifications", None) or "all four"),
        reporting.kv("seed / deterministic", f"{cfg.seed} / {cfg.deterministic}"),
    ])

    # --- data: the fold's TRAIN patients, eval transforms, in order --------- #
    dm = BreaKHisDataModule(cfg)
    dm.setup_bank(fold)
    loader = dm.bank_dataloader()

    fold_rec = get_fold(dm.splits, fold)
    train_patients = set(fold_rec["train_patients"])
    val_patients = set(fold_rec["val_patients"])
    test_patients = set(dm.splits["test_patients"])
    if train_patients & val_patients or train_patients & test_patients:
        raise AssertionError("LEAKAGE: fold train patients overlap val/test in the splits file.")

    # --- frozen encoder ----------------------------------------------------- #
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    bank = MemoryBank(key=key, key_transform=key_transform, meta={
        "experiment": experiment,
        "encoder_experiment": encoder,
        "fold": fold,
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "splits_path": str(cfg.data.splits_path),
        "n_train_patients": len(train_patients),
        "created": datetime.now().isoformat(timespec="seconds"),
    })

    amp = bool(cfg.train.mixed_precision) and device.type == "cuda"
    logger.info(f"Encoding {len(dm.bank_set)} training images ({len(train_patients)} patients) ...")
    n_images = bank.build_from_loader(model, loader, device, key=key, amp=amp, logger=logger)
    # Order is load-bearing: fit the transform on the fold's TRAIN keys only, then
    # derive centroids in the space the vote actually ranks in.
    fitted = bank.fit_key_transform(aux_dirs=classifier_direction(model, key))
    if not fitted.is_identity:
        logger.info(f"key transform '{key_transform}' fitted on {n_images} train keys: "
                    f"{fitted.info.get('in_dim')} -> {fitted.info.get('out_dim')} dims "
                    f"| steps={fitted.info.get('steps')}")
    n_slides = bank.derive_slide_rows()

    # --- Stage B1 sanity table (docs/retrieval.md) ------------------------- #
    summary = bank.summary()
    bank_patients = set(bank.unique_patients())
    bank.assert_disjoint(val_patients, what="validation")
    bank.assert_disjoint(test_patients, what="test")
    if bank_patients != train_patients:
        raise AssertionError(
            f"Bank patients != fold train patients "
            f"(missing {sorted(train_patients - bank_patients)[:5]}, "
            f"extra {sorted(bank_patients - train_patients)[:5]})."
        )
    norm_ok = abs(summary["key_norm_min"] - 1.0) < 1e-4 and abs(summary["key_norm_max"] - 1.0) < 1e-4
    if not norm_ok:
        raise AssertionError(f"Keys are not unit-norm: {summary['key_norm_min']}..{summary['key_norm_max']}")
    if len(summary["mag_shard_sizes"]) != 4:
        reporting.warn_banner(
            logger, "Magnification shards are not all populated",
            f"shards: {summary['mag_shard_sizes']}",
            "same_mag routing (D3) will fall back to fewer neighbours for the missing zooms.",
        )

    reporting.section(logger, f"BANK SANITY  fold {fold}", [
        reporting.kv("image rows", n_images),
        reporting.kv("slide rows", f"{n_slides}  (= training patients)"),
        reporting.kv("bank ∩ val patients", "empty ✓"),
        reporting.kv("bank ∩ test patients", "empty ✓"),
        reporting.kv("key L2 norms", f"{summary['key_norm_min']:.6f} .. {summary['key_norm_max']:.6f} ✓"),
        reporting.kv("mag shard sizes", summary["mag_shard_sizes"]),
        reporting.kv("max image rows / patient", summary["max_image_rows_per_patient"]),
        reporting.kv("key dim", f"{summary['dim']}  (key={key}, transform={key_transform})"),
    ])

    bank.save(out_path)
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved bank -> {out_path}  ({size_mb:.1f} MB)")
    return {"fold": fold, "path": str(out_path), "size_mb": round(size_mb, 2), **summary}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage B1: build the retrieval memory bank.")
    ap.add_argument("--experiment", default=None, choices=experiment_names(),
                    help="Retrieval identity (paths + bank metadata). "
                         "Default: retrieval.base_experiment from the config (exp3n, D9).")
    ap.add_argument("--fold", type=int, default=None, help="CV fold index; omit to build all folds.")
    ap.add_argument("--config", default="config/config.yaml",
                    help="Source of the retrieval: block and logging.log_root.")
    ap.add_argument("--out", default=None,
                    help="Directory override for the .npz files (default: retrieval.bank_path).")
    ap.add_argument("--set", action="append", default=[],
                    help="Override any config key: --set retrieval.key=embeddings (repeatable).")
    args = ap.parse_args()
    assert_not_foundation(args.experiment, "scripts/build_memory_bank.py")

    base_cfg = load_config(args.config)
    if retrieval_cfg(base_cfg) is None:
        raise SystemExit(f"No `retrieval:` block in {args.config}.")

    experiment = resolve_experiment(args.experiment, base_cfg)
    encoder = encoder_experiment(experiment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_root = Path(base_cfg.logging.log_root)

    folds = [args.fold] if args.fold is not None else available_folds(encoder, log_root)
    if not folds:
        raise SystemExit(f"No trained folds found for encoder '{encoder}' under {log_root}.")

    out_root = Path(args.out) if args.out else resolve_bank_path(base_cfg, experiment, 0).parent
    out_root.mkdir(parents=True, exist_ok=True)
    logger = get_logger("rap_mst.build_memory_bank", out_root / "build_memory_bank.log")

    env = reporting.collect_env(device)
    reporting.report_retrieval_config(
        logger, retrieval_cfg(base_cfg), title="STAGE B1 -- BUILD MEMORY BANK",
        extra=[
            reporting._rule(),
            reporting.kv("experiment", f"{experiment}  (encoder: {encoder})"),
            reporting.kv("folds", folds),
            reporting.kv("output dir", str(out_root)),
            reporting.kv("device", f"{env['device']}  ({env['gpu_name']})"),
            reporting.kv("torch / timm", f"{env['torch']} / {env['timm']}"),
        ],
    )

    results = [build_fold(args, base_cfg, experiment, encoder, f, device, logger) for f in folds]

    with (out_root / "bank_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"Saved summary -> {out_root / 'bank_summary.json'}")


if __name__ == "__main__":
    main()
