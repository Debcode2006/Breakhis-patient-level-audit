"""Stage F2 -- fit the probe head on one fold of cached foundation-model features.

What this does, in words
------------------------
Load the Stage F1 feature cache, take **fold k's TRAIN rows** of it, fit a
per-dimension standardiser and a small head (``Dropout -> Linear(D -> 2)`` by
default) on them with inverse-frequency class-weighted cross-entropy, select the
checkpoint on **fold k's VAL rows** using the same monitor exp1-exp3n used, and
write a run directory that looks exactly like a Swin run's:

    runs/expfm_ctranspath_linear/<timestamp>_train_fold<k>/
        config.yaml  train.log  metrics.csv  tensorboard/
        checkpoints/{best,last}.pt
        val_predictions.csv        <- extra: lets threshold_calibration.py skip
                                      re-running inference (the head is not a model
                                      `build_model` can rebuild)

The encoder is never loaded here. That is the design: it is frozen, its output is
cached, and this stage touches ~1.5k parameters. A fold takes seconds.

The rule that matters more than anything else here
--------------------------------------------------
The standardiser and the head see **only fold k's training patients**. Not its
validation patients, never the 16 test patients. ``split_views`` asserts patient
disjointness before any row is read, and the standardiser records which split it
was fitted on inside the checkpoint. If an assertion fires, do not work around it.

Examples
--------
    python scripts/train_linear_probe.py --experiment expfm --fold 0
    python scripts/train_linear_probe.py --experiment expfm            # all 5 folds
    python scripts/train_linear_probe.py --experiment expfm_mlp --fold 0
    python scripts/train_linear_probe.py --experiment expfm --set foundation.probe.epochs=200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rap_mst.constants import DEFAULT_NUM_FOLDS  # noqa: E402
from rap_mst.data.splits import load_splits  # noqa: E402
from rap_mst.experiments import (  # noqa: E402
    experiment_names, get_experiment_preset, is_foundation,
)
from rap_mst.foundation.builder import (  # noqa: E402
    build_probe_head, foundation_cfg, probe_cfg, resolve_cache_path,
)
from rap_mst.foundation.cache import (  # noqa: E402
    FeatureCache, split_views, write_predictions_csv,
)
from rap_mst.foundation.probe import (  # noqa: E402
    Standardizer, fit_probe, guard_probe_health, predict,
)
from rap_mst.utils import reporting  # noqa: E402
from rap_mst.utils.checkpoint import build_state, save_checkpoint  # noqa: E402
from rap_mst.utils.config import apply_overrides, load_config, parse_set_overrides  # noqa: E402
from rap_mst.utils.experiment import ExperimentDir  # noqa: E402
from rap_mst.utils.logging_utils import CSVMetricLogger, get_logger  # noqa: E402
from rap_mst.utils.seed import seed_everything  # noqa: E402


def build_config(args, fold: int):
    cfg = load_config(args.config)
    overrides = dict(get_experiment_preset(args.experiment))
    overrides["train.fold"] = fold
    overrides.update(parse_set_overrides(args.set))
    return apply_overrides(cfg, overrides)


def run_fold(args, fold: int, device: torch.device) -> dict:
    cfg = build_config(args, fold)
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    pcfg = probe_cfg(cfg)
    encoder_name = str(foundation_cfg(cfg).encoder).strip().lower()

    expdir = ExperimentDir(cfg.logging.log_root, cfg.experiment.name, fold, tag="train")
    expdir.save_config(cfg)  # persist the EXACT config used
    logger = get_logger("rap_mst.train_linear_probe", expdir.log_path)
    reporting.report_startup(logger, cfg, device, fold)

    # --- the frozen features ------------------------------------------------ #
    cache_path = Path(args.cache) if args.cache else resolve_cache_path(cfg, encoder_name)
    cache = FeatureCache.load(cache_path)
    cache.assert_encoder(encoder_name)
    splits = load_splits(cfg.data.splits_path)
    views = split_views(cache, splits, fold, include_test=False)
    train, val = views["train"], views["val"]

    reporting.section(logger, f"STAGE F2 -- PROBE  fold {fold}", [
        reporting.kv("experiment", f"{args.experiment}  ({cfg.experiment.name})"),
        reporting.kv("encoder (frozen)", f"{encoder_name}  [{cache.meta.get('hub_id')}]"),
        reporting.kv("feature cache", f"{cache_path}  ({len(cache)} rows, {cache.dim}-d)"),
        reporting.kv("cache transform", cache.meta.get("transform")),
        reporting._rule(),
        reporting.kv("train", f"{train['rows']} images / {len(train['patients'])} patients"),
        reporting.kv("val", f"{val['rows']} images / {len(val['patients'])} patients"),
        reporting.kv("train ∩ val patients", "empty ✓"),
        reporting.kv("train/val ∩ test patients", "empty ✓"),
        reporting._rule(),
        reporting.kv("head", str(getattr(pcfg, "head", "linear"))),
        reporting.kv("standardize", bool(getattr(pcfg, "standardize", True))),
        reporting.kv("class weights", bool(getattr(pcfg, "use_class_weights", True))),
        reporting.kv("epochs / batch", f"{pcfg.epochs} / {pcfg.batch_size}"),
        reporting.kv("lr / weight decay", f"{pcfg.lr} / {pcfg.weight_decay}"),
        reporting.kv("monitor / patience",
                     f"{pcfg.monitor} / {getattr(pcfg, 'early_stopping_patience', 0)}"),
    ])

    # --- standardiser: fitted on TRAIN rows only ---------------------------- #
    if bool(getattr(pcfg, "standardize", True)):
        standardizer = Standardizer.fit(train["key"], fitted_on={
            "split": "train", "fold": fold,
            "n_rows": train["rows"], "n_patients": len(train["patients"]),
        })
    else:
        standardizer = Standardizer.identity(cache.dim)
    train["x"] = standardizer.apply(train["key"])
    val["x"] = standardizer.apply(val["key"])

    # --- head --------------------------------------------------------------- #
    head = build_probe_head(cfg, in_dim=cache.dim)
    logger.info(f"Probe: {head.describe()}")

    csv_logger = CSVMetricLogger(expdir.metrics_path)
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(str(expdir.tensorboard))
    except Exception:  # pragma: no cover - tensorboard is optional here
        tb = None

    def on_epoch(epoch: int, row: dict) -> None:
        csv_logger.log(row)
        if tb is not None:
            for key, value in row.items():
                if key != "epoch":
                    tb.add_scalar(key.replace("val_", "val/"), value, epoch)
        if epoch % 10 == 0 or epoch == int(pcfg.epochs) - 1:
            logger.info(
                f"[epoch {epoch + 1}] lr={row['lr']:.2e} | ce={row['train_ce']:.4f} | "
                f"val_acc={row['val_accuracy']:.4f} | val_auc={row['val_auc']:.4f} | "
                f"val_pat_acc={row['val_patient_accuracy']:.4f}"
            )

    started = time.time()
    best, history = fit_probe(
        head, train, val,
        device=device,
        epochs=int(pcfg.epochs),
        batch_size=int(pcfg.batch_size),
        lr=float(pcfg.lr),
        weight_decay=float(pcfg.weight_decay),
        scheduler=str(getattr(pcfg, "scheduler", "cosine")),
        min_lr=float(getattr(pcfg, "min_lr", 1e-6)),
        monitor=str(pcfg.monitor),
        patience=int(getattr(pcfg, "early_stopping_patience", 0)),
        use_class_weights=bool(getattr(pcfg, "use_class_weights", True)),
        seed=int(cfg.seed),
        on_epoch=on_epoch,
    )
    elapsed = time.time() - started
    if tb is not None:
        tb.close()

    guard_probe_health(best, train["label"],
                       lambda title, *rest: reporting.warn_banner(logger, title, *rest))

    # --- val predictions at the selected checkpoint -------------------------- #
    val_probs = predict(head, torch.from_numpy(val["x"]).to(device))
    val_csv = write_predictions_csv(expdir.root / "val_predictions.csv", val, val_probs)

    # --- checkpoint ---------------------------------------------------------- #
    state = build_state(
        model=head, optimizer=None, scheduler=None, scaler=None,
        epoch=int(best["epoch"] or 0), best_metric=float(best["value"]),
        config=cfg.to_dict(),
        extra={
            "monitor": str(pcfg.monitor),
            "probe": {
                "head": str(getattr(pcfg, "head", "linear")),
                "in_dim": int(cache.dim),
                "hidden_dim": int(getattr(pcfg, "hidden_dim", 512)),
                "dropout": float(getattr(pcfg, "dropout", 0.0)),
            },
            "standardizer": standardizer.state_dict(),
            "encoder_meta": cache.meta,
            "feature_cache": str(cache_path),
            "fold": fold,
            "train_patients": list(train["patients"]),
            "val_patients": list(val["patients"]),
        },
    )
    save_checkpoint(state, expdir.best_ckpt)
    save_checkpoint(state, expdir.last_ckpt)  # identical: `best` is the fitted head

    val_metrics = best["val_metrics"]
    reporting.section(logger, f"PROBE SUMMARY  fold {fold}", [
        reporting.kv("experiment", cfg.experiment.name),
        reporting.kv("encoder", f"{encoder_name} (frozen, {cache.meta.get('parameters', 0):,} params)"),
        reporting.kv("head", head.describe()),
        reporting._rule(),
        reporting.kv("best epoch", (best["epoch"] or 0) + 1),
        reporting.kv(f"best val {pcfg.monitor}", f"{best['value']:.4f}"),
        *[reporting.kv(f"  best val_{k}", f"{val_metrics[k]:.4f}")
          for k in ("accuracy", "auc", "sensitivity", "specificity",
                    "patient_accuracy", "patient_auc") if k in val_metrics],
        reporting._rule(),
        reporting.kv("fit time", f"{elapsed:.1f}s over {len(history)} epochs"),
        reporting.kv("checkpoints", expdir.checkpoints),
        reporting.kv("val predictions", val_csv),
        reporting.kv("output directory", expdir.root),
    ])

    return {
        "fold": fold, "run_dir": str(expdir.root), "best_epoch": best["epoch"],
        "monitor": str(pcfg.monitor), "best_value": float(best["value"]),
        "val_metrics": val_metrics, "epochs_run": len(history),
        "seconds": round(elapsed, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F2: fit the probe on cached features.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--experiment", default="expfm", choices=experiment_names(),
                    help="A foundation preset (expfm | expfm_mlp).")
    ap.add_argument("--fold", type=int, default=None, help="CV fold index; omit to run all folds.")
    ap.add_argument("--cache", default=None, help="Explicit feature .npz (default: foundation.cache_path).")
    ap.add_argument("--set", action="append", default=[],
                    help="Override any config key: --set foundation.probe.epochs=200 (repeatable).")
    args = ap.parse_args()

    if not is_foundation(args.experiment):
        raise SystemExit(
            f"'{args.experiment}' is a Swin experiment; train it with scripts/train.py. "
            "This script fits a head on cached frozen features (expfm | expfm_mlp)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = [args.fold] if args.fold is not None else list(range(DEFAULT_NUM_FOLDS))
    results = [run_fold(args, fold, device) for fold in folds]

    if len(results) > 1:
        cfg = build_config(args, folds[0])
        out = Path(cfg.logging.log_root) / cfg.experiment.name / "probe_fit_summary.json"
        with out.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nSaved fit summary -> {out}")
        print(f"{'fold':>5} {'best_ep':>8} {'val_acc':>8} {'val_auc':>8} {'val_pat_acc':>12}")
        for r in results:
            m = r["val_metrics"]
            print(f"{r['fold']:>5} {(r['best_epoch'] or 0) + 1:>8} "
                  f"{m.get('accuracy', float('nan')):>8.4f} {m.get('auc', float('nan')):>8.4f} "
                  f"{m.get('patient_accuracy', float('nan')):>12.4f}")


if __name__ == "__main__":
    main()
