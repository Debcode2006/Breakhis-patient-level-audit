"""Stage F3 -- score a fitted probe on the permanent 16-patient held-out test set.

The counterpart of ``scripts/test.py`` for the frozen foundation-model baseline.
It writes the **same two files, with the same names and the same columns**, into
the same ``<run>/test/`` directory:

    test_metrics.json        image- and patient-level metrics
    test_predictions.csv     image_path, patient_id, magnification, label, pred,
                             prob_malignant

so ``diagnose_folds.py`` and ``threshold_calibration.py`` read the baseline exactly
as they read exp1-exp3n, and the main table's rows are produced by one procedure.

What is loaded, and what is not
-------------------------------
The encoder is not loaded: its output for these images is already in the Stage F1
cache, and re-encoding would be both slower and a chance for the two paths to
diverge. What *is* loaded from the checkpoint is the head **and the standardiser
fitted on that fold's training rows** -- re-fitting standardisation on the test set
would be a leak that no metric would reveal, so the stored one is used verbatim and
its provenance is printed.

Examples
--------
    python scripts/test_linear_probe.py --experiment expfm --fold 0
    python scripts/test_linear_probe.py --experiment expfm            # all trained folds
    python scripts/test_linear_probe.py --checkpoint "runs\\expfm_ctranspath_linear\\<run>\\checkpoints\\best.pt"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rap_mst.data.splits import load_splits  # noqa: E402
from rap_mst.experiments import experiment_names, is_foundation  # noqa: E402
from rap_mst.foundation.builder import foundation_cfg, resolve_cache_path  # noqa: E402
from rap_mst.foundation.cache import (  # noqa: E402
    FeatureCache, split_views, write_predictions_csv,
)
from rap_mst.foundation.probe import ProbeHead, Standardizer, predict, score  # noqa: E402
from rap_mst.utils import reporting  # noqa: E402
from rap_mst.utils.checkpoint import load_checkpoint  # noqa: E402
from rap_mst.utils.config import Config, apply_overrides, parse_set_overrides  # noqa: E402
from rap_mst.utils.logging_utils import get_logger  # noqa: E402
from rap_mst.utils.runs import available_folds, find_fold_run  # noqa: E402
from rap_mst.utils.seed import seed_everything  # noqa: E402


def evaluate_checkpoint(ckpt_path: Path, args, device: torch.device) -> dict:
    out_dir = Path(args.out) if args.out else ckpt_path.parent.parent / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("rap_mst.test_linear_probe", out_dir / "test.log")

    ckpt = load_checkpoint(ckpt_path, map_location=device)  # raises if missing
    if "probe" not in ckpt:
        raise SystemExit(
            f"{ckpt_path} is not a probe checkpoint (no 'probe' block). "
            "Swin checkpoints are evaluated with scripts/test.py."
        )
    cfg = apply_overrides(Config(ckpt["config"]), parse_set_overrides(args.set))
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    fold = int(ckpt.get("fold", cfg.train.fold))
    encoder_name = str(foundation_cfg(cfg).encoder).strip().lower()

    # --- features: the same cache Stage F2 fitted on ------------------------ #
    cache_path = Path(args.cache) if args.cache else resolve_cache_path(cfg, encoder_name)
    cache = FeatureCache.load(cache_path)
    cache.assert_encoder(encoder_name)
    if str(cache_path) != str(ckpt.get("feature_cache", cache_path)):
        reporting.warn_banner(
            logger, "Scoring against a different feature cache than the probe was fitted on",
            f"fitted on : {ckpt.get('feature_cache')}", f"scoring on: {cache_path}",
        )

    splits = load_splits(cfg.data.splits_path)
    views = split_views(cache, splits, fold, include_test=True)
    test = views["test"]

    # Belt and braces: the checkpoint records which patients it was fitted on, so
    # a mismatched --fold cannot quietly score a probe against its own training
    # patients. This is the fourth place the protocol is asserted, by design.
    FeatureCache.assert_disjoint(
        ckpt.get("train_patients", []), ckpt.get("val_patients", []), test["patients"],
        names=["checkpoint train", "checkpoint val", "test"],
    )

    reporting.report_test_header(
        logger, checkpoint=str(ckpt_path), cfg=cfg, fold=fold,
        n_patients=len(test["patients"]), n_images=test["rows"],
    )

    # --- head + the standardiser fitted on THIS fold's train rows ----------- #
    probe_spec = ckpt["probe"]
    head = ProbeHead(
        in_dim=int(probe_spec["in_dim"]), num_classes=2,
        head=str(probe_spec["head"]), hidden_dim=int(probe_spec["hidden_dim"]),
        dropout=float(probe_spec["dropout"]),
    ).to(device)
    head.load_state_dict(ckpt["model"])
    head.eval()

    standardizer = Standardizer.from_state_dict(ckpt["standardizer"])
    if int(probe_spec["in_dim"]) != cache.dim:
        raise SystemExit(
            f"Probe expects {probe_spec['in_dim']}-d features but the cache is "
            f"{cache.dim}-d. The cache and the checkpoint come from different encoders."
        )

    reporting.section(logger, "STAGE F3 -- FROZEN FOUNDATION BASELINE", [
        reporting.kv("experiment", cfg.experiment.name),
        reporting.kv("encoder (frozen)", f"{encoder_name}  [{cache.meta.get('hub_id')}]"),
        reporting.kv("encoder params", f"{cache.meta.get('parameters', 0):,}"),
        reporting.kv("feature cache", f"{cache_path}  ({cache.dim}-d)"),
        reporting.kv("head", head.describe()),
        reporting.kv("standardiser fitted on", standardizer.fitted_on or "identity (disabled)"),
        reporting.kv("selection", f"{ckpt.get('monitor')} = {ckpt.get('best_metric'):.4f} "
                                  f"@ epoch {int(ckpt.get('epoch', 0)) + 1}"),
        reporting.kv("leakage guard", "checkpoint train/val ∩ test patients empty ✓"),
    ])

    # --- score --------------------------------------------------------------- #
    x = torch.from_numpy(standardizer.apply(test["key"])).to(device)
    probs = predict(head, x)
    labels = np.asarray(test["label"], dtype=int)
    pids = np.asarray(test["patient_id"])
    metrics = score(labels, probs, pids, threshold=0.5)

    reporting.report_test_metrics(logger, metrics, out_dir)

    metrics_path = out_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    pred_path = write_predictions_csv(out_dir / "test_predictions.csv", test, probs)

    logger.info(f"Saved metrics -> {metrics_path}")
    logger.info(f"Saved preds   -> {pred_path}")
    return {"fold": fold, "run_dir": str(ckpt_path.parent.parent), "metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F3: score a fitted probe on the held-out test set.")
    ap.add_argument("--experiment", default="expfm", choices=experiment_names(),
                    help="A foundation preset (expfm | expfm_mlp).")
    ap.add_argument("--fold", type=int, default=None,
                    help="CV fold index; omit to score every trained fold.")
    ap.add_argument("--checkpoint", default=None,
                    help="Explicit best.pt (overrides --experiment/--fold discovery).")
    ap.add_argument("--config", default="config/config.yaml",
                    help="Used only to locate logging.log_root during run discovery.")
    ap.add_argument("--cache", default=None, help="Explicit feature .npz (default: foundation.cache_path).")
    ap.add_argument("--out", default=None, help="Output directory (default: <run>/test).")
    ap.add_argument("--set", action="append", default=[], help="Override any config key.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        checkpoints = [Path(args.checkpoint)]
    else:
        if not is_foundation(args.experiment):
            raise SystemExit(
                f"'{args.experiment}' is a Swin experiment; evaluate it with scripts/test.py."
            )
        from rap_mst.utils.config import load_config

        log_root = Path(load_config(args.config).logging.log_root)
        folds = [args.fold] if args.fold is not None else available_folds(args.experiment, log_root)
        if not folds:
            raise SystemExit(
                f"No fitted folds for '{args.experiment}' under {log_root}. "
                "Run scripts/train_linear_probe.py first."
            )
        checkpoints = [find_fold_run(args.experiment, log_root, f) / "checkpoints" / "best.pt"
                       for f in folds]

    results = [evaluate_checkpoint(ck, args, device) for ck in checkpoints]

    if len(results) > 1:
        print(f"\n{'fold':>5} {'img_acc':>8} {'img_auc':>8} {'sens':>7} {'spec':>7} "
              f"{'pat_acc':>8} {'pat_auc':>8}")
        for r in results:
            m = r["metrics"]
            print(f"{r['fold']:>5} {m['accuracy']:>8.4f} {m['auc']:>8.4f} "
                  f"{m['sensitivity']:>7.3f} {m['specificity']:>7.3f} "
                  f"{m['patient_accuracy']:>8.4f} {m['patient_auc']:>8.4f}")
        mean = lambda k: float(np.mean([r["metrics"][k] for r in results]))  # noqa: E731
        print(f"{'mean':>5} {mean('accuracy'):>8.4f} {mean('auc'):>8.4f} "
              f"{mean('sensitivity'):>7.3f} {mean('specificity'):>7.3f} "
              f"{mean('patient_accuracy'):>8.4f} {mean('patient_auc'):>8.4f}")


if __name__ == "__main__":
    main()
