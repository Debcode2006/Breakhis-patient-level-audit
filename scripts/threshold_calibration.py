"""Threshold-calibration experiment: does a val-tuned decision threshold recover
the accuracy that AUC says is there?

Motivation
----------
`docs/results/classifier_ladder.md` established that magnification embedding (exp2) and SupCon (exp3) did
NOT raise thresholded test accuracy at the fixed 0.5 cut, yet held or improved
**AUC** (exp3 best, 0.9607). AUC is threshold-free: it measures *ranking*. A
better ranking on a 72%-malignant set can still lose accuracy at 0.5 if the
components push the operating point benign-ward. This script tests that
hypothesis directly by *calibrating the threshold on validation data* and then
scoring the untouched held-out test set at the locked threshold.

Protocol (NO test leakage)
--------------------------
The decision threshold is chosen using **validation predictions only**. The
16-patient test set is never used to pick a threshold — it is only scored at the
already-locked value. Two calibration modes are reported:

* ``per_fold``  — the honest, deployment-matched mode. For each of the 5 CV folds
  we (1) re-run the fold's ``best.pt`` on that fold's validation set, (2) pick the
  threshold that maximises **validation accuracy**, and (3) apply *that* threshold
  to *that same fold's* test predictions. Test accuracies are then averaged across
  folds — exactly the aggregation used in `docs/results/classifier_ladder.md`, only with a calibrated cut
  instead of 0.5.

* ``pooled_oof`` — a robustness cross-check. The five folds' validation sets are a
  disjoint partition of the 66 CV patients, so concatenating their predictions is
  a full **out-of-fold** coverage in which every patient is scored by a model that
  never trained on it. We pick ONE global threshold on that pool and apply it to
  every fold's test set (then average) and to the pooled test predictions.

Both the **image level** (threshold on each image's P(malignant)) and the
**patient level** (threshold on the per-patient mean P(malignant); this is the
headline BreaKHis metric and the training monitor) are calibrated and reported.

Val predictions are recomputed on the GPU from ``best.pt`` (they are not saved
during training). Test predictions are read verbatim from the already-saved,
exact ``test/test_predictions.csv`` of each run.

Usage
-----
    python scripts/threshold_calibration.py                 # all three experiments
    python scripts/threshold_calibration.py --experiments exp1 exp3
    python scripts/threshold_calibration.py --out analysis/threshold
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rap_mst.data.datamodule import BreaKHisDataModule  # noqa: E402
from rap_mst.engine.evaluator import evaluate  # noqa: E402
from rap_mst.experiments import EXPERIMENT_DIRS  # noqa: E402
from rap_mst.models.builder import build_model  # noqa: E402
from rap_mst.utils.config import load_config  # noqa: E402
from rap_mst.utils.metrics import best_accuracy_threshold  # noqa: E402
from rap_mst.utils.seed import seed_everything  # noqa: E402

_FOLD_RE = re.compile(r"_train_fold(\d+)$")

# Experiment dir names under runs/, derived from the preset registry so a new
# experiment (e.g. exp3n) needs no edit here.
DEFAULT_EXPERIMENTS = dict(EXPERIMENT_DIRS)


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
def find_runs(runs_root: Path) -> Dict[int, Path]:
    """Map fold index -> latest run dir under ``runs_root`` that has a best.pt."""
    found: Dict[int, Path] = {}
    for d in sorted(runs_root.glob("*_train_fold*")):
        m = _FOLD_RE.search(d.name)
        if not m or not (d / "checkpoints" / "best.pt").exists():
            continue
        found[int(m.group(1))] = d  # sorted() is chronological -> keep the latest
    return found


def resolve_runs_root(experiment: str, log_root: Path) -> Path:
    """Resolve the runs/<experiment> directory, tolerating short names (exp1)."""
    if experiment in DEFAULT_EXPERIMENTS:
        cand = log_root / DEFAULT_EXPERIMENTS[experiment]
        if cand.exists():
            return cand
    direct = log_root / experiment
    if direct.exists():
        return direct
    # Prefix fallback, anchored on '_' so 'exp3' never resolves to 'exp3n_...'.
    matches = sorted(log_root.glob(f"{experiment}_*"))
    if matches:
        return matches[0]
    raise SystemExit(f"Runs dir not found for experiment '{experiment}' under {log_root}")


# --------------------------------------------------------------------------- #
# Prediction loading
# --------------------------------------------------------------------------- #
def _read_predictions_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """Read a saved predictions CSV -> (label, prob, patient_id)."""
    labels, probs, pids = [], [], []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
            probs.append(float(row["prob_malignant"]))
            pids.append(row["patient_id"])
    return {
        "label": np.asarray(labels, dtype=int),
        "prob": np.asarray(probs, dtype=float),
        "patient_id": np.asarray(pids),
    }


def val_predictions(run_dir: Path, fold: int, device: torch.device) -> Dict[str, np.ndarray]:
    """The fold's validation predictions -> (label, prob, patient_id).

    Preferred source is a ``val_predictions.csv`` saved by the run itself. The Swin
    runs (exp1-exp3n) predate it and do not have one, so they fall through to
    re-running ``best.pt`` -- unchanged behaviour, byte-identical numbers. The
    foundation-model baseline *does* write one, and must: its checkpoint holds a
    probe head over cached features, which ``build_model`` cannot reconstruct.
    """
    saved = run_dir / "val_predictions.csv"
    if saved.exists():
        return _read_predictions_csv(saved)

    cfg = load_config(str(run_dir / "config.yaml"))
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    dm = BreaKHisDataModule(cfg)
    dm.setup_fold(fold)

    model = build_model(cfg).to(device)
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    amp = bool(cfg.train.mixed_precision) and device.type == "cuda"
    _, raw = evaluate(model, dm.val_dataloader(), device, amp=amp)
    return {
        "label": np.asarray(raw["label"], dtype=int),
        "prob": np.asarray(raw["prob"], dtype=float),
        "patient_id": np.asarray(raw["patient_id"]),
    }


def test_predictions(run_dir: Path) -> Dict[str, np.ndarray]:
    """Read the saved test_predictions.csv (exact, already computed at train time)."""
    csv_path = run_dir / "test" / "test_predictions.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"Missing {csv_path}. Run scripts/test.py (or scripts/test_linear_probe.py "
            "for a foundation baseline) on this fold's best.pt first."
        )
    return _read_predictions_csv(csv_path)


# --------------------------------------------------------------------------- #
# Patient-level pooling
# --------------------------------------------------------------------------- #
def to_patient_level(pred: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse image predictions to (patient_label, patient_mean_prob) arrays."""
    pid, label, prob = pred["patient_id"], pred["label"], pred["prob"]
    p_labels, p_probs = [], []
    for p in np.unique(pid):
        mask = pid == p
        p_labels.append(int(label[mask][0]))  # constant within a patient
        p_probs.append(float(prob[mask].mean()))
    return np.asarray(p_labels, dtype=int), np.asarray(p_probs, dtype=float)


# --------------------------------------------------------------------------- #
# Threshold sweep
# --------------------------------------------------------------------------- #
# The threshold sweep itself now lives in rap_mst/utils/metrics.py (verbatim), so
# the retrieval gate's pooled-OOF calibration uses the exact same procedure.
def scored(labels: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, float]:
    """Accuracy / sensitivity / specificity at a fixed threshold."""
    preds = (probs >= thr).astype(int)
    pos, neg = labels == 1, labels == 0
    sens = float((preds[pos] == 1).mean()) if pos.any() else float("nan")
    spec = float((preds[neg] == 0).mean()) if neg.any() else float("nan")
    return {
        "accuracy": float((preds == labels).mean()),
        "sensitivity": sens,
        "specificity": spec,
        "n": int(labels.size),
    }


# --------------------------------------------------------------------------- #
# Per-experiment analysis
# --------------------------------------------------------------------------- #
def analyse_experiment(name: str, runs_root: Path, device: torch.device) -> dict:
    runs = find_runs(runs_root)
    if not runs:
        raise SystemExit(f"No completed fold runs with best.pt under {runs_root}")
    folds = sorted(runs)
    print(f"\n### {name}  ({runs_root.name})  folds={folds}")

    per_fold = []
    val_pool = {"label": [], "prob": [], "patient_id": []}  # image level, OOF
    val_pat_pool = {"label": [], "prob": []}                # patient level, OOF
    test_pool = {"label": [], "prob": [], "patient_id": []}

    for fold in folds:
        source = "reading saved val_predictions.csv" if (runs[fold] / "val_predictions.csv").exists() \
            else "re-running best.pt on the val set"
        print(f"  fold {fold}: {source}...", flush=True)
        vp = val_predictions(runs[fold], fold, device)
        tp = test_predictions(runs[fold])

        # Image-level calibration on val, applied to test.
        img_cal = best_accuracy_threshold(vp["label"], vp["prob"])
        img_test_05 = scored(tp["label"], tp["prob"], 0.5)
        img_test_cal = scored(tp["label"], tp["prob"], img_cal["threshold"])

        # Patient-level calibration on val, applied to test.
        v_pl, v_pp = to_patient_level(vp)
        t_pl, t_pp = to_patient_level(tp)
        pat_cal = best_accuracy_threshold(v_pl, v_pp)
        pat_test_05 = scored(t_pl, t_pp, 0.5)
        pat_test_cal = scored(t_pl, t_pp, pat_cal["threshold"])

        per_fold.append({
            "fold": fold,
            "image": {"cal": img_cal, "test_0.5": img_test_05, "test_cal": img_test_cal},
            "patient": {"cal": pat_cal, "test_0.5": pat_test_05, "test_cal": pat_test_cal},
        })

        for k in val_pool:
            val_pool[k].append(vp[k])
            test_pool[k].append(tp[k])
        val_pat_pool["label"].append(v_pl)
        val_pat_pool["prob"].append(v_pp)

        print(
            f"    img  thr={img_cal['threshold']:.3f} (val {img_cal['val_accuracy']:.4f}) "
            f"test 0.5={img_test_05['accuracy']:.4f} -> cal={img_test_cal['accuracy']:.4f}   "
            f"| pat thr={pat_cal['threshold']:.3f} "
            f"test 0.5={pat_test_05['accuracy']:.4f} -> cal={pat_test_cal['accuracy']:.4f}"
        )

    # ----- pooled out-of-fold single global threshold ----- #
    val_img = {k: np.concatenate(v) for k, v in val_pool.items()}
    test_img = {k: np.concatenate(v) for k, v in test_pool.items()}
    val_pat_l = np.concatenate(val_pat_pool["label"])
    val_pat_p = np.concatenate(val_pat_pool["prob"])

    oof_img_cal = best_accuracy_threshold(val_img["label"], val_img["prob"])
    oof_pat_cal = best_accuracy_threshold(val_pat_l, val_pat_p)

    # Apply the ONE global threshold per fold, then average (matches report aggregation).
    def per_fold_at(level: str, thr: float) -> Dict[str, float]:
        a, se, sp = [], [], []
        for i, fold in enumerate(folds):
            tp = test_predictions(runs[fold])
            if level == "image":
                s = scored(tp["label"], tp["prob"], thr)
            else:
                tl, tpb = to_patient_level(tp)
                s = scored(tl, tpb, thr)
            a.append(s["accuracy"]); se.append(s["sensitivity"]); sp.append(s["specificity"])
        return {"accuracy": float(np.mean(a)), "sensitivity": float(np.nanmean(se)),
                "specificity": float(np.nanmean(sp))}

    pooled = {
        "image": {
            "threshold": oof_img_cal["threshold"],
            "val_accuracy": oof_img_cal["val_accuracy"],
            "val_acc_at_0.5": oof_img_cal["acc_at_0.5"],
            "test_mean_0.5": per_fold_at("image", 0.5),
            "test_mean_cal": per_fold_at("image", oof_img_cal["threshold"]),
            "test_pooled_0.5": scored(test_img["label"], test_img["prob"], 0.5),
            "test_pooled_cal": scored(test_img["label"], test_img["prob"], oof_img_cal["threshold"]),
        },
        "patient": {
            "threshold": oof_pat_cal["threshold"],
            "val_accuracy": oof_pat_cal["val_accuracy"],
            "val_acc_at_0.5": oof_pat_cal["acc_at_0.5"],
            "test_mean_0.5": per_fold_at("patient", 0.5),
            "test_mean_cal": per_fold_at("patient", oof_pat_cal["threshold"]),
        },
    }

    # ----- per-fold means ----- #
    def mean_over(level: str, which: str, field: str) -> float:
        return float(np.nanmean([pf[level][which][field] for pf in per_fold]))

    summary = {
        "image": {
            "mean_val_threshold": mean_over("image", "cal", "threshold"),
            "mean_test_0.5": mean_over("image", "test_0.5", "accuracy"),
            "mean_test_cal": mean_over("image", "test_cal", "accuracy"),
            "mean_test_0.5_sens": mean_over("image", "test_0.5", "sensitivity"),
            "mean_test_cal_sens": mean_over("image", "test_cal", "sensitivity"),
            "mean_test_0.5_spec": mean_over("image", "test_0.5", "specificity"),
            "mean_test_cal_spec": mean_over("image", "test_cal", "specificity"),
        },
        "patient": {
            "mean_val_threshold": mean_over("patient", "cal", "threshold"),
            "mean_test_0.5": mean_over("patient", "test_0.5", "accuracy"),
            "mean_test_cal": mean_over("patient", "test_cal", "accuracy"),
        },
    }

    return {"name": name, "runs_root": str(runs_root), "folds": folds,
            "per_fold": per_fold, "per_fold_summary": summary, "pooled_oof": pooled}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Val-calibrated decision-threshold experiment.")
    ap.add_argument("--experiments", nargs="+", default=["exp1", "exp2", "exp3"],
                    help="Short names (exp1..) or run-dir names under runs/.")
    ap.add_argument("--config", default="config/config.yaml",
                    help="Config used only to locate the runs/ log_root.")
    ap.add_argument("--out", default="analysis/threshold_calibration",
                    help="Directory for results.json (report md is written to repo root).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    log_root = Path(cfg.logging.log_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  log_root={log_root}")
    results = []
    for exp in args.experiments:
        runs_root = resolve_runs_root(exp, log_root)
        results.append(analyse_experiment(exp, runs_root, device))

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_dir / 'results.json'}")

    # Compact console summary table.
    print("\n" + "=" * 88)
    print("SUMMARY  (per-fold calibration: threshold picked on each fold's val, applied to test)")
    print(f"{'exp':6} {'img_thr':>8} {'img@0.5':>8} {'img_cal':>8} {'d_img':>7} "
          f"{'pat@0.5':>8} {'pat_cal':>8} {'d_pat':>7}")
    for r in results:
        s = r["per_fold_summary"]
        di = s["image"]["mean_test_cal"] - s["image"]["mean_test_0.5"]
        dp = s["patient"]["mean_test_cal"] - s["patient"]["mean_test_0.5"]
        print(f"{r['name']:6} {s['image']['mean_val_threshold']:>8.3f} "
              f"{s['image']['mean_test_0.5']:>8.4f} {s['image']['mean_test_cal']:>8.4f} "
              f"{di:>+7.4f} {s['patient']['mean_test_0.5']:>8.4f} "
              f"{s['patient']['mean_test_cal']:>8.4f} {dp:>+7.4f}")


if __name__ == "__main__":
    main()
