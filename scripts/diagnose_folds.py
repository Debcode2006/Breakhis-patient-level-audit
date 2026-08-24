"""Diagnose cross-fold val metrics: is the accuracy 'trend' real or cluster noise?

Re-evaluates the saved ``best.pt`` of each completed fold run on that fold's
validation set and prints, per fold:

  * image-level accuracy / AUC and patient-level accuracy / AUC,
  * a PER-PATIENT table (mean malignant prob, prediction, correct?), and how
    many patients would have to flip to erase the gap to the next fold.

The point: with ~13 val patients whose ~90 images each are near-duplicates, the
effective sample size is the patient count, not the image count. If the fold-to-
fold accuracy gaps reduce to one or two patients being easy/hard, the 'trend'
is cluster variance, not leakage or a bug.

Usage
-----
    python scripts/diagnose_folds.py --experiment exp1
    python scripts/diagnose_folds.py --runs runs/exp1_swin_cls
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rap_mst.data.datamodule import BreaKHisDataModule  # noqa: E402
from rap_mst.engine.evaluator import evaluate  # noqa: E402
from rap_mst.experiments import resolve_experiment_dir  # noqa: E402
from rap_mst.models.builder import build_model  # noqa: E402
from rap_mst.utils.config import load_config  # noqa: E402

_FOLD_RE = re.compile(r"_train_fold(\d+)$")


def find_runs(runs_root: Path) -> dict[int, Path]:
    """Map fold index -> latest run dir under ``runs_root`` that has a best.pt."""
    found: dict[int, Path] = {}
    for d in sorted(runs_root.glob("*_train_fold*")):
        m = _FOLD_RE.search(d.name)
        if not m or not (d / "checkpoints" / "best.pt").exists():
            continue
        fold = int(m.group(1))
        # sorted() is chronological (timestamp prefix) -> keep the latest.
        found[fold] = d
    return found


def evaluate_fold(run_dir: Path, fold: int, device: torch.device) -> dict:
    """Load best.pt for a fold, evaluate on its val set, return metrics + raw."""
    cfg = load_config(str(run_dir / "config.yaml"))
    dm = BreaKHisDataModule(cfg)
    dm.setup_fold(fold)

    model = build_model(cfg).to(device)
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    amp = bool(cfg.train.mixed_precision) and device.type == "cuda"
    metrics, raw = evaluate(model, dm.val_dataloader(), device, amp=amp)
    return {"cfg": cfg, "metrics": metrics, "raw": raw, "best_epoch": ckpt.get("epoch")}


def per_patient(raw: dict) -> list[dict]:
    """Collapse image-level raw predictions to one row per patient."""
    pid = np.asarray(raw["patient_id"])
    label = np.asarray(raw["label"])
    prob = np.asarray(raw["prob"])
    rows = []
    for p in sorted(set(pid.tolist())):
        mask = pid == p
        mean_prob = float(prob[mask].mean())
        img_correct = int(((prob[mask] >= 0.5).astype(int) == label[mask]).sum())
        rows.append(
            {
                "patient": p,
                "label": int(label[mask][0]),
                "n_img": int(mask.sum()),
                "mean_prob": mean_prob,
                "patient_pred": int(mean_prob >= 0.5),
                "img_acc": img_correct / int(mask.sum()),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-patient cross-fold val diagnostic.")
    ap.add_argument("--experiment", default="exp1", help="Experiment name (dir under runs/).")
    ap.add_argument("--runs", default=None, help="Explicit runs dir (overrides --experiment).")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    if args.runs:
        runs_root = Path(args.runs)
    else:
        cfg = load_config(args.config)
        base = Path(cfg.logging.log_root)
        # runs/<experiment.name>/ via the preset registry ('exp3n' -> the exp3n
        # dir); fall back to an '_'-anchored prefix match so 'exp3' can never
        # resolve to 'exp3n_...'.
        cand = base / resolve_experiment_dir(args.experiment)
        if cand.exists():
            runs_root = cand
        else:
            matches = sorted(base.glob(f"{args.experiment}_*"))
            runs_root = matches[0] if matches else base / cfg.experiment.name
    if not runs_root.exists():
        raise SystemExit(f"Runs dir not found: {runs_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = find_runs(runs_root)
    if not runs:
        raise SystemExit(f"No completed fold runs with best.pt under {runs_root}")

    print(f"Runs dir: {runs_root}   device: {device}")
    summary = []
    for fold in sorted(runs):
        res = evaluate_fold(runs[fold], fold, device)
        m = res["metrics"]
        rows = per_patient(res["raw"])
        n_pat = len(rows)
        wrong = [r for r in rows if r["patient_pred"] != r["label"]]

        print("\n" + "=" * 78)
        print(f"FOLD {fold}  (best epoch {res['best_epoch']})  run={runs[fold].name}")
        print(
            f"  image:   acc={m['accuracy']:.4f}  auc={m['auc']:.4f}   "
            f"patient: acc={m['patient_accuracy']:.4f}  auc={m['patient_auc']:.4f}"
        )
        print(f"  {n_pat} val patients,  {len(wrong)} misclassified at patient level")
        print(f"  {'patient':30} {'lbl':>3} {'n_img':>5} {'meanP':>6} {'pPred':>5} {'imgAcc':>7}")
        for r in sorted(rows, key=lambda x: (x["label"], x["mean_prob"])):
            flag = "  <-- WRONG" if r["patient_pred"] != r["label"] else ""
            print(
                f"  {r['patient']:30} {r['label']:>3} {r['n_img']:>5} "
                f"{r['mean_prob']:>6.3f} {r['patient_pred']:>5} {r['img_acc']:>7.3f}{flag}"
            )
        summary.append((fold, m["accuracy"], m["auc"], m["patient_accuracy"], m["patient_auc"], n_pat))

    print("\n" + "=" * 78)
    print("CROSS-FOLD SUMMARY")
    print(f"  {'fold':>4} {'img_acc':>8} {'img_auc':>8} {'pat_acc':>8} {'pat_auc':>8} {'n_pat':>6}")
    arr = np.array([[s[1], s[2], s[3], s[4]] for s in summary])
    for s in summary:
        print(f"  {s[0]:>4} {s[1]:>8.4f} {s[2]:>8.4f} {s[3]:>8.4f} {s[4]:>8.4f} {s[5]:>6}")
    print(f"  {'mean':>4} {arr[:,0].mean():>8.4f} {arr[:,1].mean():>8.4f} "
          f"{arr[:,2].mean():>8.4f} {arr[:,3].mean():>8.4f}")
    print(f"  {'std':>4} {arr[:,0].std():>8.4f} {arr[:,1].std():>8.4f} "
          f"{arr[:,2].std():>8.4f} {arr[:,3].std():>8.4f}")
    print("\nRead: if img_auc is flat/non-monotone while img_acc trends, the 'trend'\n"
          "is threshold+prevalence, not discrimination. If one WRONG patient per fold\n"
          "explains the gaps, it is cluster variance (effective n = patient count).")


if __name__ == "__main__":
    main()
