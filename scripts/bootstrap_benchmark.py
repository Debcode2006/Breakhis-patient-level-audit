"""Patient-clustered bootstrap confidence intervals for the main benchmark table.

The paper (Section 3.7) states that confidence intervals on the paired difference in
AUC and accuracy -- not isolated point estimates -- underpin every claim of
difference. That promise covered only the retrieval streams
(`scripts/retrieval_key_ablation.py`); this script extends it to the whole benchmark
table (`docs/RESEARCH_LOG.md` Section 9.1), which is the table the paper's argument rests on.

Design, mirroring the protocol already used elsewhere in the repo:

* The five fold models share ONE held-out test set, so they are not independent
  replicates. The defensible aggregate is the fold ensemble: the mean malignant
  probability over the five fold models for each image (`docs/results/retrieval_heldout.md` 5).
* Resampling is over the 16 held-out PATIENTS, with replacement -- never over
  images. A patient contributes ~100 highly correlated images; resampling images
  would treat one slide as ~100 independent observations and shrink every interval
  by roughly an order of magnitude.
* Every model is scored on the SAME resample, so the deltas are paired.
* Thresholds are the pre-registered pooled-OOF values, locked before the test set was
  touched. They are inputs here, never re-fitted.

Reads only saved predictions -- no GPU, no model loading, no dataset access.

Usage
-----
    python scripts/bootstrap_benchmark.py
    python scripts/bootstrap_benchmark.py --resamples 2000 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent

# Experiment label -> (run directory name, predictions filename, probability column).
# The locked thresholds are the pooled-OOF operating points from
# `docs/results/threshold_calibration.md` / `docs/RESEARCH_LOG.md` 6.3 and 9.1.
EXPERIMENTS: Dict[str, dict] = {
    "exp1": {
        "run_dir": "exp1_swin_cls",
        "csv": "test_predictions.csv",
        "prob_col": "prob_malignant",
        "locked_thr": 0.450,
    },
    "exp2": {
        "run_dir": "exp2_swin_mag_cls",
        "csv": "test_predictions.csv",
        "prob_col": "prob_malignant",
        "locked_thr": 0.428,
    },
    "exp3": {
        "run_dir": "exp3_swin_mag_supcon_cls",
        "csv": "test_predictions.csv",
        "prob_col": "prob_malignant",
        "locked_thr": 0.360,
    },
    "exp3n": {
        "run_dir": "exp3n_swin_supcon_cls",
        "csv": "test_predictions.csv",
        "prob_col": "prob_malignant",
        "locked_thr": 0.540,
    },
    "exp5_retrieval": {
        "run_dir": "exp3n_swin_supcon_cls",
        "csv": "test_predictions_retrieval.csv",
        "prob_col": "prob_final",
        "locked_thr": 0.6923,
    },
    "expfm": {
        "run_dir": "expfm_ctranspath_linear",
        "csv": "test_predictions.csv",
        "prob_col": "prob_malignant",
        "locked_thr": 0.209,
    },
}

# Paired comparisons the paper actually makes. Everything is measured against the
# Swin+FPN baseline, which is the question the paper asks; the last two are the
# within-ladder removal and the retrieval delta.
COMPARISONS: List[Tuple[str, str]] = [
    ("exp2", "exp1"),
    ("exp3", "exp1"),
    ("exp3n", "exp1"),
    ("expfm", "exp1"),
    ("exp3n", "exp3"),
    ("exp5_retrieval", "exp3n"),
]

N_FOLDS = 5


def latest_fold_run(run_dir: Path, fold: int) -> Path:
    """Newest run directory for a fold. Runs are timestamped and never overwritten."""
    matches = sorted(run_dir.glob(f"*_train_fold{fold}"))
    if not matches:
        raise FileNotFoundError(f"no run directory for fold {fold} under {run_dir}")
    return matches[-1]


def load_ensemble(spec: dict) -> pd.DataFrame:
    """Fold-ensemble predictions: mean probability over the five fold models.

    Returns one row per test image with columns patient_id, label, prob.
    """
    run_dir = REPO / "runs" / spec["run_dir"]
    frames = []
    for fold in range(N_FOLDS):
        path = latest_fold_run(run_dir, fold) / "test" / spec["csv"]
        df = pd.read_csv(path, usecols=["image_path", "patient_id", "label", spec["prob_col"]])
        df = df.rename(columns={spec["prob_col"]: f"prob_{fold}"}).set_index("image_path")
        frames.append(df)

    merged = frames[0][["patient_id", "label", "prob_0"]]
    for fold, df in enumerate(frames[1:], start=1):
        # An inner join would silently hide a fold with missing rows; assert instead.
        if not merged.index.equals(df.index):
            raise ValueError(f"{spec['run_dir']} fold {fold}: test image set differs from fold 0")
        if not (merged["label"].to_numpy() == df["label"].to_numpy()).all():
            raise ValueError(f"{spec['run_dir']} fold {fold}: labels differ from fold 0")
        merged[f"prob_{fold}"] = df[f"prob_{fold}"]

    prob_cols = [f"prob_{f}" for f in range(N_FOLDS)]
    out = merged[["patient_id", "label"]].copy()
    out["prob"] = merged[prob_cols].to_numpy().mean(axis=1)
    return out.reset_index(drop=True)


def patient_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Mean-pooled malignant probability per patient, per the evaluation protocol."""
    g = df.groupby("patient_id", sort=True)
    return pd.DataFrame({"label": g["label"].first(), "prob": g["prob"].mean()})


def metrics(labels: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, float]:
    """AUC and accuracy at a fixed threshold. AUC is nan if a resample is single-class."""
    single_class = len(np.unique(labels)) < 2
    return {
        "auc": float("nan") if single_class else float(roc_auc_score(labels, probs)),
        "acc": float(((probs >= thr).astype(int) == labels).mean()),
    }


def evaluate(df: pd.DataFrame, pat: pd.DataFrame, thr: float) -> Dict[str, float]:
    img = metrics(df["label"].to_numpy(), df["prob"].to_numpy(), thr)
    # The patient level is reported at 0.5 throughout the repo (README 9.1); the
    # locked patient thresholds are a separate, documented artefact.
    ppat = metrics(pat["label"].to_numpy(), pat["prob"].to_numpy(), 0.5)
    return {
        "image_auc": img["auc"],
        "image_acc": img["acc"],
        "patient_auc": ppat["auc"],
        "patient_acc": ppat["acc"],
    }


def ci(samples: Sequence[float]) -> Dict[str, Optional[float]]:
    """Percentile interval, plus the fraction of resamples above zero."""
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "lo": None, "hi": None, "p_gt_0": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "lo": float(np.percentile(arr, 2.5)),
        "hi": float(np.percentile(arr, 97.5)),
        "p_gt_0": float((arr > 0).mean()),
        "n": int(arr.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO / "analysis" / "bootstrap_benchmark.json")
    args = ap.parse_args()

    print("Loading fold-ensemble predictions ...")
    data: Dict[str, pd.DataFrame] = {}
    for name, spec in EXPERIMENTS.items():
        data[name] = load_ensemble(spec)
        print(f"  {name:<16} {len(data[name]):>5} images  "
              f"{data[name]['patient_id'].nunique():>3} patients  thr={spec['locked_thr']}")

    # Every model must be scored on the identical image set in the identical order,
    # or the paired deltas are meaningless.
    reference = data["exp1"]
    for name, df in data.items():
        if not df["patient_id"].equals(reference["patient_id"]):
            raise ValueError(f"{name}: patient ordering differs from exp1")
        if not df["label"].equals(reference["label"]):
            raise ValueError(f"{name}: labels differ from exp1")

    patients = np.array(sorted(reference["patient_id"].unique()))
    n_pat = len(patients)
    print(f"\nHeld-out set: {len(reference)} images / {n_pat} patients "
          f"({int((patient_pool(reference)['label'] == 0).sum())} benign / "
          f"{int((patient_pool(reference)['label'] == 1).sum())} malignant)")

    # Point estimates on the full test set.
    point = {}
    for name, df in data.items():
        point[name] = evaluate(df, patient_pool(df), EXPERIMENTS[name]["locked_thr"])

    # Pre-index rows by patient so each resample is a cheap concatenation.
    idx_by_patient = {p: np.flatnonzero((reference["patient_id"] == p).to_numpy())
                      for p in patients}

    print(f"\nBootstrapping: {args.resamples} resamples of {n_pat} patients "
          f"(seed {args.seed}) ...")
    rng = np.random.default_rng(args.seed)
    abs_samples = {name: {k: [] for k in point[name]} for name in data}
    delta_samples = {f"{a}-{b}": {k: [] for k in point[a]} for a, b in COMPARISONS}
    degenerate = 0

    for it in range(args.resamples):
        drawn = rng.choice(patients, size=n_pat, replace=True)
        rows = np.concatenate([idx_by_patient[p] for p in drawn])
        # Patients drawn twice must count twice at the patient level too, so the
        # patient-level vector is built from the draw, not from a groupby.
        if len(np.unique(reference["label"].to_numpy()[rows])) < 2:
            degenerate += 1

        per_model = {}
        for name, df in data.items():
            lab = df["label"].to_numpy()[rows]
            prob = df["prob"].to_numpy()[rows]
            thr = EXPERIMENTS[name]["locked_thr"]
            img = metrics(lab, prob, thr)

            pat_lab, pat_prob = [], []
            for p in drawn:
                ridx = idx_by_patient[p]
                pat_lab.append(df["label"].to_numpy()[ridx][0])
                pat_prob.append(df["prob"].to_numpy()[ridx].mean())
            pat = metrics(np.asarray(pat_lab), np.asarray(pat_prob), 0.5)

            m = {"image_auc": img["auc"], "image_acc": img["acc"],
                 "patient_auc": pat["auc"], "patient_acc": pat["acc"]}
            per_model[name] = m
            for k, v in m.items():
                abs_samples[name][k].append(v)

        for a, b in COMPARISONS:
            for k in per_model[a]:
                delta_samples[f"{a}-{b}"][k].append(per_model[a][k] - per_model[b][k])

        if (it + 1) % 500 == 0:
            print(f"  {it + 1}/{args.resamples}")

    result = {
        "protocol": {
            "resamples": args.resamples,
            "seed": args.seed,
            "cluster": "patient",
            "n_patients": int(n_pat),
            "n_images": int(len(reference)),
            "aggregate": "fold ensemble (mean probability over 5 fold models)",
            "thresholds": {n: s["locked_thr"] for n, s in EXPERIMENTS.items()},
            "patient_threshold": 0.5,
            "single_class_resamples": degenerate,
        },
        "point_estimates": point,
        "absolute_ci": {n: {k: ci(v) for k, v in d.items()} for n, d in abs_samples.items()},
        "paired_deltas": {n: {k: ci(v) for k, v in d.items()} for n, d in delta_samples.items()},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf8")

    print(f"\nSingle-class resamples (AUC undefined, excluded): {degenerate}")
    print("\n=== Point estimates, fold ensemble, locked thresholds ===")
    print(f"{'model':<16}{'img AUC':>10}{'img acc':>10}{'pat AUC':>10}{'pat acc':>10}")
    for name, m in point.items():
        print(f"{name:<16}{m['image_auc']:>10.4f}{m['image_acc']:>10.4f}"
              f"{m['patient_auc']:>10.4f}{m['patient_acc']:>10.4f}")

    print("\n=== Paired deltas, 95% patient-clustered bootstrap CI ===")
    for key, d in result["paired_deltas"].items():
        print(f"\n  {key}")
        for metric in ("image_auc", "image_acc", "patient_acc"):
            c = d[metric]
            if c["mean"] is None:
                print(f"    {metric:<12} undefined")
                continue
            crosses = "" if (c["lo"] > 0 or c["hi"] < 0) else "   (straddles 0)"
            print(f"    {metric:<12} {c['mean']:+.4f}  "
                  f"[{c['lo']:+.4f}, {c['hi']:+.4f}]  P(d>0)={c['p_gt_0']:.2f}{crosses}")

    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
