"""Retrieval probe 5 -- why the memory module does not help exp3n.

Evidence behind ``docs/retrieval.md`` Part VII. Reads the built banks directly (no
torch, no model, no checkpoints -- only numpy) and answers three questions:

  1. How anisotropic is the ``features`` key space?  (effective rank via the
     participation ratio of the covariance spectrum, dominant-axis variance share,
     norm of the mean unit key, random-pair vs nearest-neighbour cosine)
  2. Is the ~0.999 top-1 similarity a real match or the floor of a collapsed cone?
  3. Does the class signal that retrieval reads generalise to held-out patients?
     -> leave-patient-out kNN vote AUC measured on TRAIN queries (in-distribution)
        vs the val AUC the gate already reports (out-of-distribution).

Run: ``python scripts/retrieval_probe5.py``
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

BANK_GLOB = str(Path(__file__).resolve().parents[1] / "analysis" / "retrieval" / "exp3n" / "bank_fold*.npz")
T = 0.07          # D4 vote temperature, matches rap_mst/retrieval config
K = 15            # image-view k
rng = np.random.default_rng(0)


def load_image_rows(path):
    d = np.load(path, allow_pickle=False)
    lv = d["levels"].astype(str)
    img = lv == "image"
    keys = d["keys"][img].astype(np.float64)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True) + 1e-12
    return keys, d["labels"][img].astype(int), d["patient_ids"][img].astype(str), d["mag_index"][img].astype(int)


def effective_rank(keys):
    """Participation ratio (sum l)^2 / sum l^2 of the centred covariance spectrum."""
    centred = keys - keys.mean(0, keepdims=True)
    sv = np.linalg.svd(centred, compute_uv=False)
    lam = sv ** 2
    return (lam.sum() ** 2) / (lam ** 2).sum(), lam / lam.sum()


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def probe(path):
    keys, y, pid, mag = load_image_rows(path)
    n = len(keys)

    sample = rng.choice(n, size=min(4000, n), replace=False)
    sim = keys[sample] @ keys[sample].T
    rand_pair = sim[np.triu_indices(len(sample), k=1)]
    pr, spectrum = effective_rank(keys)

    # leave-patient-out kNN vote on TRAIN queries, mirroring route=same_mag.
    qsel = rng.choice(n, size=min(2000, n), replace=False)
    nn_cos, votes, ytrue = [], [], []
    for qi in qsel:
        cand = np.where((mag == mag[qi]) & (pid != pid[qi]))[0]
        sims = keys[cand] @ keys[qi]
        top = cand[np.argsort(-sims)[:K]]
        tsim = np.sort(sims)[::-1][:K]
        nn_cos.append(tsim[0])
        w = np.exp((tsim - tsim.max()) / T)
        votes.append(float((w * y[top]).sum() / w.sum()))
        ytrue.append(y[qi])
    nn_cos, votes, ytrue = map(np.asarray, (nn_cos, votes, ytrue))

    return dict(
        fold=Path(path).stem.replace("bank_fold", ""),
        n_image_rows=n,
        effective_rank=round(float(pr), 3),
        dominant_axis_variance_share=round(float(spectrum[0]), 4),
        mean_unit_key_norm=round(float(np.linalg.norm(keys.mean(0))), 4),
        random_pair_cos_mean=round(float(rand_pair.mean()), 4),
        nn_cos_mean=round(float(nn_cos.mean()), 4),
        train_knn_vote_auc=round(float(auc(votes, ytrue)), 4),
    )


def main():
    banks = sorted(glob.glob(BANK_GLOB))
    if not banks:
        raise SystemExit(f"No banks found at {BANK_GLOB} -- run scripts/build_memory_bank.py first.")
    rows = [probe(p) for p in banks]
    for r in rows:
        print(json.dumps(r))
    print("\n==== AGGREGATE (mean over folds) ====")
    for key in ("effective_rank", "dominant_axis_variance_share", "mean_unit_key_norm",
                "random_pair_cos_mean", "nn_cos_mean", "train_knn_vote_auc"):
        print(f"  {key:30s}: {np.mean([r[key] for r in rows]):.4f}")
    print("\nCompare train_knn_vote_auc (in-distribution) with the gate's val "
          "prob_img image_auc (out-of-distribution, ~0.868) and prob_param (~0.885).")


if __name__ == "__main__":
    main()
