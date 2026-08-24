"""Retrieval-design probe #3 — does the memory need a *learned* metric, and does a
two-level (image + slide) memory beat either level alone?

Probe #1 fixed the key space (1024-d pre-magnification vector).
Probe #2 showed (a) retrieval rescues ~32% of exp3's parametric errors,
(b) same-magnification routing is best, (c) inverse-frequency "subtype-balanced"
voting HURTS, (d) image-level kNN and patient-level prototypes fail on *different*
hard cases (kNN fixes DC-12312, patient-level fixes TA-16184 and PC-9146).

This probe answers the two design questions that remain:

    H1  Metric.  Is raw cosine on the backbone vector the right similarity, or does
        the memory need a learned re-projection? Four metrics are compared under
        strict leave-one-patient-out (fitted on the BANK only — the query patient's
        labels are never seen):
            cosine            — the current proposal
            pca_whiten        — unsupervised whitening (standard retrieval trick)
            lda_binary        — supervised by benign/malignant (control)
            lda_subtype       — supervised by the 8 tumour subtypes  <-- the claim
        If lda_subtype > lda_binary > cosine, a subtype-supervised metric is
        justified by evidence rather than by intuition.

    H2  Granularity.  Grid over a 3-way blend of {parametric, image-kNN,
        slide-prototype} to see whether the two memory levels are complementary.

Usage
-----
    python scripts/retrieval_probe3.py --experiments exp1 exp3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_probe import (  # noqa: E402
    HARD_PATIENTS, l2, blocked_sim, topk, vote, patient_acc, sens_spec, auc,
)
from retrieval_probe2 import primary_key, patient_level_vote  # noqa: E402


# --------------------------------------------------------------------------- #
# H1 -- metric comparison under leave-one-patient-out
# --------------------------------------------------------------------------- #
def _fit_transform(kind: str, bank: np.ndarray, bank_lab: np.ndarray,
                   query: np.ndarray, n_pca: int = 128):
    """Fit a metric on the bank only, return (bank', query')."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    if kind == "cosine":
        return bank, query
    if kind == "pca_whiten":
        p = PCA(n_components=min(n_pca, bank.shape[0] - 1, bank.shape[1]),
                whiten=True, random_state=0).fit(bank)
        return p.transform(bank), p.transform(query)
    # supervised: PCA first (1024-d with ~1500 samples is ill-conditioned for LDA)
    p = PCA(n_components=min(n_pca, bank.shape[0] - 1, bank.shape[1]),
            random_state=0).fit(bank)
    b, q = p.transform(bank), p.transform(query)
    classes, counts = np.unique(bank_lab, return_counts=True)
    keep = np.isin(bank_lab, classes[counts >= 10])
    if len(np.unique(bank_lab[keep])) < 2:
        return b, q
    lda = LDA(solver="eigen", shrinkage="auto").fit(b[keep], bank_lab[keep])
    return lda.transform(b), lda.transform(q)


def metric_study(vecs, labels, subtypes, pids, k=15) -> Dict[str, dict]:
    out = {}
    for kind in ("cosine", "pca_whiten", "lda_binary", "lda_subtype"):
        probs = np.zeros(len(vecs))
        nb_same_sub = []
        for p in np.unique(pids):
            q = pids == p
            bank = ~q
            lab_for_fit = labels[bank] if kind != "lda_subtype" else subtypes[bank]
            try:
                B, Q = _fit_transform(kind, vecs[bank].astype(np.float64),
                                      lab_for_fit, vecs[q].astype(np.float64))
            except Exception:
                B, Q = vecs[bank].astype(np.float64), vecs[q].astype(np.float64)
            B, Q = l2(B), l2(Q)
            s = Q @ B.T
            kk = min(k, B.shape[0] - 1)
            idx = np.argpartition(-s, kth=kk, axis=1)[:, :kk]
            rows = np.arange(s.shape[0])[:, None]
            sv = s[rows, idx]
            w = np.exp((sv - sv.max(axis=1, keepdims=True)) / 0.07)
            bl = labels[bank][idx]
            probs[q] = (w * bl).sum(1) / (w.sum(1) + 1e-12)
            nb_same_sub.append(float((subtypes[bank][idx] == subtypes[q][:, None]).mean()))
        se, sp = sens_spec(probs, labels)
        row = {
            "img_acc": round(float(((probs >= .5).astype(int) == labels).mean()), 4),
            "auc": round(auc(probs, labels), 4),
            "patient_acc": round(patient_acc(probs, labels, pids), 4),
            "sens": round(se, 4), "spec": round(sp, 4),
            "neighbour_subtype_purity": round(float(np.mean(nb_same_sub)), 4),
        }
        for hp in HARD_PATIENTS:
            m = pids == hp
            if m.any():
                row[f"hp::{hp}"] = round(float(probs[m].mean()), 4)
        out[kind] = row
    return out


# --------------------------------------------------------------------------- #
# H2 -- two-level memory: parametric + image-kNN + slide-prototype
# --------------------------------------------------------------------------- #
def two_level(vecs, labels, pids, param_prob, k=15, k_pat=5) -> List[dict]:
    sim = blocked_sim(vecs, pids)
    nn = topk(sim, k)
    img_p = vote(sim, nn, labels, "softmax")
    sld_p = patient_level_vote(vecs, labels, pids, k_pat=k_pat)
    rows = []
    grid = np.round(np.arange(0, 1.01, 0.1), 2)
    for wp in grid:
        for wi in np.round(np.arange(0, 1.01 - wp + 1e-9, 0.1), 2):
            ws = round(1.0 - wp - wi, 2)
            if ws < -1e-9:
                continue
            f = wp * param_prob + wi * img_p + ws * sld_p
            se, sp = sens_spec(f, labels)
            r = {"w_param": float(wp), "w_img": float(wi), "w_slide": float(ws),
                 "img_acc": round(float(((f >= .5).astype(int) == labels).mean()), 4),
                 "auc": round(auc(f, labels), 4),
                 "patient_acc": round(patient_acc(f, labels, pids), 4),
                 "sens": round(se, 4), "spec": round(sp, 4)}
            for hp in HARD_PATIENTS:
                m = pids == hp
                if m.any():
                    r[f"hp::{hp}"] = round(float(f[m].mean()), 4)
            rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default="analysis/embeddings/embeddings")
    ap.add_argument("--out", default="analysis/retrieval_probe")
    ap.add_argument("--experiments", nargs="+", default=["exp1", "exp3"])
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    emb_dir, out_dir = Path(args.emb_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, dict] = {}

    for exp in args.experiments:
        files = sorted(emb_dir.glob(f"{exp}_fold*.npz"))
        if not files:
            continue
        met_rows, tl_rows = [], []
        for f in files:
            d = np.load(f, allow_pickle=True)
            labels, pids = d["label"].astype(int), d["patient_id"]
            subs, prob = d["subtype"], d["prob"]
            v = primary_key(d)
            met_rows.append(metric_study(v, labels, subs, pids, args.k))
            tl_rows.append(two_level(v, labels, pids, prob, args.k))
            print(f"  {exp} {f.stem} done", flush=True)

        def mnum(ds):
            return {k: round(float(np.mean([d[k] for d in ds])), 4)
                    for k in ds[0] if isinstance(ds[0][k], (int, float))}

        report[exp] = {
            "metrics": {m: mnum([r[m] for r in met_rows]) for m in met_rows[0]},
            "two_level": [
                {**{k: tl_rows[0][i][k] for k in ("w_param", "w_img", "w_slide")},
                 **mnum([L[i] for L in tl_rows])}
                for i in range(len(tl_rows[0]))],
        }

    with (out_dir / "probe3.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    for exp, R in report.items():
        print("\n" + "=" * 84)
        print(f"{exp.upper()}")
        print("-" * 84)
        print("  H1) retrieval metric (LOPO, fitted on bank only):")
        print(f"     {'metric':<14}{'img':>8}{'auc':>8}{'pat':>8}{'sens':>8}{'spec':>8}"
              f"{'nbrSubPurity':>14}")
        for m, v in R["metrics"].items():
            print(f"     {m:<14}{v['img_acc']:>8.4f}{v['auc']:>8.4f}{v['patient_acc']:>8.4f}"
                  f"{v['sens']:>8.3f}{v['spec']:>8.3f}{v['neighbour_subtype_purity']:>14.4f}")
        print("     hard patients (y: 12312=1 9146=1 16184=0 16456=1 20636=1):")
        for m, v in R["metrics"].items():
            hp = "  ".join(f"{k.split('-')[-1]}={v[k]:.2f}" for k in v if k.startswith("hp::"))
            print(f"       {m:<14} {hp}")
        print("\n  H2) two-level blend — top 8 by patient acc then img acc:")
        top = sorted(R["two_level"], key=lambda r: (-r["patient_acc"], -r["img_acc"]))[:8]
        for r in top:
            hp = "  ".join(f"{k.split('-')[-1]}={r[k]:.2f}" for k in r if k.startswith("hp::"))
            print(f"     param={r['w_param']:.1f} img={r['w_img']:.1f} slide={r['w_slide']:.1f} "
                  f"| img={r['img_acc']:.4f} auc={r['auc']:.4f} pat={r['patient_acc']:.4f} "
                  f"se={r['sens']:.3f} sp={r['spec']:.3f}")
            print(f"        {hp}")
        print("\n  H2b) reference corners:")
        for r in R["two_level"]:
            c = (r["w_param"], r["w_img"], r["w_slide"])
            if c in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                     (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.4, 0.3, 0.3)]:
                print(f"     param={c[0]:.1f} img={c[1]:.1f} slide={c[2]:.1f} "
                      f"| img={r['img_acc']:.4f} auc={r['auc']:.4f} "
                      f"pat={r['patient_acc']:.4f} se={r['sens']:.3f} sp={r['spec']:.3f}")
    print(f"\nWrote {out_dir/'probe3.json'}")


if __name__ == "__main__":
    main()
