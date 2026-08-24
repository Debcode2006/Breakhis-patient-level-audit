"""Retrieval-design probe — measures, on the saved test-set embeddings, every
property a Retrieval Memory module's architecture depends on.

Why this exists
---------------
`docs/results/classifier_ladder.md` §8 proposed a retrieval module from *error analysis*; `docs/results/embedding_geometry.md`
then showed the indexed space is locally class-pure (kNN 0.898) but subtype-collapsed
(subtype silhouette negative). Neither answers the *design* questions:

    Q1  Which vector should be the key -- embeddings (1088), the pre-mag slice (1024),
        or the SupCon projections (128)?
    Q2  Does the 64-d magnification block concatenated into `embeddings` dominate
        cosine similarity (i.e. is retrieval magnification-locked rather than
        morphology-locked)?
    Q3  Does a neighbourhood vote actually fix the patients the parametric head
        misses (DC-12312, PC-9146, TA-16184), or does it miss the same ones?
    Q4  Is a confidence gate needed, and where should the gate/blend sit?
    Q5  Are neighbourhoods subtype-pure? (decides whether a subtype-aware term is
        required, and whether rare subtypes get outvoted by the DC mass)
    Q6  Is the space hubby? (decides whether hubness correction / CSLS is needed)
    Q7  kNN vote vs class prototypes vs subtype prototypes vs patient prototypes
        (decides retrieval-only, prototype-only, or hybrid)

Protocol / caveat
-----------------
Bank = the *other* test patients (leave-one-patient-out). The deployed module will
index the 66 CV-training patients, a far richer bank, so every number here is a
LOWER BOUND on retrieval quality -- but the *relative* comparisons (which space,
which vote rule, gated vs ungated) are the design-relevant signal and transfer.
Same-patient neighbours are always blocked, so slide near-duplicates cannot cheat.

Usage
-----
    python scripts/retrieval_probe.py                     # all exps, all folds
    python scripts/retrieval_probe.py --experiments exp3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

BENIGN_SUBTYPES = ["A", "F", "TA", "PT"]
MALIGNANT_SUBTYPES = ["DC", "LC", "MC", "PC"]
HARD_PATIENTS = [
    "SOB_M_DC-14-12312",   # low-grade ductal -- hardest false negative
    "SOB_M_PC-14-9146",    # papillary, rare subtype -- false negative
    "SOB_B_TA-14-16184",   # tubular adenoma -- hardest false positive
    "SOB_M_MC-14-16456",   # mucinous, borderline under exp3
    "SOB_M_DC-14-20636",   # borderline malignant under exp3
]


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


# --------------------------------------------------------------------------- #
# Key spaces
# --------------------------------------------------------------------------- #
def key_spaces(d: Dict[str, np.ndarray], exp: str) -> Dict[str, np.ndarray]:
    """The candidate index keys. For exp2/exp3 `embeddings` is concat(fused_1024,
    mag_embed_64); `emb_nomag` strips the magnification block."""
    spaces = {"embeddings": d["embeddings"]}
    if d["embeddings"].shape[1] > 1024:
        spaces["emb_nomag"] = d["embeddings"][:, :1024]
        spaces["mag_block"] = d["embeddings"][:, 1024:]
    if "projections" in d and d["projections"].size:
        spaces["projections"] = d["projections"]
    return spaces


# --------------------------------------------------------------------------- #
# Core retrieval machinery (patient-blocked)
# --------------------------------------------------------------------------- #
def blocked_sim(vecs: np.ndarray, patient_ids: np.ndarray) -> np.ndarray:
    x = l2(vecs.astype(np.float64))
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    sim[patient_ids[:, None] == patient_ids[None, :]] = -np.inf
    return sim


def topk(sim: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    order = np.argsort(-sim[rows, idx], axis=1)
    return idx[rows, order]


def vote(sim: np.ndarray, nn_idx: np.ndarray, labels: np.ndarray,
         mode: str = "softmax", temp: float = 0.07,
         weights: np.ndarray | None = None,
         row_ids: np.ndarray | None = None) -> np.ndarray:
    """P(malignant) from the neighbourhood. `weights` optionally re-weights each
    bank item (used for the subtype-balanced / rare-subtype-upweighted variants).
    `row_ids` names the query rows when `nn_idx` covers only a subset of them."""
    rows = (np.arange(sim.shape[0]) if row_ids is None else row_ids)[:, None]
    s = sim[rows, nn_idx]
    lab = labels[nn_idx]
    if mode == "uniform":
        w = np.ones_like(s)
    else:  # similarity softmax
        w = np.exp((s - s.max(axis=1, keepdims=True)) / temp)
    if weights is not None:
        w = w * weights[nn_idx]
    return (w * lab).sum(1) / (w.sum(1) + 1e-12)


def patient_acc(prob: np.ndarray, labels: np.ndarray, pids: np.ndarray,
                thr: float = 0.5) -> float:
    ok = 0
    uniq = np.unique(pids)
    for p in uniq:
        m = pids == p
        ok += int((prob[m].mean() >= thr) == bool(labels[m][0]))
    return ok / len(uniq)


def sens_spec(prob: np.ndarray, labels: np.ndarray, thr: float = 0.5):
    pred = (prob >= thr).astype(int)
    mal, ben = labels == 1, labels == 0
    return float((pred[mal] == 1).mean()), float((pred[ben] == 0).mean())


def auc(prob: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(prob)
    ranks = np.empty(len(prob), float)
    ranks[order] = np.arange(1, len(prob) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(prob, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    npos, nneg = int((labels == 1).sum()), int((labels == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# --------------------------------------------------------------------------- #
# Q1/Q2 -- which key space, and is it magnification-locked?
# --------------------------------------------------------------------------- #
def probe_space(vecs, labels, subtypes, pids, mags, k=15) -> Dict[str, float]:
    sim = blocked_sim(vecs, pids)
    nn = topk(sim, k)
    p = vote(sim, nn, labels, "softmax")
    same_mag = (mags[nn] == mags[:, None]).mean()
    same_sub = (subtypes[nn] == subtypes[:, None]).mean()
    # chance rates for the same bank composition
    chance_mag = float(np.mean([(mags == m).mean() for m in mags]))
    chance_sub = float(np.mean([(subtypes == s).mean() for s in subtypes]))
    return {
        "knn_acc": float(((p >= 0.5).astype(int) == labels).mean()),
        "knn_auc": auc(p, labels),
        "knn_patient_acc": patient_acc(p, labels, pids),
        "same_mag_rate": float(same_mag),
        "same_mag_chance": chance_mag,
        "mag_lock_lift": float(same_mag) - chance_mag,
        "same_subtype_rate": float(same_sub),
        "same_subtype_chance": chance_sub,
        "subtype_lift": float(same_sub) - chance_sub,
    }


# --------------------------------------------------------------------------- #
# Q5 -- per-subtype neighbourhood purity + who the neighbours actually are
# --------------------------------------------------------------------------- #
def subtype_purity(vecs, labels, subtypes, pids, k=15) -> Dict[str, dict]:
    sim = blocked_sim(vecs, pids)
    nn = topk(sim, k)
    out = {}
    for st in BENIGN_SUBTYPES + MALIGNANT_SUBTYPES:
        m = subtypes == st
        if not m.any():
            continue
        nb = subtypes[nn[m]]
        same = float((nb == st).mean())
        # what dominates the neighbourhood instead
        vals, cnts = np.unique(nb, return_counts=True)
        top = vals[np.argsort(-cnts)][:3].tolist()
        frac = (np.sort(cnts)[::-1][:3] / cnts.sum()).round(3).tolist()
        p = vote(sim, nn[m], labels, "softmax", row_ids=np.flatnonzero(m))
        out[st] = {
            "n_images": int(m.sum()),
            "n_patients": int(len(np.unique(pids[m]))),
            "same_subtype_precision_at_k": round(same, 4),
            "binary_knn_acc": round(float(((p >= 0.5).astype(int) == labels[m]).mean()), 4),
            "top_neighbour_subtypes": top,
            "top_neighbour_fracs": frac,
        }
    return out


# --------------------------------------------------------------------------- #
# Q6 -- hubness
# --------------------------------------------------------------------------- #
def hubness(vecs, pids, k=15) -> Dict[str, float]:
    sim = blocked_sim(vecs, pids)
    nn = topk(sim, k)
    counts = np.bincount(nn.ravel(), minlength=len(pids)).astype(float)
    mu, sd = counts.mean(), counts.std() + 1e-12
    skew = float((((counts - mu) / sd) ** 3).mean())
    order = np.sort(counts)[::-1]
    return {
        "Nk_skewness": skew,
        "never_retrieved_frac": float((counts == 0).mean()),
        "top1pct_share": float(order[: max(1, len(order) // 100)].sum() / counts.sum()),
        "top10pct_share": float(order[: max(1, len(order) // 10)].sum() / counts.sum()),
    }


# --------------------------------------------------------------------------- #
# Q7 -- kNN vs prototypes
# --------------------------------------------------------------------------- #
def prototype_variants(vecs, labels, subtypes, pids, k=15) -> Dict[str, dict]:
    """Leave-one-patient-out prototypes at three granularities."""
    x = l2(vecs.astype(np.float64))
    res = {}

    def score(build_protos):
        probs = np.zeros(len(x))
        for p in np.unique(pids):
            q = pids == p
            bank = ~q
            protos, plabs = build_protos(x[bank], labels[bank], subtypes[bank])
            if protos is None or len(protos) < 2:
                probs[q] = 0.5
                continue
            s = x[q] @ l2(protos).T
            # softmax over prototypes, sum mass on malignant prototypes
            e = np.exp((s - s.max(axis=1, keepdims=True)) / 0.07)
            e /= e.sum(1, keepdims=True)
            probs[q] = e[:, plabs == 1].sum(1)
        return probs

    def binary_protos(v, l, s):
        return np.stack([v[l == c].mean(0) for c in (0, 1)]), np.array([0, 1])

    def subtype_protos(v, l, s):
        sts = [t for t in np.unique(s) if (s == t).sum() >= 5]
        if len(sts) < 2:
            return None, None
        P = np.stack([v[s == t].mean(0) for t in sts])
        L = np.array([1 if t in MALIGNANT_SUBTYPES else 0 for t in sts])
        return P, L

    for name, fn in [("binary_prototype", binary_protos),
                     ("subtype_prototype", subtype_protos)]:
        pr = score(fn)
        res[name] = {
            "img_acc": round(float(((pr >= 0.5).astype(int) == labels).mean()), 4),
            "auc": round(auc(pr, labels), 4),
            "patient_acc": round(patient_acc(pr, labels, pids), 4),
        }

    sim = blocked_sim(vecs, pids)
    nn = topk(sim, k)
    for name, mode in [("knn_uniform", "uniform"), ("knn_softmax", "softmax")]:
        pr = vote(sim, nn, labels, mode)
        res[name] = {
            "img_acc": round(float(((pr >= 0.5).astype(int) == labels).mean()), 4),
            "auc": round(auc(pr, labels), 4),
            "patient_acc": round(patient_acc(pr, labels, pids), 4),
        }
    return res


# --------------------------------------------------------------------------- #
# Q3/Q4 -- does retrieval fix the misses, and does gating help?
# --------------------------------------------------------------------------- #
def fusion_sweep(vecs, labels, pids, param_prob, k_list, alphas, taus) -> List[dict]:
    sim = blocked_sim(vecs, pids)
    rows = []
    for k in k_list:
        nn = topk(sim, k)
        rp = vote(sim, nn, labels, "softmax")
        for a in alphas:
            for t in taus:
                gate = np.abs(param_prob - 0.5) < t      # uncertain -> consult memory
                fused = param_prob.copy()
                fused[gate] = a * param_prob[gate] + (1 - a) * rp[gate]
                se, sp = sens_spec(fused, labels)
                rows.append({
                    "k": k, "alpha": a, "tau": t,
                    "gated_frac": round(float(gate.mean()), 4),
                    "img_acc": round(float(((fused >= 0.5).astype(int) == labels).mean()), 4),
                    "auc": round(auc(fused, labels), 4),
                    "patient_acc": round(patient_acc(fused, labels, pids), 4),
                    "sens": round(se, 4), "spec": round(sp, 4),
                })
    return rows


def per_patient(vecs, labels, pids, param_prob, k=15) -> Dict[str, dict]:
    sim = blocked_sim(vecs, pids)
    nn = topk(sim, k)
    rp = vote(sim, nn, labels, "softmax")
    out = {}
    for p in np.unique(pids):
        m = pids == p
        out[str(p)] = {
            "label": int(labels[m][0]),
            "n": int(m.sum()),
            "param_mean_prob": round(float(param_prob[m].mean()), 4),
            "retrieval_mean_prob": round(float(rp[m].mean()), 4),
            "param_img_acc": round(float(((param_prob[m] >= .5).astype(int) == labels[m]).mean()), 4),
            "retrieval_img_acc": round(float(((rp[m] >= .5).astype(int) == labels[m]).mean()), 4),
        }
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default="analysis/embeddings/embeddings")
    ap.add_argument("--out", default="analysis/retrieval_probe")
    ap.add_argument("--experiments", nargs="+", default=["exp1", "exp2", "exp3"])
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    emb_dir, out_dir = Path(args.emb_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, dict] = {}

    for exp in args.experiments:
        files = sorted(emb_dir.glob(f"{exp}_fold*.npz"))
        if not files:
            continue
        per_fold: Dict[str, List[dict]] = {}
        fusion_rows: Dict[tuple, List[dict]] = {}
        patient_rows: List[dict] = []
        subtype_rows: List[dict] = []
        hub_rows: List[dict] = []
        proto_rows: List[dict] = []

        for f in files:
            d = np.load(f, allow_pickle=True)
            labels, pids = d["label"].astype(int), d["patient_id"]
            subs, mags, prob = d["subtype"], d["magnification"], d["prob"]
            spaces = key_spaces(d, exp)

            for name, v in spaces.items():
                per_fold.setdefault(name, []).append(
                    probe_space(v, labels, subs, pids, mags, k=args.k))

            # design probes run on the *primary* key space for this exp
            primary = "emb_nomag" if "emb_nomag" in spaces else "embeddings"
            v = spaces[primary]
            subtype_rows.append(subtype_purity(v, labels, subs, pids, args.k))
            hub_rows.append(hubness(v, pids, args.k))
            proto_rows.append(prototype_variants(v, labels, subs, pids, args.k))
            patient_rows.append(per_patient(v, labels, pids, prob, args.k))
            for r in fusion_sweep(v, labels, pids, prob,
                                  k_list=[5, 15, 30, 50],
                                  alphas=[0.0, 0.3, 0.5, 0.7, 1.0],
                                  taus=[0.05, 0.10, 0.20, 0.50]):
                fusion_rows.setdefault((r["k"], r["alpha"], r["tau"]), []).append(r)

        def mean_of(rows: List[dict]) -> dict:
            return {k: round(float(np.nanmean([r[k] for r in rows])), 4) for k in rows[0]}

        report[exp] = {
            "spaces": {n: mean_of(rs) for n, rs in per_fold.items()},
            "hubness": mean_of(hub_rows),
            "prototypes_vs_knn": {
                m: {k: round(float(np.mean([r[m][k] for r in proto_rows])), 4)
                    for k in proto_rows[0][m]} for m in proto_rows[0]},
            "subtype_purity": {
                st: {k: (round(float(np.mean([r[st][k] for r in subtype_rows])), 4)
                         if isinstance(subtype_rows[0][st][k], (int, float)) else
                         subtype_rows[0][st][k])
                     for k in subtype_rows[0][st]}
                for st in subtype_rows[0]},
            "per_patient": {
                p: {k: round(float(np.mean([r[p][k] for r in patient_rows])), 4)
                    for k in patient_rows[0][p]} for p in patient_rows[0]},
            "fusion_sweep": sorted(
                [mean_of(rs) for rs in fusion_rows.values()],
                key=lambda r: -r["patient_acc"]),
        }

    with (out_dir / "probe.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # ---------------- console summary ---------------- #
    for exp, R in report.items():
        print("\n" + "=" * 78)
        print(f"{exp.upper()}   (mean over folds, patient-blocked, bank = other test patients)")
        print("-" * 78)
        print(f"{'key space':<14}{'kNN acc':>9}{'kNN AUC':>9}{'pat acc':>9}"
              f"{'sameMag':>9}{'(chance)':>10}{'subLift':>9}")
        for n, m in R["spaces"].items():
            print(f"{n:<14}{m['knn_acc']:>9.3f}{m['knn_auc']:>9.3f}{m['knn_patient_acc']:>9.3f}"
                  f"{m['same_mag_rate']:>9.3f}{m['same_mag_chance']:>10.3f}{m['subtype_lift']:>9.3f}")
        print("\n  prototypes vs kNN:")
        for m, v in R["prototypes_vs_knn"].items():
            print(f"    {m:<20} img={v['img_acc']:.3f}  auc={v['auc']:.3f}  pat={v['patient_acc']:.3f}")
        print(f"\n  hubness: skew={R['hubness']['Nk_skewness']:.2f}  "
              f"never-retrieved={R['hubness']['never_retrieved_frac']:.3f}  "
              f"top10%share={R['hubness']['top10pct_share']:.3f}")
        print("\n  subtype neighbourhood purity (precision@k / binary knn acc):")
        for st, v in R["subtype_purity"].items():
            print(f"    {st:<3} n={v['n_images']:>4} pat={v['n_patients']}  "
                  f"P@k={v['same_subtype_precision_at_k']:.3f}  "
                  f"knn={v['binary_knn_acc']:.3f}  nb={v['top_neighbour_subtypes']}")
        print("\n  hard patients (param vs retrieval mean prob):")
        for p in HARD_PATIENTS:
            if p in R["per_patient"]:
                v = R["per_patient"][p]
                print(f"    {p:<22} y={v['label']}  param={v['param_mean_prob']:.3f}"
                      f"  retr={v['retrieval_mean_prob']:.3f}"
                      f"  acc {v['param_img_acc']:.3f} -> {v['retrieval_img_acc']:.3f}")
        print("\n  top fusion configs by patient acc:")
        for r in R["fusion_sweep"][:6]:
            print(f"    k={r['k']:<3} a={r['alpha']:<4} tau={r['tau']:<5} "
                  f"gated={r['gated_frac']:.2f}  img={r['img_acc']:.4f} "
                  f"pat={r['patient_acc']:.4f} auc={r['auc']:.4f} "
                  f"sens={r['sens']:.3f} spec={r['spec']:.3f}")
    print(f"\nWrote {out_dir/'probe.json'}")


if __name__ == "__main__":
    main()
