"""Retrieval-design probe #2 — vote rules, magnification routing, gating policy,
and the complementarity ceiling.

Probe #1 (`retrieval_probe.py`) settled the *key space*: the 64-d magnification
block concatenated into `embeddings` makes cosine retrieval ~100% magnification-
locked, so the key must be the 1024-d pre-magnification vector (`features`).
This probe settles what is built on top of that key:

    A  Complementarity ceiling  — when the parametric head is wrong, how often is
       the neighbourhood right? (upper bound on ANY fusion rule; if ~0 there is no
       point building the module at all)
    B  Magnification routing    — bank = same mag only / all mags / cross-mag only
    C  Vote rules               — uniform, similarity-softmax, subtype-balanced,
       per-patient capped, patient-level (retrieve patients, not images)
    D  Subtype purity, honestly — only subtypes with >=2 test patients are
       measurable under patient-blocking; unblocked numbers given as upper bound
    E  Gating policy            — gate on parametric confidence vs on neighbourhood
       agreement (entropy), and the alpha/k surface
    F  Temperature sweep        — sharpness of the similarity-softmax vote

Same caveat as probe #1: bank = the other 15 test patients, so absolute numbers
are a lower bound on a real 66-patient training bank; relative comparisons are the
design signal.

Usage
-----
    python scripts/retrieval_probe2.py --experiments exp1 exp2 exp3
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
    BENIGN_SUBTYPES, MALIGNANT_SUBTYPES, HARD_PATIENTS,
    l2, blocked_sim, topk, vote, patient_acc, sens_spec, auc,
)

MAL = set(MALIGNANT_SUBTYPES)


def primary_key(d) -> np.ndarray:
    """The 1024-d pre-magnification vector — identical to `embeddings` when the
    model has no magnification embedding (exp1), the leading slice otherwise."""
    return d["embeddings"][:, :1024]


# --------------------------------------------------------------------------- #
# A -- complementarity ceiling
# --------------------------------------------------------------------------- #
def complementarity(sim, labels, param_prob, k=15) -> Dict[str, float]:
    nn = topk(sim, k)
    rp = vote(sim, nn, labels, "softmax")
    pc = (param_prob >= 0.5).astype(int) == labels
    rc = (rp >= 0.5).astype(int) == labels
    return {
        "param_acc": round(float(pc.mean()), 4),
        "retrieval_acc": round(float(rc.mean()), 4),
        "both_correct": round(float((pc & rc).mean()), 4),
        "param_wrong_retrieval_right": round(float((~pc & rc).mean()), 4),
        "param_right_retrieval_wrong": round(float((pc & ~rc).mean()), 4),
        "both_wrong": round(float((~pc & ~rc).mean()), 4),
        "oracle_of_two": round(float((pc | rc).mean()), 4),
        "rescue_rate_on_param_errors": round(
            float((~pc & rc).sum() / max((~pc).sum(), 1)), 4),
        "prob_corr": round(float(np.corrcoef(param_prob, rp)[0, 1]), 4),
    }


# --------------------------------------------------------------------------- #
# B -- magnification routing
# --------------------------------------------------------------------------- #
def mag_routing(vecs, labels, pids, mags, k=15) -> Dict[str, dict]:
    base = blocked_sim(vecs, pids)
    out = {}
    for name in ("all_mags", "same_mag_only", "cross_mag_only"):
        sim = base.copy()
        if name == "same_mag_only":
            sim[mags[:, None] != mags[None, :]] = -np.inf
        elif name == "cross_mag_only":
            sim[mags[:, None] == mags[None, :]] = -np.inf
        nn = topk(sim, k)
        p = vote(sim, nn, labels, "softmax")
        out[name] = {
            "img_acc": round(float(((p >= .5).astype(int) == labels).mean()), 4),
            "auc": round(auc(p, labels), 4),
            "patient_acc": round(patient_acc(p, labels, pids), 4),
        }
    return out


# --------------------------------------------------------------------------- #
# C -- vote rules
# --------------------------------------------------------------------------- #
def patient_capped_vote(sim, labels, pids, k=15, cap=3) -> np.ndarray:
    """Take the top-k neighbours but allow at most `cap` from any one bank patient
    — stops a single 235-image slide from owning the neighbourhood."""
    n = sim.shape[0]
    codes = {p: i for i, p in enumerate(np.unique(pids))}
    pcode = np.array([codes[p] for p in pids])
    probs = np.zeros(n)
    order_all = np.argsort(-sim, axis=1)[:, : k * 8]  # candidate pool
    for i in range(n):
        seen: Dict[int, int] = {}
        chosen = []
        for j in order_all[i]:
            if not np.isfinite(sim[i, j]):
                continue
            c = pcode[j]
            if seen.get(c, 0) >= cap:
                continue
            seen[c] = seen.get(c, 0) + 1
            chosen.append(j)
            if len(chosen) == k:
                break
        if not chosen:
            probs[i] = 0.5
            continue
        ch = np.array(chosen)
        s = sim[i, ch]
        w = np.exp((s - s.max()) / 0.07)
        probs[i] = float((w * labels[ch]).sum() / (w.sum() + 1e-12))
    return probs


def patient_level_vote(vecs, labels, pids, k_pat=5) -> np.ndarray:
    """Retrieve *patients* rather than images: score each bank patient by the mean
    similarity of its images to the query, then vote over the top-k patients."""
    x = l2(vecs.astype(np.float64))
    uniq = np.unique(pids)
    plabel = np.array([labels[pids == p][0] for p in uniq])
    centro = np.stack([x[pids == p].mean(0) for p in uniq])
    centro = l2(centro)
    s_all = x @ centro.T
    probs = np.zeros(len(x))
    for i, p in enumerate(pids):
        s = s_all[i].copy()
        s[uniq == p] = -np.inf
        top = np.argsort(-s)[:k_pat]
        w = np.exp((s[top] - s[top].max()) / 0.07)
        probs[i] = float((w * plabel[top]).sum() / (w.sum() + 1e-12))
    return probs


def vote_rules(vecs, labels, subtypes, pids, k=15) -> Dict[str, dict]:
    sim = blocked_sim(vecs, pids)
    nn = topk(sim, k)
    # subtype-balanced bank weights: 1/count so rare subtypes are not outvoted
    cnt = {s: int((subtypes == s).sum()) for s in np.unique(subtypes)}
    wsub = np.array([1.0 / cnt[s] for s in subtypes])
    wsub = wsub / wsub.mean()

    cands = {
        "uniform": vote(sim, nn, labels, "uniform"),
        "softmax": vote(sim, nn, labels, "softmax"),
        "subtype_balanced": vote(sim, nn, labels, "softmax", weights=wsub),
        "patient_capped_c3": patient_capped_vote(sim, labels, pids, k, cap=3),
        "patient_level_k5": patient_level_vote(vecs, labels, pids, k_pat=5),
    }
    out = {}
    for name, p in cands.items():
        se, sp = sens_spec(p, labels)
        row = {
            "img_acc": round(float(((p >= .5).astype(int) == labels).mean()), 4),
            "auc": round(auc(p, labels), 4),
            "patient_acc": round(patient_acc(p, labels, pids), 4),
            "sens": round(se, 4), "spec": round(sp, 4),
        }
        for hp in HARD_PATIENTS:
            m = pids == hp
            if m.any():
                row[f"hp::{hp}"] = round(float(p[m].mean()), 4)
        out[name] = row
    return out


# --------------------------------------------------------------------------- #
# D -- subtype purity, measurable subset only
# --------------------------------------------------------------------------- #
def subtype_purity_honest(vecs, labels, subtypes, pids, k=15) -> Dict[str, dict]:
    blocked = blocked_sim(vecs, pids)
    x = l2(vecs.astype(np.float64))
    unblocked = x @ x.T
    np.fill_diagonal(unblocked, -np.inf)
    nb_b, nb_u = topk(blocked, k), topk(unblocked, k)
    out = {}
    for st in np.unique(subtypes):
        m = subtypes == st
        npat = int(len(np.unique(pids[m])))
        # among MALIGNANT queries, how much of the neighbourhood is the DC mass?
        nb = subtypes[nb_b[m]]
        out[str(st)] = {
            "n_patients": npat,
            "measurable_blocked": npat >= 2,
            "P@k_blocked": round(float((nb == st).mean()), 4) if npat >= 2 else None,
            "P@k_unblocked_upper_bound": round(float((subtypes[nb_u[m]] == st).mean()), 4),
            "frac_neighbours_DC": round(float((nb == "DC").mean()), 4),
            "frac_neighbours_malignant": round(
                float(np.isin(nb, list(MAL)).mean()), 4),
        }
    return out


# --------------------------------------------------------------------------- #
# E/F -- gating policy, alpha/k surface, temperature
# --------------------------------------------------------------------------- #
def gating_study(vecs, labels, pids, param_prob, k=15) -> Dict[str, list]:
    sim = blocked_sim(vecs, pids)
    nn = topk(sim, k)
    rp = vote(sim, nn, labels, "softmax")
    agree = np.abs(rp - 0.5) * 2          # neighbourhood agreement in [0,1]
    rows_param, rows_agree = [], []

    for tau in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        for a in [0.0, 0.25, 0.5, 0.75]:
            g = np.abs(param_prob - 0.5) < tau
            f = param_prob.copy()
            f[g] = a * param_prob[g] + (1 - a) * rp[g]
            se, sp = sens_spec(f, labels)
            rows_param.append({"gate": "param_conf", "tau": tau, "alpha": a,
                               "gated_frac": round(float(g.mean()), 4),
                               "img_acc": round(float(((f >= .5).astype(int) == labels).mean()), 4),
                               "auc": round(auc(f, labels), 4),
                               "patient_acc": round(patient_acc(f, labels, pids), 4),
                               "sens": round(se, 4), "spec": round(sp, 4)})
    for thr in [0.0, 0.3, 0.5, 0.7]:
        for a in [0.0, 0.25, 0.5, 0.75]:
            g = agree >= thr
            f = param_prob.copy()
            f[g] = a * param_prob[g] + (1 - a) * rp[g]
            se, sp = sens_spec(f, labels)
            rows_agree.append({"gate": "nbr_agreement", "thr": thr, "alpha": a,
                               "gated_frac": round(float(g.mean()), 4),
                               "img_acc": round(float(((f >= .5).astype(int) == labels).mean()), 4),
                               "auc": round(auc(f, labels), 4),
                               "patient_acc": round(patient_acc(f, labels, pids), 4),
                               "sens": round(se, 4), "spec": round(sp, 4)})
    return {"param_conf_gate": rows_param, "agreement_gate": rows_agree}


def alpha_k_surface(vecs, labels, pids, param_prob) -> List[dict]:
    sim = blocked_sim(vecs, pids)
    rows = []
    for k in [5, 10, 15, 20, 30, 50, 100]:
        nn = topk(sim, k)
        rp = vote(sim, nn, labels, "softmax")
        for a in np.round(np.arange(0, 1.01, 0.1), 2):
            f = a * param_prob + (1 - a) * rp
            se, sp = sens_spec(f, labels)
            rows.append({"k": k, "alpha": float(a),
                         "img_acc": round(float(((f >= .5).astype(int) == labels).mean()), 4),
                         "auc": round(auc(f, labels), 4),
                         "patient_acc": round(patient_acc(f, labels, pids), 4),
                         "sens": round(se, 4), "spec": round(sp, 4)})
    return rows


def temperature_sweep(vecs, labels, pids, k=15) -> List[dict]:
    sim = blocked_sim(vecs, pids)
    nn = topk(sim, k)
    rows = []
    for t in [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5, 1e9]:
        p = vote(sim, nn, labels, "softmax", temp=t)
        rows.append({"temp": t,
                     "img_acc": round(float(((p >= .5).astype(int) == labels).mean()), 4),
                     "auc": round(auc(p, labels), 4),
                     "patient_acc": round(patient_acc(p, labels, pids), 4)})
    return rows


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
        acc: Dict[str, list] = {}
        for f in files:
            d = np.load(f, allow_pickle=True)
            labels, pids = d["label"].astype(int), d["patient_id"]
            subs, mags, prob = d["subtype"], d["magnification"], d["prob"]
            v = primary_key(d)
            sim = blocked_sim(v, pids)
            acc.setdefault("complementarity", []).append(
                complementarity(sim, labels, prob, args.k))
            acc.setdefault("mag_routing", []).append(mag_routing(v, labels, pids, mags, args.k))
            acc.setdefault("vote_rules", []).append(vote_rules(v, labels, subs, pids, args.k))
            acc.setdefault("subtype_purity", []).append(
                subtype_purity_honest(v, labels, subs, pids, args.k))
            acc.setdefault("gating", []).append(gating_study(v, labels, pids, prob, args.k))
            acc.setdefault("alpha_k", []).append(alpha_k_surface(v, labels, pids, prob))
            acc.setdefault("temperature", []).append(temperature_sweep(v, labels, pids, args.k))

        def mnum(dicts, keys=None):
            ks = keys or [k for k in dicts[0] if isinstance(dicts[0][k], (int, float))]
            return {k: round(float(np.mean([d[k] for d in dicts])), 4) for k in ks}

        def mrows(list_of_lists, idkeys):
            out = []
            for i in range(len(list_of_lists[0])):
                rows = [L[i] for L in list_of_lists]
                r = {k: rows[0][k] for k in idkeys}
                r.update(mnum(rows, [k for k in rows[0] if k not in idkeys and
                                     isinstance(rows[0][k], (int, float))]))
                out.append(r)
            return out

        report[exp] = {
            "complementarity": mnum(acc["complementarity"]),
            "mag_routing": {m: mnum([r[m] for r in acc["mag_routing"]])
                            for m in acc["mag_routing"][0]},
            "vote_rules": {m: mnum([r[m] for r in acc["vote_rules"]])
                           for m in acc["vote_rules"][0]},
            "subtype_purity": {
                s: {"n_patients": acc["subtype_purity"][0][s]["n_patients"],
                    "measurable_blocked": acc["subtype_purity"][0][s]["measurable_blocked"],
                    **{k: (round(float(np.mean([r[s][k] for r in acc["subtype_purity"]])), 4)
                           if acc["subtype_purity"][0][s][k] is not None else None)
                       for k in ("P@k_blocked", "P@k_unblocked_upper_bound",
                                 "frac_neighbours_DC", "frac_neighbours_malignant")}}
                for s in acc["subtype_purity"][0]},
            "gating": {
                "param_conf_gate": mrows([g["param_conf_gate"] for g in acc["gating"]],
                                         ["gate", "tau", "alpha"]),
                "agreement_gate": mrows([g["agreement_gate"] for g in acc["gating"]],
                                        ["gate", "thr", "alpha"]),
            },
            "alpha_k": mrows(acc["alpha_k"], ["k", "alpha"]),
            "temperature": mrows(acc["temperature"], ["temp"]),
        }

    with (out_dir / "probe2.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    for exp, R in report.items():
        print("\n" + "=" * 78)
        print(f"{exp.upper()} — key = 1024-d pre-magnification vector")
        print("-" * 78)
        c = R["complementarity"]
        print(f"  A) complementarity: param={c['param_acc']:.4f} retr={c['retrieval_acc']:.4f} "
              f"oracle={c['oracle_of_two']:.4f}")
        print(f"     param wrong & retrieval right = {c['param_wrong_retrieval_right']:.4f} "
              f"(rescue {c['rescue_rate_on_param_errors']:.1%} of errors); "
              f"reverse = {c['param_right_retrieval_wrong']:.4f}; corr={c['prob_corr']:.3f}")
        print("  B) magnification routing:")
        for m, v in R["mag_routing"].items():
            print(f"     {m:<16} img={v['img_acc']:.4f} auc={v['auc']:.4f} pat={v['patient_acc']:.4f}")
        print("  C) vote rules:")
        for m, v in R["vote_rules"].items():
            hp = "  ".join(f"{k.split('::')[1].split('-')[-1]}={v[k]:.2f}"
                           for k in v if k.startswith("hp::"))
            print(f"     {m:<20} img={v['img_acc']:.4f} auc={v['auc']:.4f} "
                  f"pat={v['patient_acc']:.4f} se={v['sens']:.3f} sp={v['spec']:.3f} | {hp}")
        print("  D) subtype purity (blocked; None = only 1 test patient):")
        for s, v in R["subtype_purity"].items():
            print(f"     {s:<3} pat={v['n_patients']} P@k={v['P@k_blocked']} "
                  f"upper={v['P@k_unblocked_upper_bound']:.3f} "
                  f"nbDC={v['frac_neighbours_DC']:.3f} nbMal={v['frac_neighbours_malignant']:.3f}")
        print("  E) best gating configs (by patient acc, then img acc):")
        allg = R["gating"]["param_conf_gate"] + R["gating"]["agreement_gate"]
        for r in sorted(allg, key=lambda r: (-r["patient_acc"], -r["img_acc"]))[:6]:
            key = f"tau={r.get('tau')}" if "tau" in r else f"thr={r.get('thr')}"
            print(f"     {r['gate']:<14} {key:<10} a={r['alpha']:<5} gated={r['gated_frac']:.2f} "
                  f"img={r['img_acc']:.4f} pat={r['patient_acc']:.4f} "
                  f"se={r['sens']:.3f} sp={r['spec']:.3f}")
        print("  F) alpha/k surface (top 6 by img acc):")
        for r in sorted(R["alpha_k"], key=lambda r: -r["img_acc"])[:6]:
            print(f"     k={r['k']:<4} a={r['alpha']:<4} img={r['img_acc']:.4f} "
                  f"auc={r['auc']:.4f} pat={r['patient_acc']:.4f} "
                  f"se={r['sens']:.3f} sp={r['spec']:.3f}")
        print("  G) temperature:", "  ".join(
            f"{r['temp']:g}:{r['img_acc']:.3f}" for r in R["temperature"]))
    print(f"\nWrote {out_dir/'probe2.json'}")


if __name__ == "__main__":
    main()
