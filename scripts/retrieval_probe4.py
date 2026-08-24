"""Retrieval-design probe #4 — (A) magnification-embedding audit and
(B) unified vs. split memory-bank storage.

Probes 1-3 settled the key space, the vote rule, the granularity pair and the
fusion gate (see `docs/retrieval.md`). This probe answers two follow-up questions that
those probes raised but did not test:

    A  MAGNIFICATION AUDIT.  exp2/exp3 concatenate a 64-d learned magnification
       embedding onto the 1024-d fused feature.  Probe #1 showed it poisons the
       retrieval key.  Does it earn its place anywhere *else* -- i.e. does the
       *classifier* actually use it?  Tested four ways:
         A1 norm/geometry share of the block inside the 1088-d vector
         A2 counterfactual mag-swap on the frozen linear head: re-classify every
            test image under all four magnification ids, and under a zeroed /
            mean block.  If accuracy is unchanged, the block is a per-mag *bias*,
            not a conditioning signal.
         A3 is magnification already linearly decodable from the 1024-d feature
            alone?  (if yes, the lookup table is redundant information)
         A4 per-magnification accuracy of exp1 vs exp2 vs exp3 -- does the mag
            signal help at any specific zoom?

    B  UNIFIED MEMORY BANK.  `docs/retrieval.md` D5 specifies two retrieval
       granularities (image kNN + slide prototypes).  Can both live in ONE store?
       Three storage/query designs are compared on identical evidence:
         B1 split indices, separate queries, evidence combined downstream (D5)
         B2 ONE index holding image rows AND centroid rows, ONE top-k over the
            union (the naive "just put everything in one table" reading)
         B3 one unified store + a `level` column, two *views* queried separately
            (== B1 numerically; verified here, not assumed)
       Plus the diagnostic that explains B2: the cosine-similarity distributions
       of image keys and centroid keys are not on the same scale.

Same caveat as probes 1-3: the bank is the other 15 test patients under strict
leave-one-patient-out, so absolutes are a lower bound; relative order is the
design signal.

Usage
-----
    python scripts/retrieval_probe4.py --experiments exp1 exp2 exp3
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rap_mst.experiments import resolve_experiment_dir  # noqa: E402

FEATURE_DIM = 1024
MAG_ORDER = [40, 100, 200, 400]  # rap_mst/constants.py MAGNIFICATION_TO_INDEX


def features_of(d) -> np.ndarray:
    return d["embeddings"][:, :FEATURE_DIM]


def has_mag_block(d) -> bool:
    return d["embeddings"].shape[1] > FEATURE_DIM


# =========================================================================== #
# A -- magnification-embedding audit
# =========================================================================== #
def a1_geometry(d) -> Dict[str, float]:
    """How much of the 1088-d vector's mass is the constant lookup block?"""
    emb = d["embeddings"].astype(np.float64)
    feat, blk = emb[:, :FEATURE_DIM], emb[:, FEATURE_DIM:]
    nf = np.linalg.norm(feat, axis=1)
    nb = np.linalg.norm(blk, axis=1)
    return {
        "mean_feature_norm": round(float(nf.mean()), 4),
        "mean_magblock_norm": round(float(nb.mean()), 4),
        "magblock_norm_share": round(float((nb ** 2 / (nf ** 2 + nb ** 2)).mean()), 4),
        # cosine between two random images of the SAME mag, block only vs full
        "n_distinct_block_values": int(len(np.unique(blk.round(6), axis=0))),
    }


def a2_counterfactual(state, d) -> Dict[str, object]:
    """Re-run the frozen linear head under every magnification id, and with the
    block zeroed / replaced by the table mean.  Pure numpy: the head is
    Dropout+Linear, so logits = W @ [feat, blk] + b."""
    W = state["classifier.net.1.weight"].astype(np.float64)   # [2, 1088]
    b = state["classifier.net.1.bias"].astype(np.float64)     # [2]
    E = state["magnification.embedding.weight"].astype(np.float64)  # [4, 64]

    feat = features_of(d).astype(np.float64)
    labels = d["label"].astype(int)
    mags = d["magnification"].astype(int)
    mag_idx = np.array([MAG_ORDER.index(m) for m in mags])

    def probs_for(blocks: np.ndarray) -> np.ndarray:
        z = np.concatenate([feat, blocks], axis=1) @ W.T + b
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e[:, 1] / e.sum(axis=1)

    p_true = probs_for(E[mag_idx])
    out: Dict[str, object] = {
        "acc_true_mag": round(float(((p_true >= .5).astype(int) == labels).mean()), 4),
        "auc_true_mag": round(auc(p_true, labels), 4),
    }

    # (i) every image forced to each of the four magnification ids
    per_forced = {}
    for j, m in enumerate(MAG_ORDER):
        p = probs_for(np.repeat(E[j][None, :], len(feat), axis=0))
        per_forced[f"forced_{m}x"] = {
            "acc": round(float(((p >= .5).astype(int) == labels).mean()), 4),
            "auc": round(auc(p, labels), 4),
            "mean_abs_prob_shift_vs_true": round(float(np.abs(p - p_true).mean()), 5),
        }
    out["forced_magnification"] = per_forced

    # (ii) block ablated: zeros, and the table mean (a pure constant bias)
    for name, blk in (("zeroed", np.zeros((len(feat), E.shape[1]))),
                      ("table_mean", np.repeat(E.mean(0)[None, :], len(feat), axis=0))):
        p = probs_for(blk)
        out[f"acc_block_{name}"] = round(float(((p >= .5).astype(int) == labels).mean()), 4)
        out[f"auc_block_{name}"] = round(auc(p, labels), 4)
        out[f"mean_abs_prob_shift_{name}"] = round(float(np.abs(p - p_true).mean()), 5)

    # (iii) the block's whole contribution to the malignant logit, per mag: it is
    # a scalar offset shared by every image at that magnification.
    delta = (E @ (W[1] - W[0])[FEATURE_DIM:])          # [4] logit offset per mag
    out["logit_offset_per_mag"] = {f"{m}x": round(float(delta[j]), 4)
                                   for j, m in enumerate(MAG_ORDER)}
    out["logit_offset_spread"] = round(float(delta.max() - delta.min()), 4)
    # scale reference: spread of the feature-driven logit across images
    feat_logit = feat @ (W[1] - W[0])[:FEATURE_DIM]
    out["feature_logit_std"] = round(float(feat_logit.std()), 4)
    out["offset_spread_over_feature_std"] = round(
        float((delta.max() - delta.min()) / (feat_logit.std() + 1e-12)), 4)
    return out


def a3_mag_decodable(d, k: int = 15) -> Dict[str, float]:
    """Is magnification already recoverable from the 1024-d feature alone?
    Patient-blocked kNN over the feature key -- no learned probe, no leakage."""
    feat = features_of(d)
    mags = d["magnification"].astype(int)
    pids = d["patient_id"]
    sim = blocked_sim(feat, pids)
    nn = topk(sim, k)
    nb = mags[nn]
    pred = np.array([np.bincount(row, minlength=401)[MAG_ORDER].argmax() for row in nb])
    pred = np.array(MAG_ORDER)[pred]
    return {
        "knn_mag_recovery_acc": round(float((pred == mags).mean()), 4),
        "chance": round(float(max(np.bincount(
            [MAG_ORDER.index(m) for m in mags], minlength=4) / len(mags))), 4),
        "same_mag_neighbour_rate": round(float((nb == mags[:, None]).mean()), 4),
    }


def a4_per_mag_head(d) -> Dict[str, dict]:
    """Parametric-head accuracy split by magnification (uses the saved probs)."""
    prob, labels, mags = d["prob"], d["label"].astype(int), d["magnification"].astype(int)
    out = {}
    for m in MAG_ORDER:
        s = mags == m
        if not s.any():
            continue
        se, sp = sens_spec(prob[s], labels[s])
        out[f"{m}x"] = {
            "n": int(s.sum()),
            "acc": round(float(((prob[s] >= .5).astype(int) == labels[s]).mean()), 4),
            "auc": round(auc(prob[s], labels[s]), 4),
            "sens": round(se, 4), "spec": round(sp, 4),
        }
    return out


# =========================================================================== #
# B -- unified vs split memory bank
# =========================================================================== #
def _centroids(x, labels, pids):
    uniq = np.unique(pids)
    plab = np.array([labels[pids == p][0] for p in uniq])
    cen = l2(np.stack([x[pids == p].mean(0) for p in uniq]))
    return uniq, plab, cen


def _metrics(p, labels, pids) -> Dict[str, float]:
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
    return row


def b_storage_designs(vecs, labels, pids, k=15, slide_k=5, temp=0.07) -> Dict[str, dict]:
    x = l2(vecs.astype(np.float64))
    uniq, plab, cen = _centroids(x, labels, pids)

    # ---- B1: split indices, queried separately (docs/retrieval.md D5) ----------- #
    sim_img = blocked_sim(vecs, pids)
    nn = topk(sim_img, k)
    p_img = vote(sim_img, nn, labels, "softmax", temp=temp)

    s_cen = x @ cen.T
    for i, p in enumerate(pids):
        s_cen[i, uniq == p] = -np.inf
    p_slide = np.zeros(len(x))
    for i in range(len(x)):
        top = np.argsort(-s_cen[i])[:slide_k]
        w = np.exp((s_cen[i, top] - s_cen[i, top].max()) / temp)
        p_slide[i] = float((w * plab[top]).sum() / (w.sum() + 1e-12))

    out = {
        "B1_split_image_view": _metrics(p_img, labels, pids),
        "B1_split_slide_view": _metrics(p_slide, labels, pids),
        "B1_split_blend_50_50": _metrics(0.5 * p_img + 0.5 * p_slide, labels, pids),
    }

    # ---- similarity-scale diagnostic -------------------------------------- #
    fin_i = sim_img[np.isfinite(sim_img)]
    fin_c = s_cen[np.isfinite(s_cen)]
    # per-query top-1 of each store, the quantity a merged top-k compares
    t1_i = np.max(np.where(np.isfinite(sim_img), sim_img, -np.inf), axis=1)
    t1_c = np.max(np.where(np.isfinite(s_cen), s_cen, -np.inf), axis=1)
    out["similarity_scale"] = {
        "mean_sim_image_rows": round(float(fin_i.mean()), 4),
        "mean_sim_centroid_rows": round(float(fin_c.mean()), 4),
        "mean_top1_image": round(float(t1_i.mean()), 4),
        "mean_top1_centroid": round(float(t1_c.mean()), 4),
        "frac_queries_centroid_beats_best_image": round(float((t1_c > t1_i).mean()), 4),
        "std_sim_image_rows": round(float(fin_i.std()), 4),
        "std_sim_centroid_rows": round(float(fin_c.std()), 4),
    }

    # ---- B2: one index over the union, one top-k -------------------------- #
    merged = np.concatenate([sim_img, s_cen], axis=1)
    mlab = np.concatenate([labels, plab])
    is_cen = np.concatenate([np.zeros(len(labels), bool), np.ones(len(plab), bool)])
    for kk in (k, slide_k + k):
        order = np.argsort(-merged, axis=1)[:, :kk]
        p = np.zeros(len(x))
        cen_share = np.zeros(len(x))
        for i in range(len(x)):
            sel = order[i][np.isfinite(merged[i, order[i]])]
            s = merged[i, sel]
            w = np.exp((s - s.max()) / temp)
            p[i] = float((w * mlab[sel]).sum() / (w.sum() + 1e-12))
            cen_share[i] = float(is_cen[sel].mean())
        row = _metrics(p, labels, pids)
        row["mean_centroid_share_of_topk"] = round(float(cen_share.mean()), 4)
        row["frac_queries_zero_centroids"] = round(float((cen_share == 0).mean()), 4)
        out[f"B2_merged_single_topk_k{kk}"] = row

    # ---- B2b: merged index, but centroid sims re-scaled to image scale ----- #
    #  (does per-level score normalisation rescue the naive merge?)
    mu_i, sd_i = fin_i.mean(), fin_i.std()
    mu_c, sd_c = fin_c.mean(), fin_c.std()
    s_cen_z = (s_cen - mu_c) / (sd_c + 1e-12) * sd_i + mu_i
    merged_z = np.concatenate([sim_img, s_cen_z], axis=1)
    order = np.argsort(-merged_z, axis=1)[:, :k]
    p = np.zeros(len(x))
    cen_share = np.zeros(len(x))
    for i in range(len(x)):
        sel = order[i][np.isfinite(merged_z[i, order[i]])]
        s = merged_z[i, sel]
        w = np.exp((s - s.max()) / temp)
        p[i] = float((w * mlab[sel]).sum() / (w.sum() + 1e-12))
        cen_share[i] = float(is_cen[sel].mean())
    row = _metrics(p, labels, pids)
    row["mean_centroid_share_of_topk"] = round(float(cen_share.mean()), 4)
    out["B2b_merged_zscored_levels"] = row

    # ---- B3: unified store + level column, two views (must equal B1) ------- #
    keys = np.concatenate([x, cen], axis=0)
    level = np.concatenate([np.zeros(len(x), int), np.ones(len(cen), int)])
    lab_all = np.concatenate([labels, plab])
    pid_all = np.concatenate([pids, uniq])
    s_all = x @ keys.T
    s_all[:, level == 1] = np.where(
        pids[:, None] == pid_all[None, level == 1], -np.inf, s_all[:, level == 1])
    s_all[:, level == 0] = np.where(
        pids[:, None] == pid_all[None, level == 0], -np.inf, s_all[:, level == 0])
    np.fill_diagonal(s_all[:, : len(x)], -np.inf)

    v_img = s_all[:, level == 0]
    nn2 = topk(v_img, k)
    p_img2 = vote(v_img, nn2, lab_all[level == 0], "softmax", temp=temp)
    v_cen = s_all[:, level == 1]
    p_slide2 = np.zeros(len(x))
    for i in range(len(x)):
        top = np.argsort(-v_cen[i])[:slide_k]
        w = np.exp((v_cen[i, top] - v_cen[i, top].max()) / temp)
        p_slide2[i] = float((w * lab_all[level == 1][top]).sum() / (w.sum() + 1e-12))
    out["B3_unified_two_views_image"] = _metrics(p_img2, labels, pids)
    out["B3_unified_two_views_slide"] = _metrics(p_slide2, labels, pids)
    out["B3_equals_B1"] = {
        "max_abs_diff_p_img": round(float(np.abs(p_img2 - p_img).max()), 8),
        "max_abs_diff_p_slide": round(float(np.abs(p_slide2 - p_slide).max()), 8),
    }
    return out


# =========================================================================== #
def load_state(exp: str, fold: int) -> Dict[str, np.ndarray] | None:
    run_root = Path("runs") / resolve_experiment_dir(exp)
    cands = sorted(run_root.glob(f"*train_fold{fold}/checkpoints/best.pt"))
    if not cands:
        return None
    try:
        import torch
    except ModuleNotFoundError:
        return None
    ck = torch.load(cands[-1], map_location="cpu")
    sd = ck["model"]
    want = ("classifier.net.1.weight", "classifier.net.1.bias",
            "magnification.embedding.weight")
    if not all(w in sd for w in want):
        return None
    return {w: sd[w].float().numpy() for w in want}


def mean_rows(dicts: List[dict]) -> dict:
    """Recursively average a list of same-shaped numeric dicts."""
    out = {}
    for k, v in dicts[0].items():
        if isinstance(v, dict):
            out[k] = mean_rows([d[k] for d in dicts])
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = round(float(np.mean([d[k] for d in dicts])), 5)
        else:
            out[k] = v
    return out


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
            fold = int(f.stem.split("fold")[-1])
            d = np.load(f, allow_pickle=True)
            if has_mag_block(d):
                acc.setdefault("A1_geometry", []).append(a1_geometry(d))
                st = load_state(exp, fold)
                if st is not None:
                    acc.setdefault("A2_counterfactual", []).append(a2_counterfactual(st, d))
            acc.setdefault("A3_mag_decodable", []).append(a3_mag_decodable(d, args.k))
            acc.setdefault("A4_per_mag_head", []).append(a4_per_mag_head(d))
            acc.setdefault("B_storage", []).append(
                b_storage_designs(features_of(d), d["label"].astype(int),
                                  d["patient_id"], k=args.k))
        report[exp] = {name: mean_rows(rows) for name, rows in acc.items()}

    with (out_dir / "probe4.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    for exp, R in report.items():
        print("\n" + "=" * 78)
        print(exp.upper())
        print("-" * 78)
        if "A1_geometry" in R:
            g = R["A1_geometry"]
            print(f"  A1 block norm share of vector: {g['magblock_norm_share']:.4f} "
                  f"(|feat|={g['mean_feature_norm']:.2f} |blk|={g['mean_magblock_norm']:.2f}, "
                  f"{g['n_distinct_block_values']} distinct block values)")
        if "A2_counterfactual" in R:
            c = R["A2_counterfactual"]
            print(f"  A2 head acc  true mag={c['acc_true_mag']:.4f}  "
                  f"block zeroed={c['acc_block_zeroed']:.4f}  "
                  f"table mean={c['acc_block_table_mean']:.4f}")
            for m, v in c["forced_magnification"].items():
                print(f"       {m:<12} acc={v['acc']:.4f} auc={v['auc']:.4f} "
                      f"|dp|={v['mean_abs_prob_shift_vs_true']:.4f}")
            print(f"     logit offsets {c['logit_offset_per_mag']}  spread="
                  f"{c['logit_offset_spread']:.3f} vs feature-logit std="
                  f"{c['feature_logit_std']:.3f} "
                  f"(ratio {c['offset_spread_over_feature_std']:.3f})")
        if "A3_mag_decodable" in R:
            a = R["A3_mag_decodable"]
            print(f"  A3 mag recoverable from 1024-d feature: kNN acc="
                  f"{a['knn_mag_recovery_acc']:.4f} (chance {a['chance']:.3f}), "
                  f"same-mag nbr rate={a['same_mag_neighbour_rate']:.4f}")
        if "A4_per_mag_head" in R:
            print("  A4 head per magnification: " + "  ".join(
                f"{m}:acc={v['acc']:.3f}/auc={v['auc']:.3f}"
                for m, v in R["A4_per_mag_head"].items()))
        if "B_storage" in R:
            B = R["B_storage"]
            print("  B storage designs:")
            for name, v in B.items():
                if name in ("similarity_scale", "B3_equals_B1"):
                    continue
                extra = ""
                if "mean_centroid_share_of_topk" in v:
                    extra = f" centroid_share={v['mean_centroid_share_of_topk']:.3f}"
                print(f"     {name:<32} img={v['img_acc']:.4f} auc={v['auc']:.4f} "
                      f"pat={v['patient_acc']:.4f} se={v['sens']:.3f} "
                      f"sp={v['spec']:.3f}{extra}")
            s = B["similarity_scale"]
            print(f"     scale: mean sim img={s['mean_sim_image_rows']:.4f} "
                  f"cen={s['mean_sim_centroid_rows']:.4f} | top1 img="
                  f"{s['mean_top1_image']:.4f} cen={s['mean_top1_centroid']:.4f} | "
                  f"centroid beats best image on {s['frac_queries_centroid_beats_best_image']:.1%}")
            e = B["B3_equals_B1"]
            print(f"     B3==B1 check: max|dp_img|={e['max_abs_diff_p_img']:.2e} "
                  f"max|dp_slide|={e['max_abs_diff_p_slide']:.2e}")
    print(f"\nWrote {out_dir/'probe4.json'}")


if __name__ == "__main__":
    main()
