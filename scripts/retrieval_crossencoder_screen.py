"""Cross-encoder retrieval memory -- the pooled-OOF **screen** (no test-set touch).

The question
------------
``docs/results/retrieval_key_ablation.md`` §6 found exactly one quantity in the whole retrieval
study that responds to anything: **error-decorrelation from the parametric head**
(``awrong`` = AUC of ``p_img`` restricted to the images the head misclassifies).
It does not respond to the key (43 configurations, 338x effective-rank range,
corr = +0.007); it responds to *encoder identity* (x1.88 when the bank is built
from exp1 instead of exp3n). That was turned into a
prescription -- *do not build a retrieval bank on the same frozen encoder as the
classifier* -- which the paper cannot currently support, because "a correlate
moved" is not "acting on it works".

This script adds the **third point** of the dose-response curve, chosen by an
independent measurement rather than by convenience: a bank encoded by frozen
**CTransPath**, whose errors are known to differ from the Swin ladder's
(``docs/results/foundation_baseline.md`` §5.7 -- it solves ``DC-14-12312`` and ``TA-14-16184``
and breaks on ``MC-14-16456``).

    bank keys + query keys  <- CTransPath      (encoder A, foreign, 768-d)
    p_param                 <- exp3n's head    (encoder B, unchanged)
    bank / vote / cap / two-level ranking / gate  <- the PRODUCTION classes

Pre-registration (written before execution; see the header of the output JSON)
-----------------------------------------------------------------------------
* **Primary endpoints are mechanism, not accuracy**: ``awrong``, the gate weight
  kept on foreign evidence, the subtype lift, and the neighbour overlap with the
  same-encoder memory. Predicted: ``awrong`` above the exp1 cross-encoder band
  (0.127-0.136) if encoder identity is the governing variable.
* **Accuracy is secondary** and expected to be null. The success bar is
  deliberately higher than "beats ``p_param``": ``p_final`` must beat **both**
  ``p_param`` **and** the two-probe ensemble of exp3n's head with CTransPath's own
  linear probe. Anything less is an ensemble wearing a retrieval costume.
* **The test set is not touched by this script.** Promotion to the production
  B1->B2->C run is conditional on this screen.

What is deliberately *not* varied: route, k, cap, temperature and the two-level
ranking (D3-D7) stay at their measured production values, and the gate is fitted
by the Stage B2 protocol (pooled OOF, same seed/epochs/lr). The temperature is the
one place where holding the *number* fixed does not hold the *rule* fixed across
encoders -- so it gets a reported sensitivity curve (``--t-sweep``) rather than a
silent rescale.

Usage
-----
    python scripts/retrieval_crossencoder_screen.py
    python scripts/retrieval_crossencoder_screen.py --folds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# The production retrieval stack, and the ablation harness's already-validated
# scoring helpers. Importing them (rather than copying) is what guarantees this
# screen's numbers are computed identically to the 43-key study they must be
# compared against -- `awrong`, the gate protocol and the LOSO control included.
import retrieval_key_ablation as kab  # noqa: E402

from rap_mst.data.splits import load_splits  # noqa: E402
from rap_mst.experiments import encoder_experiment  # noqa: E402
from rap_mst.retrieval.bank import LEVEL_IMAGE  # noqa: E402
from rap_mst.retrieval.builder import retrieval_cfg  # noqa: E402
from rap_mst.retrieval.foreign import load_foreign_caches  # noqa: E402
from rap_mst.utils import reporting  # noqa: E402
from rap_mst.utils.config import load_config  # noqa: E402
from rap_mst.utils.logging_utils import get_logger  # noqa: E402
from rap_mst.utils.runs import available_folds, find_fold_run  # noqa: E402

# --------------------------------------------------------------------------- #
# The pre-registered grid -- three configurations, mirroring exp1's CROSS_GRID so
# the two cross-encoder conditions are compared like for like. This is a
# mechanism test, not a search: no configuration is selected on the outcome.
# --------------------------------------------------------------------------- #
FOREIGN_GRID: Tuple[Tuple[str, str, str], ...] = (
    ("CTP features",             "features", "none"),          # <- primary
    ("CTP features whiten:128",  "features", "whiten:128"),
    ("CTP features pca_drop:10", "features", "pca_drop:10"),
)

#: Re-run in the same process so every band this screen is judged against is
#: recomputed here rather than quoted from a report.
BASE_GRID: Tuple[Tuple[str, str, str], ...] = (
    ("exp3n features (same-encoder D1)", "features", "none"),
)
EXP1_GRID: Tuple[Tuple[str, str, str], ...] = (
    ("exp1 features (cross-encoder control)", "features", "none"),
    ("exp1 fpn.std (cross-encoder control)",  "fpn.std",  "none"),
)

PRIMARY = "CTP features"

#: Temperature sensitivity. T=0.07 was fixed on exp3n's key space, where the
#: top1-to-top15 similarity spread is 0.0008 -- i.e. the softmax is effectively a
#: uniform 15-NN majority. CTransPath's spread is ~0.07, so the SAME T is a
#: materially sharper vote. Reported as a curve; 0.07 remains the primary.
T_SWEEP = (0.01, 0.02, 0.035, 0.07, 0.15, 0.3, 1.0)

OVERLAP_SUBSAMPLE = 600
BOOTSTRAP_RESAMPLES = 2000


# --------------------------------------------------------------------------- #
# Extra diagnostics this screen adds
# --------------------------------------------------------------------------- #
@torch.no_grad()
def neighbour_overlap(bank_a, cache_a, spec_a, bank_b, cache_b, spec_b,
                      rng: np.random.Generator, k: int = 15) -> Dict[str, float]:
    """Do the two encoders retrieve the *same slides*? (not in the mandated list)

    ``awrong`` says whether the memory's **vote** is decorrelated from the head.
    This says whether the memory is even *reading a different book*: the mean
    Jaccard / intersection of the top-k image-level neighbour sets retrieved for
    the same query under the two key spaces, plus the rate at which the single
    nearest neighbour agrees.

    It is the cleanest possible test of "different encoder => different memory",
    and it is independent of the labels, so it cannot be confounded by the vote
    rule or the temperature. Bank row indices are comparable because both banks
    are filled from the same asserted-aligned row order.
    """
    n = cache_a.col("val", "label").shape[0]
    sel = rng.choice(n, size=min(OVERLAP_SUBSAMPLE, n), replace=False)
    pids = list(cache_a.col("val", "patient_id")[sel])
    mags = torch.as_tensor(cache_a.col("val", "mag_index")[sel], dtype=torch.long,
                           device=cache_a.device)

    def top(bank, cache, spec):
        keys = cache.keys("val", spec)[sel]
        res = bank.query(keys, mag_index=mags, level=LEVEL_IMAGE, k=k, route="same_mag",
                         per_patient_cap=3, temperature=0.07, query_patient_ids=pids)
        idx = res.idx.cpu().numpy()
        val = res.valid.cpu().numpy()
        return [set(idx[b][val[b]].tolist()) for b in range(len(sel))], idx[:, 0], val[:, 0]

    a_sets, a_top1, a_ok = top(bank_a, cache_a, spec_a)
    b_sets, b_top1, b_ok = top(bank_b, cache_b, spec_b)

    inter, jacc, patient_inter = [], [], []
    for A, B in zip(a_sets, b_sets):
        if not A or not B:
            continue
        inter.append(len(A & B) / max(len(A), len(B)))
        jacc.append(len(A & B) / len(A | B))
    both = a_ok & b_ok
    # Neighbour *patients* rather than images: a different field of the same slide
    # is arguably the same evidence, so this is the generous version of the test.
    pa = bank_a.patient_ids
    for A, B in zip(a_sets, b_sets):
        if not A or not B:
            continue
        PA, PB = set(pa[list(A)].tolist()), set(pa[list(B)].tolist())
        patient_inter.append(len(PA & PB) / max(len(PA), len(PB)))
    return {
        "topk_overlap_frac": round(float(np.mean(inter)), 4),
        "topk_jaccard": round(float(np.mean(jacc)), 4),
        "topk_patient_overlap_frac": round(float(np.mean(patient_inter)), 4),
        "top1_identical_rate": round(float(np.mean(a_top1[both] == b_top1[both])), 4),
        "n_queries": int(len(inter)),
    }


def gate_opens_on_errors(pool: Dict[str, np.ndarray], weights: np.ndarray) -> Dict[str, float]:
    """``docs/results/retrieval_heldout.md`` §7.1's positive control, on the OOF surface.

    A gate that keeps a *constant* weight is not consulting the memory; a gate
    that relaxes onto the memory precisely where the head is wrong is a working
    selective-consultation mechanism. Reported separately from accuracy, because
    the module can be mechanically correct and still buy nothing.
    """
    y = pool["label"].astype(int)
    wrong = (pool["prob_param"] >= 0.5).astype(int) != y
    wp = weights[:, 0]
    return {
        "mean_w_param_where_head_correct": round(float(wp[~wrong].mean()), 4),
        "mean_w_param_where_head_wrong": round(float(wp[wrong].mean()), 4),
        "gap": round(float(wp[wrong].mean() - wp[~wrong].mean()), 4),
        "mean_w_img_where_head_wrong": round(float(weights[wrong, 1].mean()), 4),
        "mean_w_slide_where_head_wrong": round(float(weights[wrong, 2].mean()), 4),
        "frac_w_param_above_0.95": round(float((wp > 0.95).mean()), 4),
        "n_head_wrong": int(wrong.sum()),
    }


def decision_flips(labels: np.ndarray, base: np.ndarray, fused: np.ndarray,
                   thr: float = 0.5) -> Dict[str, int]:
    """The whole mechanical footprint of fusion: what actually changed, and how."""
    b = (base >= thr).astype(int)
    f = (fused >= thr).astype(int)
    ch = b != f
    return {
        "n_changed": int(ch.sum()),
        "changed_to_correct": int(((f == labels) & ch).sum()),
        "changed_to_wrong": int(((f != labels) & ch).sum()),
        "net": int(((f == labels) & ch).sum() - ((f != labels) & ch).sum()),
        "frac_changed": round(float(ch.mean()), 5),
    }


def best_threshold(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """Accuracy-optimal threshold on this surface (in-sample -> optimistic; labelled)."""
    grid = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))
    accs = [float(((probs >= t).astype(int) == labels).mean()) for t in grid]
    i = int(np.argmax(accs))
    return float(grid[i]), float(accs[i])


def patient_bootstrap(labels: np.ndarray, pids: np.ndarray, a: np.ndarray, b: np.ndarray,
                      resamples: int, seed: int) -> Dict[str, object]:
    """Paired, patient-clustered bootstrap of (a - b) in AUC and accuracy@0.5.

    Resampling is over the 66 CV **patients**, with replacement, never over
    images: one patient contributes 60-235 near-identical fields, so image-level
    resampling would shrink every interval by roughly an order of magnitude
    (``scripts/bootstrap_benchmark.py``, same protocol).
    """
    from sklearn.metrics import roc_auc_score

    uniq = np.unique(pids)
    rows = {p: np.nonzero(pids == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    d_auc, d_acc = [], []
    for _ in range(resamples):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([rows[uniq[i]] for i in pick])
        y = labels[idx]
        if len(np.unique(y)) < 2:
            continue
        d_auc.append(roc_auc_score(y, a[idx]) - roc_auc_score(y, b[idx]))
        d_acc.append(float(((a[idx] >= 0.5).astype(int) == y).mean()
                           - ((b[idx] >= 0.5).astype(int) == y).mean()))

    def ci(v):
        v = np.asarray(v)
        return {
            "delta": round(float(v.mean()), 5),
            "ci95": [round(float(np.percentile(v, 2.5)), 5),
                     round(float(np.percentile(v, 97.5)), 5)],
            "p_two_sided": round(float(2 * min((v <= 0).mean(), (v >= 0).mean())), 4),
        }

    from sklearn.metrics import roc_auc_score as _auc
    return {
        "point_delta_auc": round(float(_auc(labels, a) - _auc(labels, b)), 5),
        "point_delta_acc": round(float(((a >= 0.5).astype(int) == labels).mean()
                                       - ((b >= 0.5).astype(int) == labels).mean()), 5),
        "auc": ci(d_auc), "accuracy": ci(d_acc), "n_resamples": len(d_auc),
    }


# --------------------------------------------------------------------------- #
# The CTransPath probe's own out-of-fold probabilities (the ensemble control)
# --------------------------------------------------------------------------- #
def load_probe_val_probs(experiment: str, log_root: Path, folds: Sequence[int],
                         logger) -> Dict[int, Dict[str, float]]:
    """``image_path -> P(malignant)`` per fold, from Stage F2's ``val_predictions.csv``.

    These are the *same* out-of-fold rows the gate is fitted on, produced by the
    same splits file, so the ensemble control lives on the identical surface as
    everything else here -- no test-set touch is needed to run it.
    """
    out: Dict[int, Dict[str, float]] = {}
    for fold in folds:
        run = find_fold_run(experiment, log_root, fold)
        path = run / "val_predictions.csv"
        if not path.exists():
            raise SystemExit(
                f"Missing {path}. The ensemble control needs Stage F2's val predictions:\n"
                f"  python scripts/train_linear_probe.py --experiment {experiment}"
            )
        with path.open(newline="", encoding="utf-8") as fh:
            out[fold] = {r["image_path"]: float(r["prob_malignant"])
                         for r in csv.DictReader(fh)}
        logger.info(f"  probe val predictions fold {fold}: {len(out[fold])} rows <- {path.parent.name}")
    return out


# --------------------------------------------------------------------------- #
# One configuration, end to end
# --------------------------------------------------------------------------- #
def run_config(name: str, spec: str, transform: str, base_caches, key_caches,
               experiment: str, key_encoder: str, rcfg, folds, seed: int,
               epochs: int, lr: float, k_img: int, device, logger,
               temperature: Optional[float] = None,
               overlap_against: Optional[Dict] = None,
               route: Optional[str] = None, cap: Optional[int] = None,
               k: Optional[int] = None) -> Dict[str, object]:
    """Build 5 banks, query 5 validation splits, pool, fit the gate, diagnose.

    ``temperature`` / ``route`` / ``cap`` / ``k`` are the **artefact-check** knobs.
    They are not part of the grid and never select a reported configuration: D3-D7
    were fixed by measurement on exp3n and are held at those values throughout.
    They exist so the report can state that the null is not an artefact of
    operating parameters chosen for a different key space.
    """
    rng = np.random.default_rng(seed)
    parts, geo, diag, overlaps = [], [], [], []
    for fold in folds:
        base = base_caches[fold]                 # always owns p_param + labels
        kc = key_caches[fold]                    # owns the retrieval keys
        bank = kab.build_bank(kc, spec, transform, key_encoder)
        bank.assert_disjoint(set(base.col("val", "patient_id").tolist()), what="validation")
        memory = kab.make_memory(bank, rcfg, spec, transform)
        if temperature is not None:              # the T sensitivity curve only
            memory.image.temperature = float(temperature)
            memory.slide.temperature = float(temperature)
        if route is not None:
            memory.image.route = str(route)
        if cap is not None:
            memory.image.per_patient_cap = int(cap)
        if k is not None:
            memory.image.k = int(k)
        ev = kab.query_val(memory, kc, spec)
        parts.append({
            "fold": np.full(len(ev["p_img"]), fold),
            "label": base.col("val", "label").astype(int),
            "patient_id": base.col("val", "patient_id"),
            "image_path": kc.col("val", "image_path") if hasattr(kc, "_cols")
            else np.asarray([""] * len(ev["p_img"])),
            "prob_param": base.p_param("val"),
            **ev,
        })
        img_keys = bank.keys[torch.as_tensor(bank.levels == LEVEL_IMAGE, device=bank.keys.device)]
        geo.append(kab.geometry(img_keys, rng))
        diag.append(kab.route_all_diagnostics(bank, kc, spec, rng))
        if overlap_against is not None:
            ob, oc, os_ = overlap_against[fold]
            overlaps.append(neighbour_overlap(bank, kc, spec, ob, oc, os_, rng))
        del bank, memory
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pool = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    labels = pool["label"].astype(int)
    pids = pool["patient_id"]
    gate_out = kab.fit_and_score(pool, seed, epochs, lr, k_img, folds)

    features, probs3, _ = kab._gate_tensors(pool, k_img)
    gate = kab._fit(features, probs3, torch.as_tensor(labels, dtype=torch.float32),
                    seed, epochs, lr)
    with torch.no_grad():
        weights = gate.weights_from_features(features).numpy()

    row: Dict[str, object] = {
        "name": name, "key": spec, "key_transform": transform,
        "key_encoder": key_encoder, "p_param_encoder": experiment,
        "temperature": float(temperature) if temperature is not None else 0.07,
        "n_rows": int(len(labels)), "n_patients": int(len(set(pids.tolist()))),
        "geometry": {k: round(float(np.mean([g[k] for g in geo])), 4) for k in geo[0]},
        "neighbourhood": {
            **{k: round(float(np.mean([d[k] for d in diag])), 4) for k in diag[0]},
            "mean_top1_sim": round(float(pool["top1_sim"].mean()), 4),
            "std_top1_sim": round(float(pool["top1_sim"].std()), 4),
            "mean_distinct_patients": round(float(pool["n_distinct"].mean()), 3),
        },
        "p_param": kab.score_column(labels, pids, pool["prob_param"]),
        "p_img": kab.score_column(labels, pids, pool["p_img"]),
        "p_slide": kab.score_column(labels, pids, pool["p_slide"]),
        "p_final": kab.score_column(labels, pids, gate_out["p_final"]),
        "p_final_loso": kab.score_column(labels, pids, gate_out["p_final_loso"]),
        "gate_mean_weights": gate_out["mean_weights"],
        "gate_behaviour": gate_opens_on_errors(pool, weights),
        "complementarity": kab.complementarity(labels, pool["prob_param"], pool["p_img"]),
        "flips_vs_param": decision_flips(labels, pool["prob_param"], gate_out["p_final_loso"]),
    }
    if overlaps:
        row["neighbour_overlap_vs_same_encoder"] = {
            k: round(float(np.mean([o[k] for o in overlaps])), 4) for k in overlaps[0]
        }
    for tag in ("p_final", "p_final_loso"):
        sfx = "" if tag == "p_final" else "_loso"
        row[f"delta_auc_final{sfx}_vs_param"] = round(
            row[tag]["image_auc"] - row["p_param"]["image_auc"], 4)
        row[f"delta_acc_final{sfx}_vs_param"] = round(
            row[tag]["image_acc"] - row["p_param"]["image_acc"], 4)
        row[f"delta_patacc_final{sfx}_vs_param"] = round(
            row[tag]["patient_acc"] - row["p_param"]["patient_acc"], 4)
    return row, pool, gate_out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pooled-OOF screen: a CTransPath-built memory bank behind exp3n's head.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--experiment", default=None, help="Base encoder (default: retrieval.base_experiment).")
    ap.add_argument("--foreign-encoder", default="ctranspath")
    ap.add_argument("--probe-experiment", default="expfm",
                    help="Run dir supplying the foreign encoder's own OOF probe probabilities.")
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--cross-encoder", default="exp1",
                    help="Also recompute the existing Swin cross-encoder control (or '' to skip).")
    ap.add_argument("--out", default="analysis/retrieval_crossencoder")
    ap.add_argument("--keys-dir", default="analysis/retrieval_keys")
    ap.add_argument("--t-sweep", action="store_true", default=True)
    ap.add_argument("--no-t-sweep", dest="t_sweep", action="store_false")
    ap.add_argument("--artefact-checks", action="store_true", default=True)
    ap.add_argument("--no-artefact-checks", dest="artefact_checks", action="store_false")
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rcfg = retrieval_cfg(cfg)
    if rcfg is None:
        raise SystemExit(f"No `retrieval:` block in {args.config}.")
    experiment = args.experiment or str(getattr(rcfg, "base_experiment", "exp3n"))
    encoder = encoder_experiment(experiment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_root = Path(cfg.logging.log_root)
    folds = args.folds or available_folds(encoder, log_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("rap_mst.retrieval_crossencoder", out_dir / "screen.log")
    logging.getLogger("rap_mst.retrieval.memory").setLevel(logging.WARNING)

    k_img = int(getattr(getattr(getattr(rcfg, "levels", None), "image", None), "k", 15))
    gate_cfg = getattr(rcfg, "gate", None)
    epochs = int(getattr(gate_cfg, "epochs", 400))
    lr = float(getattr(gate_cfg, "lr", 0.01))
    seed = int(cfg.seed)

    fcache = str(getattr(cfg.foundation, "cache_path")).format(encoder=args.foreign_encoder)
    reporting.section(logger, "CROSS-ENCODER RETRIEVAL SCREEN (pooled OOF)", [
        reporting.kv("base encoder (p_param)", f"{experiment} -> {encoder}"),
        reporting.kv("foreign encoder (bank+query keys)", args.foreign_encoder),
        reporting.kv("foreign feature cache", fcache),
        reporting.kv("folds", folds),
        reporting.kv("held fixed", "route/k/cap/two-level ranking (D3-D7) + Stage B2 gate protocol"),
        reporting.kv("surface", "pooled out-of-fold validation -- the TEST SET IS NOT TOUCHED"),
        reporting.kv("primary endpoints", "awrong, gate weight on memory, subtype lift, "
                                          "neighbour overlap"),
        reporting.kv("secondary endpoint", "accuracy vs p_param AND vs the two-probe ensemble"),
        reporting.kv("output dir", str(out_dir)),
        reporting.kv("device", str(device)),
    ])

    # --- caches ------------------------------------------------------------- #
    base_caches = {}
    for fold in folds:
        p = Path(args.keys_dir) / experiment / f"cache_fold{fold}.npz"
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run: python scripts/retrieval_key_ablation.py --stage cache")
        base_caches[fold] = kab.FoldCache(p, device)
    logger.info(f"loaded {len(base_caches)} base fold caches from {Path(args.keys_dir) / experiment}")

    splits = load_splits(cfg.data.splits_path)
    foreign = load_foreign_caches(fcache, cfg, splits, folds, device, args.foreign_encoder)
    align = {}
    for fold in folds:
        align[fold] = foreign[fold].assert_aligned_with(base_caches[fold])
    logger.info("ROW ALIGNMENT VERIFIED (image_path join; patient_id/label/mag_index checked "
                f"element-wise on {sum(v['bank']['rows'] + v['val']['rows'] for v in align.values())} rows)")

    x_caches = {}
    if args.cross_encoder:
        for fold in folds:
            p = Path(args.keys_dir) / args.cross_encoder / f"cache_fold{fold}.npz"
            if not p.exists():
                logger.warning(f"skipping the {args.cross_encoder} control: {p} not found")
                x_caches = {}
                break
            xc = kab.FoldCache(p, device)
            for split in ("bank", "val"):
                if not np.array_equal(xc.col(split, "patient_id"),
                                      base_caches[fold].col(split, "patient_id")):
                    raise AssertionError(f"fold {fold}/{split}: {args.cross_encoder} cache not aligned.")
            x_caches[fold] = xc

    probe_val = load_probe_val_probs(args.probe_experiment, log_root, folds, logger)

    # --- the same-encoder baseline first: it defines the overlap reference --- #
    results: List[Dict] = []
    pools: Dict[str, Dict[str, np.ndarray]] = {}
    gates: Dict[str, Dict] = {}
    overlap_ref: Dict[int, Tuple] = {}
    for fold in folds:
        b = kab.build_bank(base_caches[fold], "features", "none", experiment)
        overlap_ref[fold] = (b, base_caches[fold], "features")

    todo: List[Tuple] = [(n, s, t, base_caches, experiment, None) for n, s, t in BASE_GRID]
    if x_caches:
        todo += [(n, s, t, x_caches, args.cross_encoder, None) for n, s, t in EXP1_GRID]
    todo += [(n, s, t, foreign, args.foreign_encoder, overlap_ref) for n, s, t in FOREIGN_GRID]

    for i, (name, spec, transform, kc, kenc, ov) in enumerate(todo, 1):
        logger.info("")
        logger.info(f"[{i}/{len(todo)}] {name}  key={spec!r} transform={transform!r} "
                    f"key_encoder={kenc}")
        row, pool, gate_out = run_config(name, spec, transform, base_caches, kc, experiment,
                                         kenc, rcfg, folds, seed, epochs, lr, k_img, device,
                                         logger, overlap_against=ov)
        results.append(row)
        pools[name] = pool
        gates[name] = gate_out
        c, n_ = row["complementarity"], row["neighbourhood"]
        logger.info(
            f"    dim={row['geometry']['dim']:.0f} eff_rank={row['geometry']['effective_rank']:.2f} "
            f"| awrong={c['auc_where_param_wrong']:.4f} corr={c['corr_param_img']:.3f} "
            f"| p_img auc={row['p_img']['image_auc']:.4f} "
            f"| LOSO auc={row['p_final_loso']['image_auc']:.4f} "
            f"({row['delta_auc_final_loso_vs_param']:+.4f}) "
            f"acc={row['p_final_loso']['image_acc']:.4f} "
            f"({row['delta_acc_final_loso_vs_param']:+.4f}) "
            f"pat={row['delta_patacc_final_loso_vs_param']:+.4f}")
        logger.info(
            f"    w={row['gate_mean_weights']} (w_param wrong {row['gate_behaviour']['mean_w_param_where_head_wrong']:.3f} "
            f"vs correct {row['gate_behaviour']['mean_w_param_where_head_correct']:.3f}) "
            f"| mag_lock={n_['same_mag_rate_route_all']:.3f} subtype_lift={n_['subtype_lift']:+.3f} "
            f"| top1_sim={n_['mean_top1_sim']:.4f} "
            f"| best-alpha {c['best_fixed_alpha']:.2f} -> {c['blend_gain_auc']:+.4f}")
        if "neighbour_overlap_vs_same_encoder" in row:
            o = row["neighbour_overlap_vs_same_encoder"]
            logger.info(f"    neighbour overlap vs exp3n memory: top-{k_img} {o['topk_overlap_frac']:.3f} "
                        f"(jaccard {o['topk_jaccard']:.3f}), patients {o['topk_patient_overlap_frac']:.3f}, "
                        f"top-1 identical {o['top1_identical_rate']:.3f}")

    # ----------------------------------------------------------------- #
    # The ensemble control -- the bar §B.4 says p_final must clear
    # ----------------------------------------------------------------- #
    primary = next(r for r in results if r["name"] == PRIMARY)
    pool = pools[PRIMARY]
    labels = pool["label"].astype(int)
    pids = pool["patient_id"]
    p_param = pool["prob_param"]
    p_probe = np.asarray([probe_val[int(f)][p] for f, p in zip(pool["fold"], pool["image_path"])])
    p_ens = 0.5 * (p_param + p_probe)
    p_final = gates[PRIMARY]["p_final_loso"]

    # A fitted-weight ensemble, LOSO-fitted so it is not flattered relative to the
    # gate: the fairest possible version of the objection "this is just ensembling".
    p_ens_fit = np.zeros_like(p_ens)
    grid = np.linspace(0.0, 1.0, 101)
    for held in folds:
        tr = pool["fold"] != held
        if not tr.any():                       # single-fold smoke run: no held-out fit
            p_ens_fit[~tr] = p_ens[~tr]
            continue
        aucs = [kab.auc(labels[tr], a * p_param[tr] + (1 - a) * p_probe[tr]) for a in grid]
        a_star = float(grid[int(np.nanargmax(aucs))])
        p_ens_fit[~tr] = a_star * p_param[~tr] + (1 - a_star) * p_probe[~tr]

    ens = {
        "p_probe_ctranspath": kab.score_column(labels, pids, p_probe),
        "p_ens_half": kab.score_column(labels, pids, p_ens),
        "p_ens_fitted_loso": kab.score_column(labels, pids, p_ens_fit),
        "awrong_of_probe": round(kab.auc(labels[(p_param >= 0.5).astype(int) != labels],
                                         p_probe[(p_param >= 0.5).astype(int) != labels]), 4),
        "corr_param_probe": round(float(np.corrcoef(p_param, p_probe)[0, 1]), 4),
        "final_beats_param":  bool(primary["p_final_loso"]["image_auc"] > primary["p_param"]["image_auc"]),
        "final_beats_ens_half": bool(primary["p_final_loso"]["image_auc"]
                                     > kab.score_column(labels, pids, p_ens)["image_auc"]),
        "final_beats_ens_fitted": bool(primary["p_final_loso"]["image_auc"]
                                       > kab.score_column(labels, pids, p_ens_fit)["image_auc"]),
    }
    reporting.section(logger, "ENSEMBLE CONTROL (the bar p_final must clear)", [
        reporting.kv("p_param (exp3n head)", f"auc {primary['p_param']['image_auc']:.4f} "
                                             f"acc {primary['p_param']['image_acc']:.4f}"),
        reporting.kv("p_probe (CTransPath linear head)",
                     f"auc {ens['p_probe_ctranspath']['image_auc']:.4f} "
                     f"acc {ens['p_probe_ctranspath']['image_acc']:.4f}"),
        reporting.kv("p_ens = 1/2(param+probe)", f"auc {ens['p_ens_half']['image_auc']:.4f} "
                                                 f"acc {ens['p_ens_half']['image_acc']:.4f}"),
        reporting.kv("p_ens fitted (LOSO alpha)", f"auc {ens['p_ens_fitted_loso']['image_auc']:.4f} "
                                                  f"acc {ens['p_ens_fitted_loso']['image_acc']:.4f}"),
        reporting.kv("p_final (CTP memory, LOSO)", f"auc {primary['p_final_loso']['image_auc']:.4f} "
                                                   f"acc {primary['p_final_loso']['image_acc']:.4f}"),
        reporting._rule(),
        reporting.kv("awrong of p_probe (head's errors)", f"{ens['awrong_of_probe']:.4f}"),
        reporting.kv("awrong of p_img (CTP memory)",
                     f"{primary['complementarity']['auc_where_param_wrong']:.4f}"),
        reporting.kv("corr(p_param, p_probe)", f"{ens['corr_param_probe']:.4f}"),
        reporting.kv("corr(p_param, p_img)", f"{primary['complementarity']['corr_param_img']:.4f}"),
    ])

    # --- paired, patient-clustered bootstrap -------------------------------- #
    boot = {
        "final_vs_param": patient_bootstrap(labels, pids, p_final, p_param, args.resamples, seed),
        "final_vs_ens_half": patient_bootstrap(labels, pids, p_final, p_ens, args.resamples, seed),
        "final_vs_ens_fitted": patient_bootstrap(labels, pids, p_final, p_ens_fit, args.resamples, seed),
        "ens_half_vs_param": patient_bootstrap(labels, pids, p_ens, p_param, args.resamples, seed),
        "img_vs_param": patient_bootstrap(labels, pids, pool["p_img"], p_param, args.resamples, seed),
    }
    logger.info("")
    logger.info(f"Patient-clustered bootstrap ({args.resamples} resamples over "
                f"{len(np.unique(pids))} CV patients, paired)")
    logger.info(f"{'comparison':24s} {'dAUC':>9s} {'95% CI':>22s} {'dAcc':>9s} {'95% CI':>22s}")
    for k, v in boot.items():
        logger.info(f"{k:24s} {v['auc']['delta']:+9.5f} "
                    f"[{v['auc']['ci95'][0]:+.5f},{v['auc']['ci95'][1]:+.5f}] "
                    f"{v['accuracy']['delta']:+9.5f} "
                    f"[{v['accuracy']['ci95'][0]:+.5f},{v['accuracy']['ci95'][1]:+.5f}]")

    # --- threshold-freedom accounting --------------------------------------- #
    thr = {}
    for tag, probs in (("p_param", p_param), ("p_final_loso", p_final),
                       ("p_ens_half", p_ens), ("p_img", pool["p_img"])):
        t, a = best_threshold(labels, probs)
        thr[tag] = {"acc_at_0.5": round(float(((probs >= 0.5).astype(int) == labels).mean()), 4),
                    "oof_optimal_threshold": t, "acc_at_optimal_IN_SAMPLE": round(a, 4)}

    # --- temperature sensitivity -------------------------------------------- #
    t_rows: List[Dict] = []
    if args.t_sweep:
        logger.info("")
        logger.info("Temperature sensitivity of the foreign vote (primary config only). "
                    "T=0.07 was fixed on exp3n's key space, whose top1-top15 similarity "
                    "spread is ~0.001; CTransPath's is ~0.07, so the SAME T is a sharper rule.")
        for T in T_SWEEP:
            r, pl, go = run_config(f"{PRIMARY} T={T}", "features", "none", base_caches, foreign,
                                   experiment, args.foreign_encoder, rcfg, folds, seed, epochs,
                                   lr, k_img, device, logger, temperature=T)
            t_rows.append({
                "temperature": T,
                "p_img_auc": r["p_img"]["image_auc"],
                "awrong": r["complementarity"]["auc_where_param_wrong"],
                "corr_param_img": r["complementarity"]["corr_param_img"],
                "gate_w_img": r["gate_mean_weights"][1],
                "delta_auc_loso": r["delta_auc_final_loso_vs_param"],
                "delta_acc_loso": r["delta_acc_final_loso_vs_param"],
                "mean_top1_sim": r["neighbourhood"]["mean_top1_sim"],
            })
            logger.info(f"  T={T:<5} p_img auc={r['p_img']['image_auc']:.4f} "
                        f"awrong={r['complementarity']['auc_where_param_wrong']:.4f} "
                        f"w_img={r['gate_mean_weights'][1]:.4f} "
                        f"dAUC={r['delta_auc_final_loso_vs_param']:+.4f} "
                        f"dAcc={r['delta_acc_final_loso_vs_param']:+.4f}")

    # --- artefact checks: were the D3-D7 knobs mis-specified for a foreign key? -- #
    # route/cap/k were fixed by measurement on exp3n's near-rank-1 space, where the
    # unrouted same-magnification neighbour rate is 0.41. CTransPath's is 0.74, so
    # 'same_mag' discards far less there and the cap binds differently. If the null
    # survives every one of these, it is not an operating-point artefact.
    art_rows: List[Dict] = []
    if args.artefact_checks:
        logger.info("")
        logger.info("Artefact checks on the primary config (NOT candidate configurations -- "
                    "reported whatever they say; D3-D7 remain at their measured values).")
        checks = (("route=all", dict(route="all")),
                  ("route=cross_mag", dict(route="cross_mag")),
                  ("cap=1", dict(cap=1)), ("cap=15 (no cap)", dict(cap=15)),
                  ("k=5", dict(k=5)), ("k=30", dict(k=30)))
        for label, kw in checks:
            r, _, g = run_config(f"{PRIMARY} [{label}]", "features", "none", base_caches,
                                 foreign, experiment, args.foreign_encoder, rcfg, folds,
                                 seed, epochs, lr, k_img, device, logger, **kw)
            b = patient_bootstrap(labels, pids, g["p_final_loso"], p_ens, args.resamples, seed)
            art_rows.append({
                "check": label,
                "p_img_auc": r["p_img"]["image_auc"],
                "awrong": r["complementarity"]["auc_where_param_wrong"],
                "p_final_loso_auc": r["p_final_loso"]["image_auc"],
                "delta_auc_vs_param": r["delta_auc_final_loso_vs_param"],
                "delta_auc_vs_ens_half": b["point_delta_auc"],
                "ci95_vs_ens_half": b["auc"]["ci95"],
                "beats_ens_half": bool(r["p_final_loso"]["image_auc"] > ens["p_ens_half"]["image_auc"]),
            })
            logger.info(f"  {label:16s} p_img auc={r['p_img']['image_auc']:.4f} "
                        f"awrong={r['complementarity']['auc_where_param_wrong']:.4f} "
                        f"p_final={r['p_final_loso']['image_auc']:.4f} "
                        f"(vs param {r['delta_auc_final_loso_vs_param']:+.4f}, "
                        f"vs ensemble {b['point_delta_auc']:+.4f} "
                        f"[{b['auc']['ci95'][0]:+.4f},{b['auc']['ci95'][1]:+.4f}])")

    # --- persist ------------------------------------------------------------ #
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "preregistration": {
            "hypothesis": "Error-decorrelation from the parametric head (awrong) is a property "
                          "of encoder identity, not of key definition; a bank encoded by a "
                          "pathology foundation model with a measurably different error profile "
                          "should produce awrong above the exp1 cross-encoder band (0.127-0.136).",
            "primary_endpoints": ["auc_where_param_wrong", "gate weight on p_img",
                                  "subtype_lift", "neighbour_overlap_vs_same_encoder"],
            "secondary_endpoint": "image AUC/accuracy of p_final_loso",
            "success_criterion": "p_final must beat BOTH p_param AND the two-probe ensemble "
                                 "(half and LOSO-fitted) on the same surface.",
            "test_set": "NOT touched by this script.",
            "held_fixed": "route/k/cap/two-level ranking (D3-D7); gate protocol (pooled OOF, "
                          "5->16->3, same seed/epochs/lr); T=0.07 primary with a reported sweep.",
            "selection_bias": "p_final is fitted and scored on the same pooled-OOF rows; "
                              "p_final_loso is the honest ordering. Neither replaces a held-out "
                              "confirmation.",
        },
        "base_experiment": experiment, "base_encoder": encoder,
        "foreign_encoder": args.foreign_encoder, "foreign_cache": fcache,
        "folds": list(folds), "alignment": align,
        "results": results,
        "ensemble_control": ens,
        "bootstrap": boot,
        "threshold_freedom": thr,
        "temperature_sweep": t_rows,
        "artefact_checks": art_rows,
    }
    out_path = out_dir / "crossencoder_screen.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    np.savez_compressed(
        out_dir / "probs_primary.npz",
        fold=pool["fold"], label=labels, patient_id=pids, image_path=pool["image_path"],
        prob_param=p_param, prob_img=pool["p_img"], prob_slide=pool["p_slide"],
        prob_final=gates[PRIMARY]["p_final"], prob_final_loso=p_final,
        prob_probe=p_probe, prob_ens_half=p_ens, prob_ens_fitted=p_ens_fit,
    )
    logger.info(f"\nSaved -> {out_path}")

    header = (f"{'configuration':38s} {'enc':>10s} {'awrong':>7s} {'corr':>6s} {'pImgAUC':>8s} "
              f"{'w_img':>6s} {'dAUC':>8s} {'dAcc':>8s} {'dPat':>8s} {'slift':>7s}")
    logger.info("\n" + header)
    logger.info("-" * len(header))
    for r in results:
        logger.info(
            f"{r['name'][:38]:38s} {str(r['key_encoder'])[:10]:>10s} "
            f"{r['complementarity']['auc_where_param_wrong']:7.4f} "
            f"{r['complementarity']['corr_param_img']:6.3f} {r['p_img']['image_auc']:8.4f} "
            f"{r['gate_mean_weights'][1]:6.3f} {r['delta_auc_final_loso_vs_param']:+8.4f} "
            f"{r['delta_acc_final_loso_vs_param']:+8.4f} "
            f"{r['delta_patacc_final_loso_vs_param']:+8.4f} "
            f"{r['neighbourhood']['subtype_lift']:+7.3f}")


if __name__ == "__main__":
    main()
