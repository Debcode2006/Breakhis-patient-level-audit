"""Retrieval **key** ablation -- what should the memory index on?

Why this exists
---------------
``docs/retrieval.md`` Part VII established that Retrieval Memory v1 cannot beat the
parametric head on exp3n because of *what it keys on*: the pooled 1024-d
``features`` vector is the classifier's own input, it is near-rank-1 (effective
rank 1.19 / 1024), and a cosine-kNN vote over it is a noisier re-reading of the
one axis the linear head already reads optimally. That report listed the untested
alternatives; this script measures all of them under one protocol.

Two stages, so a ~30-configuration sweep costs one forward pass per fold
-----------------------------------------------------------------------
``--stage cache``  encode each fold's bank split (its TRAIN patients, eval
    transforms) and its validation split ONCE with the frozen exp3n checkpoint,
    storing every *pooled component* of the forward dict: per-FPN-level GAP / max
    / std / GeM, the SupCon ``projections``, the classifier logits, and the linear
    head's decision direction. -> ``analysis/retrieval_keys/<exp>/cache_fold<k>.npz``

``--stage eval``   for every (key, key_transform) in the grid: rebuild the real
    :class:`~rap_mst.retrieval.bank.MemoryBank` from the cached vectors (image rows
    -> fit transform -> derive slide centroids), query each fold's validation split
    through the real :class:`~rap_mst.retrieval.memory.RetrievalMemory`, pool the
    five folds' out-of-fold rows and fit the real
    :class:`~rap_mst.retrieval.gate.FusionGate` exactly as Stage B2 does.
    -> ``analysis/retrieval_keys/<exp>/key_ablation.json``

Because the bank, the vote, the per-patient cap, the two-level ranking and the
gate are the *production* classes, a configuration that wins here transfers to
``build_memory_bank.py`` + ``train_gate.py`` by setting ``retrieval.key`` and
``retrieval.key_transform`` -- nothing else changes.

Protocol notes (read before believing a number)
-----------------------------------------------
* Fold *k*'s bank holds only fold *k*'s TRAIN patients; ``assert_disjoint`` is
  re-run for every configuration, and queries additionally block their own
  patient. The key transform is fitted on the bank's rows only, so it never sees
  a validation image.
* The reported numbers are **pooled out-of-fold validation**, the same surface
  Stage B2 fits on. Sweeping ~30 configurations against it is model selection:
  the winner's pooled-OOF gain is optimistically biased, and the honest test of
  the winner is the untouched 16-patient test set, run **once**, through the
  normal pipeline. This script never touches the test split.
* D3-D7 (route, k, cap, temperature, two-level ranking) are held at their
  measured values throughout. This ablation varies the **key** only.

Examples
--------
    python scripts/retrieval_key_ablation.py --stage cache
    python scripts/retrieval_key_ablation.py --stage eval
    python scripts/retrieval_key_ablation.py --stage eval --only features,projections
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rap_mst.data.breakhis import subtype_from_patient_id  # noqa: E402
from rap_mst.data.datamodule import BreaKHisDataModule  # noqa: E402
from rap_mst.data.splits import get_fold  # noqa: E402
from rap_mst.experiments import encoder_experiment, experiment_names  # noqa: E402
from rap_mst.models.builder import build_model  # noqa: E402
from rap_mst.retrieval.bank import LEVEL_IMAGE, BankEntry, MemoryBank  # noqa: E402
from rap_mst.retrieval.builder import retrieval_cfg  # noqa: E402
from rap_mst.retrieval.gate import FusionGate, fit_gate  # noqa: E402
from rap_mst.retrieval.keys import (  # noqa: E402
    POOLINGS, assert_gap_matches_features, extract_key_from_pooled, pool_map,
)
from rap_mst.retrieval.memory import LevelConfig, RetrievalMemory  # noqa: E402
from rap_mst.retrieval.transform import classifier_direction  # noqa: E402
from rap_mst.utils import reporting  # noqa: E402
from rap_mst.utils.checkpoint import load_checkpoint  # noqa: E402
from rap_mst.utils.config import (  # noqa: E402
    apply_overrides, load_config, merge_section, parse_set_overrides,
)
from rap_mst.utils.logging_utils import get_logger  # noqa: E402
from rap_mst.utils.metrics import compute_metrics  # noqa: E402
from rap_mst.utils.runs import available_folds, find_fold_run  # noqa: E402
from rap_mst.utils.seed import seed_everything  # noqa: E402

# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #
#: (name, key spec, key_transform). Keep the D1 baseline first: it must reproduce
#: gate_fit.json bit-for-bit, which is what validates the whole offline harness.
GRID: Tuple[Tuple[str, str, str], ...] = (
    # --- the D1 baseline and the two other forward-dict vectors --------------
    ("features (D1 baseline)",      "features",            "none"),
    ("projections (SupCon)",        "projections",         "none"),
    # --- what pooling throws away: dispersion, extremes, soft-max -----------
    ("fpn.std",                     "fpn.std",             "none"),
    ("fpn.max",                     "fpn.max",             "none"),
    ("fpn.gem",                     "fpn.gem",             "none"),
    # --- one pyramid level at a time (0 = finest / most textural) -----------
    ("fpn.gap@0",                   "fpn.gap@0",           "none"),
    ("fpn.gap@1",                   "fpn.gap@1",           "none"),
    ("fpn.gap@2",                   "fpn.gap@2",           "none"),
    ("fpn.gap@3",                   "fpn.gap@3",           "none"),
    ("fpn.std@0",                   "fpn.std@0",           "none"),
    ("fpn.std@3",                   "fpn.std@3",           "none"),
    # --- level balance: is the concat cosine just one level? ----------------
    ("fpn.gap.ln",                  "fpn.gap.ln",          "none"),
    # --- composites ---------------------------------------------------------
    ("features+fpn.std",            "features+fpn.std",    "none"),
    ("features+projections",        "features+projections", "none"),
    # --- geometry fixes on the D1 key (no retraining) -----------------------
    # pca_drop:n is swept, not picked: the whole curve is reported so the choice
    # of n is visible as a trend rather than a single tuned point.
    ("features  center",            "features",            "center"),
    ("features  pca_drop:1",        "features",            "pca_drop:1"),
    ("features  pca_drop:3",        "features",            "pca_drop:3"),
    ("features  pca_drop:10",       "features",            "pca_drop:10"),
    ("features  pca_drop:20",       "features",            "pca_drop:20"),
    ("features  pca_drop:50",       "features",            "pca_drop:50"),
    ("features  pca_drop:100",      "features",            "pca_drop:100"),
    ("features  pca_drop:200",      "features",            "pca_drop:200"),
    ("features  whiten:64",         "features",            "whiten:64"),
    ("features  whiten:128",        "features",            "whiten:128"),
    ("features  whiten:256",        "features",            "whiten:256"),
    ("features  whiten:512",        "features",            "whiten:512"),
    ("features  pca_drop:3+whiten", "features",            "pca_drop:3,whiten:128"),
    # --- delete exactly what the linear head reads --------------------------
    ("features  drop_clf_dir",      "features",            "center,drop_dirs"),
    ("features  drop_clf+whiten:128", "features",          "center,drop_dirs,whiten:128"),
    # --- the same fixes on the runners-up -----------------------------------
    ("projections whiten:64",       "projections",         "whiten:64"),
    ("projections pca_drop:1",      "projections",         "pca_drop:1"),
    ("projections pca_drop:10",     "projections",         "pca_drop:10"),
    ("fpn.std  whiten:128",         "fpn.std",             "whiten:128"),
    ("fpn.std  pca_drop:1",         "fpn.std",             "pca_drop:1"),
    ("fpn.std  pca_drop:10",        "fpn.std",             "pca_drop:10"),
    ("fpn.gap@0 whiten:128",        "fpn.gap@0",           "whiten:128"),
    ("fpn.gap@0 pca_drop:10",       "fpn.gap@0",           "pca_drop:10"),
    ("features+fpn.std whiten:128", "features+fpn.std",    "whiten:128"),
    ("features+fpn.std pca_drop:10", "features+fpn.std",   "pca_drop:10"),
)

#: The cross-encoder control (``--cross-encoder exp1``). Identical protocol, except
#: the bank and the query keys are produced by a **different** frozen encoder while
#: ``p_param`` still comes from exp3n's own head. If every same-encoder key stays
#: redundant with ``p_param`` but a foreign-encoder key does not, the ceiling is the
#: *shared encoder*, not the choice of key -- which is a different fix entirely.
CROSS_GRID: Tuple[Tuple[str, str, str], ...] = (
    ("X-enc features",              "features",            "none"),
    ("X-enc features pca_drop:10",  "features",            "pca_drop:10"),
    ("X-enc features whiten:128",   "features",            "whiten:128"),
    ("X-enc fpn.std",               "fpn.std",             "none"),
)

#: Pooled components written to the cache. ``gap`` is what ``FeatureFusion`` uses
#: (so ``fpn.gap`` == ``features``); the rest are the operations GAP destroys.
CACHED_POOLINGS = POOLINGS

QUERY_BATCH = 256          # the per-patient cap is a python scan; batch the queries
DIAG_SUBSAMPLE = 600       # val images per fold used for the route='all' diagnostics


# --------------------------------------------------------------------------- #
# Stage 1 -- cache
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_split(model, loader, device, amp: bool, logger, tag: str) -> Dict[str, np.ndarray]:
    """One pass -> every pooled component of the forward dict, for later re-keying."""
    pooled: Dict[str, List[List[np.ndarray]]] = {p: [] for p in CACHED_POOLINGS}
    direct: Dict[str, List[np.ndarray]] = {"projections": []}
    logits, labels, mags = [], [], []
    pids: List[str] = []
    autocast = torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda")
    n = 0
    for i, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        mag_index = batch["mag_index"].to(device, non_blocking=True)
        with autocast:
            out = model(images, mag_index)
        if i == 0:
            # The identity that makes pyramid keys comparable to the D1 baseline.
            assert_gap_matches_features(out)
        pyramid = out["fpn_features"]
        for p in CACHED_POOLINGS:
            pooled[p].append([pool_map(m.float(), p).cpu().numpy() for m in pyramid])
        proj = out.get("projections")
        direct["projections"].append(proj.float().cpu().numpy() if proj is not None else None)
        logits.append(out["logits"].float().cpu().numpy())
        labels.append(batch["label"].numpy())
        mags.append(batch["mag_index"].numpy())
        pids.extend(batch["patient_id"])
        n += images.shape[0]
        if i % 50 == 0:
            logger.info(f"  [{tag}] encoded {n} images ...")

    n_levels = len(pooled[CACHED_POOLINGS[0]][0])
    data: Dict[str, np.ndarray] = {}
    for p in CACHED_POOLINGS:
        for lvl in range(n_levels):
            data[f"pool_{p}_{lvl}"] = np.concatenate([b[lvl] for b in pooled[p]], axis=0)
    if direct["projections"][0] is not None:
        data["projections"] = np.concatenate(direct["projections"], axis=0)
    data["logits"] = np.concatenate(logits, axis=0)
    data["label"] = np.concatenate(labels, axis=0)
    data["mag_index"] = np.concatenate(mags, axis=0)
    data["patient_id"] = np.asarray(pids)
    data["subtype"] = np.asarray([subtype_from_patient_id(p) for p in pids])
    data["n_levels"] = np.asarray(n_levels)
    return data


def stage_cache(args, base_cfg, experiment: str, encoder: str, folds: Sequence[int],
                out_dir: Path, device: torch.device, logger) -> None:
    log_root = Path(base_cfg.logging.log_root)
    for fold in folds:
        out_path = out_dir / f"cache_fold{fold}.npz"
        if out_path.exists() and not args.overwrite:
            logger.info(f"fold {fold}: cache exists, skipping ({out_path.name}). Use --overwrite.")
            continue

        run_dir = find_fold_run(encoder, log_root, fold)
        cfg = load_config(str(run_dir / "config.yaml"))
        cfg = merge_section(cfg, "retrieval", base_cfg)
        cfg = apply_overrides(cfg, parse_set_overrides(args.set))
        seed_everything(cfg.seed, deterministic=cfg.deterministic)

        model = build_model(cfg).to(device)
        ckpt = load_checkpoint(run_dir / "checkpoints" / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        dm = BreaKHisDataModule(cfg)
        dm.setup_bank(fold)          # TRAIN patients, eval transforms, in order
        dm.setup_fold(fold)          # + the fold's validation split
        fold_rec = get_fold(dm.splits, fold)
        train_patients = set(fold_rec["train_patients"])
        val_patients = set(fold_rec["val_patients"])
        test_patients = set(dm.splits["test_patients"])
        if train_patients & val_patients or (train_patients | val_patients) & test_patients:
            raise AssertionError("LEAKAGE: fold splits overlap in the splits file.")

        reporting.section(logger, f"CACHE  fold {fold}", [
            reporting.kv("checkpoint", str(run_dir / "checkpoints" / "best.pt")),
            reporting.kv("bank split", f"{len(dm.bank_set)} images / {len(train_patients)} train patients"),
            reporting.kv("val split", f"{len(dm.val_set)} images / {len(val_patients)} val patients"),
            reporting.kv("output", str(out_path)),
        ])

        amp = bool(cfg.train.mixed_precision) and device.type == "cuda"
        bank_data = encode_split(model, dm.bank_dataloader(), device, amp, logger, f"fold{fold}/bank")
        val_data = encode_split(model, dm.val_dataloader(), device, amp, logger, f"fold{fold}/val")

        clf_dir = classifier_direction(model, "features")
        payload = {f"bank_{k}": v for k, v in bank_data.items()}
        payload.update({f"val_{k}": v for k, v in val_data.items()})
        payload["clf_dir"] = (clf_dir.numpy() if clf_dir is not None
                              else np.zeros((0, 0), dtype=np.float32))
        payload["meta"] = np.asarray(json.dumps({
            "experiment": experiment, "encoder": encoder, "fold": fold,
            "run_dir": str(run_dir), "poolings": list(CACHED_POOLINGS),
            "created": datetime.now().isoformat(timespec="seconds"),
        }))
        np.savez_compressed(out_path, **payload)
        logger.info(f"fold {fold}: cached -> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# --------------------------------------------------------------------------- #
# Stage 2 -- eval
# --------------------------------------------------------------------------- #
class FoldCache:
    """Cached pooled vectors for one fold, re-keyable without touching the GPU."""

    def __init__(self, path: Path, device: torch.device) -> None:
        d = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(d["meta"]))
        self.fold = int(self.meta["fold"])
        self.device = device
        self.n_levels = int(d["bank_n_levels"])
        self._d = {k: d[k] for k in d.files}
        clf = self._d["clf_dir"]
        self.clf_dir = torch.as_tensor(clf) if clf.size else None

    def _t(self, name: str) -> torch.Tensor:
        return torch.as_tensor(self._d[name], dtype=torch.float32, device=self.device)

    def keys(self, split: str, spec: str) -> torch.Tensor:
        direct = {"features": None, "embeddings": None, "projections": None}
        if f"{split}_projections" in self._d:
            direct["projections"] = self._t(f"{split}_projections")
        pooled = {
            p: [self._t(f"{split}_pool_{p}_{lvl}") for lvl in range(self.n_levels)]
            for p in self.meta["poolings"]
        }
        # 'features' == concat of per-level GAP (asserted at cache time), and on
        # exp3n 'embeddings' == 'features' (no magnification block).
        gap_concat = torch.cat(pooled["gap"], dim=1)
        direct["features"] = gap_concat
        direct["embeddings"] = gap_concat
        return extract_key_from_pooled(spec, direct, pooled)

    def col(self, split: str, name: str) -> np.ndarray:
        return self._d[f"{split}_{name}"]

    def p_param(self, split: str) -> np.ndarray:
        logits = self._d[f"{split}_logits"].astype(np.float64)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (e[:, 1] / e.sum(axis=1))


def build_bank(cache: FoldCache, spec: str, transform: str, experiment: str) -> MemoryBank:
    """The production :class:`MemoryBank`, filled from cached vectors."""
    keys = cache.keys("bank", spec)
    labels = cache.col("bank", "label")
    pids = cache.col("bank", "patient_id")
    subs = cache.col("bank", "subtype")
    mags = cache.col("bank", "mag_index")
    bank = MemoryBank(key=spec, key_transform=transform, meta={
        "experiment": experiment, "fold": cache.fold, "source": "retrieval_key_ablation cache",
    })
    bank.add(keys.cpu(), [
        BankEntry(level=LEVEL_IMAGE, label=int(labels[i]), subtype=str(subs[i]),
                  patient_id=str(pids[i]), mag_index=int(mags[i]))
        for i in range(len(labels))
    ])
    bank.fit_key_transform(aux_dirs=cache.clf_dir)
    bank.derive_slide_rows()
    return bank.to(cache.device)


def make_memory(bank: MemoryBank, rcfg, spec: str, transform: str) -> RetrievalMemory:
    """Level configuration straight from ``config.yaml`` -- D3-D7 are never varied here."""
    levels = getattr(rcfg, "levels", None)
    return RetrievalMemory(
        bank=bank,
        key=spec,
        key_transform=transform,
        image=LevelConfig.from_cfg(getattr(levels, "image", None),
                                   enabled=True, route="same_mag", k=15,
                                   per_patient_cap=3, temperature=0.07),
        slide=LevelConfig.from_cfg(getattr(levels, "slide", None),
                                   enabled=True, route="all", k=5,
                                   per_patient_cap=1, temperature=0.07),
        merge_levels=False,
        block_query_patients=True,
    )


def query_val(memory: RetrievalMemory, cache: FoldCache, spec: str) -> Dict[str, np.ndarray]:
    keys = cache.keys("val", spec)
    pids = cache.col("val", "patient_id")
    mags = torch.as_tensor(cache.col("val", "mag_index"), dtype=torch.long, device=cache.device)
    out: Dict[str, List[np.ndarray]] = {k: [] for k in
                                        ("p_img", "p_slide", "agreement", "top1_sim", "n_distinct")}
    for start in range(0, keys.shape[0], QUERY_BATCH):
        stop = min(start + QUERY_BATCH, keys.shape[0])
        ev = memory(keys[start:stop], mags[start:stop],
                    query_patient_ids=list(pids[start:stop]))
        out["p_img"].append(ev["p_img"].cpu().numpy())
        out["p_slide"].append(ev["p_slide"].cpu().numpy())
        out["agreement"].append(ev["agreement"].cpu().numpy())
        out["top1_sim"].append(ev["top1_sim"].cpu().numpy())
        out["n_distinct"].append(ev["n_distinct_patients"].cpu().numpy())
    return {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def geometry(keys: torch.Tensor, rng: np.random.Generator) -> Dict[str, float]:
    """The anisotropy numbers ``docs/retrieval.md`` Part VII §3, measured on any key.

    Effective rank is the participation ratio of the covariance spectrum -- "how
    many directions does this space really use". The D1 key scores 1.19 out of
    1024, which is the whole diagnosis in one number.
    """
    x = keys.detach().to(torch.float64)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    centred = x - x.mean(0, keepdim=True)
    lam = torch.linalg.eigvalsh(centred.t() @ centred).clamp(min=0.0)
    eff_rank = float((lam.sum() ** 2) / (lam.pow(2).sum() + 1e-30))
    n = x.shape[0]
    sample = torch.as_tensor(rng.choice(n, size=min(3000, n), replace=False), device=x.device)
    sim = x[sample] @ x[sample].t()
    iu = torch.triu_indices(len(sample), len(sample), offset=1, device=x.device)
    return {
        "dim": int(x.shape[1]),
        "effective_rank": round(eff_rank, 3),
        "dominant_axis_share": round(float(lam.max() / lam.sum()), 4),
        "mean_unit_key_norm": round(float(x.mean(0).norm()), 4),
        "random_pair_cos": round(float(sim[iu[0], iu[1]].mean()), 4),
    }


@torch.no_grad()
def route_all_diagnostics(bank: MemoryBank, cache: FoldCache, spec: str,
                          rng: np.random.Generator) -> Dict[str, float]:
    """Magnification lock and subtype lift, measured under ``route='all'``.

    Under the production ``same_mag`` route the same-magnification rate is 1.0 by
    construction and says nothing about the key. Only an unrouted ranking reveals
    whether the *key itself* is magnification-locked (25% is chance) or whether its
    neighbourhoods carry morphology (subtype lift above the bank's base rate).
    """
    keys = cache.keys("val", spec)
    n = keys.shape[0]
    sel = rng.choice(n, size=min(DIAG_SUBSAMPLE, n), replace=False)
    pids = cache.col("val", "patient_id")[sel]
    subs = cache.col("val", "subtype")[sel]
    mags = cache.col("val", "mag_index")[sel]
    res = bank.query(
        keys[sel], mag_index=torch.as_tensor(mags, device=cache.device),
        level=LEVEL_IMAGE, k=15, route="all", per_patient_cap=3, temperature=0.07,
        query_patient_ids=list(pids),
    )
    from rap_mst.constants import IDX_TO_MAG
    want_mag = [IDX_TO_MAG.get(int(m), -1) for m in mags]
    same_mag, same_sub, total = 0, 0, 0
    for b in range(len(sel)):
        for m in res.magnifications[b]:
            same_mag += int(m == want_mag[b])
        for s in res.subtypes[b]:
            same_sub += int(s == subs[b])
        total += len(res.subtypes[b])
    bank_subs = bank.subtypes[bank.levels == LEVEL_IMAGE]
    base = float(np.mean([np.mean(bank_subs == s) for s in subs]))
    return {
        "same_mag_rate_route_all": round(same_mag / max(total, 1), 4),
        "subtype_match_rate": round(same_sub / max(total, 1), 4),
        "subtype_base_rate": round(base, 4),
        "subtype_lift": round(same_sub / max(total, 1) - base, 4),
    }


def auc(labels: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def patient_pool(pids: np.ndarray, labels: np.ndarray, probs: np.ndarray):
    uniq = np.unique(pids)
    p_lab = np.asarray([int(labels[pids == p][0]) for p in uniq])
    p_prob = np.asarray([float(probs[pids == p].mean()) for p in uniq])
    return p_lab, p_prob


def complementarity(labels: np.ndarray, p_param: np.ndarray, p_img: np.ndarray) -> Dict[str, float]:
    """Does the memory know anything the head does not?

    Three angles, none of which a raw AUC comparison shows:
    ``corr`` -- how redundant the two readouts are;
    ``auc_where_param_wrong`` -- can retrieval rescue the head's mistakes at all
    (0.5 means it is wrong in exactly the same places);
    ``best_fixed_alpha`` -- the best achievable *scalar* blend, i.e. an upper bound
    on what any 3-weight gate can extract from this pair.
    """
    wrong = (p_param >= 0.5).astype(int) != labels
    alphas = np.linspace(0.0, 1.0, 101)
    aucs = [auc(labels, a * p_param + (1 - a) * p_img) for a in alphas]
    best = int(np.nanargmax(aucs))
    return {
        "corr_param_img": round(float(np.corrcoef(p_param, p_img)[0, 1]), 4),
        "auc_where_param_wrong": round(auc(labels[wrong], p_img[wrong]), 4)
        if wrong.sum() > 10 and len(np.unique(labels[wrong])) > 1 else float("nan"),
        "n_param_wrong": int(wrong.sum()),
        "best_fixed_alpha": round(float(alphas[best]), 2),
        "best_blend_auc": round(float(aucs[best]), 4),
        "blend_gain_auc": round(float(aucs[best] - auc(labels, p_param)), 4),
    }


def _gate_tensors(pool: Dict[str, np.ndarray], k_img: int, sel=None):
    t = lambda a: torch.as_tensor(a[sel] if sel is not None else a, dtype=torch.float32)  # noqa: E731
    features = FusionGate.build_features(
        t(pool["prob_param"]), t(pool["agreement"]), t(pool["top1_sim"]),
        t(pool["n_distinct"] / max(k_img, 1)),
    )
    probs3 = torch.stack([t(pool["prob_param"]), t(pool["p_img"]), t(pool["p_slide"])], dim=1)
    return features, probs3, t(pool["label"])


def _fit(features, probs3, labels, seed: int, epochs: int, lr: float) -> FusionGate:
    torch.manual_seed(seed)      # the hidden layer is random; fix it before init
    gate = FusionGate(hidden=16, init_weights=(0.5, 0.5, 0.0), term_mask=(1, 1, 1))
    fit_gate(gate, features, probs3, labels, epochs=epochs, lr=lr, log_every=0)
    return gate


def fit_and_score(pool: Dict[str, np.ndarray], seed: int, epochs: int, lr: float,
                  k_img: int, folds: Sequence[int]) -> Dict[str, object]:
    """Two fused scores: the Stage B2 protocol, and a leave-one-fold-out control.

    ``p_final`` is exactly what Stage B2 produces -- one gate fitted on all pooled
    OOF rows -- so it is directly comparable to ``gate_fit.json``. But it is also
    *fitted and scored on the same rows*, which flatters every configuration
    equally and by an unknown amount.

    ``p_final_loso`` removes that: for each fold the gate is refitted on the other
    four folds' rows and applied to this fold's, so no row is ever scored by a gate
    that saw it. It does not remove the bias from *selecting* a configuration on
    this surface -- only the held-out test set does that -- but it is the honest
    number to compare configurations by.
    """
    features, probs3, labels = _gate_tensors(pool, k_img)
    gate = _fit(features, probs3, labels, seed, epochs, lr)
    with torch.no_grad():
        w = gate.weights_from_features(features)
        p_final = (w * probs3).sum(dim=1).numpy().astype(np.float64)

    fold_col = pool["fold"]
    p_loso = np.zeros_like(p_final)
    for held in folds:
        tr = fold_col != held
        te = ~tr
        f_tr, p_tr, y_tr = _gate_tensors(pool, k_img, sel=tr)
        g = _fit(f_tr, p_tr, y_tr, seed, epochs, lr)
        f_te, p_te, _ = _gate_tensors(pool, k_img, sel=te)
        with torch.no_grad():
            p_loso[te] = (g.weights_from_features(f_te) * p_te).sum(dim=1).numpy()

    return {"p_final": p_final, "p_final_loso": p_loso,
            "mean_weights": [round(float(v), 4) for v in w.mean(0)]}


def score_column(labels, pids, probs) -> Dict[str, float]:
    m = compute_metrics(labels, (probs >= 0.5).astype(int), probs)
    p_lab, p_prob = patient_pool(pids, labels, probs)
    pm = compute_metrics(p_lab, (p_prob >= 0.5).astype(int), p_prob)
    return {
        "image_acc": round(m["accuracy"], 4), "image_auc": round(m["auc"], 4),
        "image_sens": round(m["sensitivity"], 4), "image_spec": round(m["specificity"], 4),
        "patient_acc": round(pm["accuracy"], 4), "patient_auc": round(pm["auc"], 4),
    }


def stage_eval(args, base_cfg, experiment: str, folds: Sequence[int], out_dir: Path,
               device: torch.device, logger) -> None:
    rcfg = retrieval_cfg(base_cfg)
    k_img = int(getattr(getattr(getattr(rcfg, "levels", None), "image", None), "k", 15))
    gate_cfg = getattr(rcfg, "gate", None)
    epochs = int(getattr(gate_cfg, "epochs", 400))
    lr = float(getattr(gate_cfg, "lr", 0.01))

    caches = {}
    for fold in folds:
        path = out_dir / f"cache_fold{fold}.npz"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run --stage cache first.")
        caches[fold] = FoldCache(path, device)
        logger.info(f"loaded {path.name}: {len(caches[fold].col('bank', 'label'))} bank rows, "
                    f"{len(caches[fold].col('val', 'label'))} val rows")

    # --- optional cross-encoder control ------------------------------------ #
    xcaches: Dict[int, FoldCache] = {}
    if args.cross_encoder:
        xdir = Path(args.out) / args.cross_encoder
        for fold in folds:
            path = xdir / f"cache_fold{fold}.npz"
            if not path.exists():
                raise SystemExit(
                    f"Missing {path}. Build it with:\n"
                    f"  python scripts/retrieval_key_ablation.py --stage cache "
                    f"--experiment {args.cross_encoder}"
                )
            xc = FoldCache(path, device)
            # The two caches must describe the same images in the same order, or
            # 'p_param from A, keys from B' would silently pair unrelated rows.
            for split in ("bank", "val"):
                if not np.array_equal(xc.col(split, "patient_id"), caches[fold].col(split, "patient_id")) \
                        or not np.array_equal(xc.col(split, "label"), caches[fold].col(split, "label")):
                    raise AssertionError(
                        f"fold {fold}/{split}: {args.cross_encoder} cache is not row-aligned with "
                        f"{experiment}'s. Re-cache both with the same splits file."
                    )
            xcaches[fold] = xc
            logger.info(f"cross-encoder cache {path.name} loaded and row-aligned with {experiment}")

    grid = list(GRID) + [g for g in (CROSS_GRID if args.cross_encoder else ())]
    if args.only:
        wanted = {w.strip() for w in args.only.split(",")}
        grid = [g for g in grid if g[1] in wanted or g[0] in wanted]
        if not grid:
            raise SystemExit(f"--only '{args.only}' matched nothing in the grid.")

    results: List[Dict] = []
    baseline: Optional[Dict] = None
    for gi, (name, spec, transform) in enumerate(grid, start=1):
        rng = np.random.default_rng(base_cfg.seed)
        logger.info("")
        logger.info(f"[{gi}/{len(grid)}] key={spec!r} transform={transform!r}  ({name})")
        parts: List[Dict[str, np.ndarray]] = []
        geo: List[Dict[str, float]] = []
        diag: List[Dict[str, float]] = []
        cross = name.startswith("X-enc")
        try:
            for fold in folds:
                cache = caches[fold]                       # always owns p_param + labels
                kcache = xcaches[fold] if cross else cache  # owns the retrieval keys
                bank = build_bank(kcache, spec, transform, experiment)
                bank.assert_disjoint(set(cache.col("val", "patient_id").tolist()), what="validation")
                memory = make_memory(bank, rcfg, spec, transform)
                ev = query_val(memory, kcache, spec)
                parts.append({
                    "fold": np.full(len(ev["p_img"]), fold),
                    "label": cache.col("val", "label").astype(int),
                    "patient_id": cache.col("val", "patient_id"),
                    "prob_param": cache.p_param("val"),
                    **ev,
                })
                img_keys = bank.keys[torch.as_tensor(bank.levels == LEVEL_IMAGE,
                                                     device=bank.keys.device)]
                geo.append(geometry(img_keys, rng))
                diag.append(route_all_diagnostics(bank, kcache, spec, rng))
                del bank, memory
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception as exc:                     # a bad spec must not kill the sweep
            logger.error(f"    FAILED: {type(exc).__name__}: {exc}")
            results.append({"name": name, "key": spec, "key_transform": transform,
                            "error": f"{type(exc).__name__}: {exc}"})
            continue

        pool = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        labels = pool["label"].astype(int)
        pids = pool["patient_id"]
        gate_out = fit_and_score(pool, base_cfg.seed, epochs, lr, k_img, folds)

        row = {
            "name": name, "key": spec, "key_transform": transform,
            "key_encoder": args.cross_encoder if cross else experiment,
            "n_rows": int(len(labels)), "n_patients": int(len(set(pids.tolist()))),
            "geometry": {k: round(float(np.mean([g[k] for g in geo])), 4) for k in geo[0]},
            "neighbourhood": {
                **{k: round(float(np.mean([d[k] for d in diag])), 4) for k in diag[0]},
                "mean_top1_sim": round(float(pool["top1_sim"].mean()), 4),
                "mean_distinct_patients": round(float(pool["n_distinct"].mean()), 3),
            },
            "p_param": score_column(labels, pids, pool["prob_param"]),
            "p_img": score_column(labels, pids, pool["p_img"]),
            "p_slide": score_column(labels, pids, pool["p_slide"]),
            "p_final": score_column(labels, pids, gate_out["p_final"]),
            "p_final_loso": score_column(labels, pids, gate_out["p_final_loso"]),
            "gate_mean_weights": gate_out["mean_weights"],
            "complementarity": complementarity(labels, pool["prob_param"], pool["p_img"]),
        }
        for tag in ("p_final", "p_final_loso"):
            suffix = "" if tag == "p_final" else "_loso"
            row[f"delta_auc_final{suffix}_vs_param"] = round(
                row[tag]["image_auc"] - row["p_param"]["image_auc"], 4)
            row[f"delta_acc_final{suffix}_vs_param"] = round(
                row[tag]["image_acc"] - row["p_param"]["image_acc"], 4)
            row[f"delta_patacc_final{suffix}_vs_param"] = round(
                row[tag]["patient_acc"] - row["p_param"]["patient_acc"], 4)
        results.append(row)
        if baseline is None:
            baseline = row
        if args.dump_probs:
            # Per-row columns so the report's confidence intervals can be recomputed
            # (patient-clustered bootstrap) without re-running the sweep.
            slug = f"{spec}__{transform}".replace("/", "_").replace(":", "-").replace(",", "+")
            np.savez_compressed(
                out_dir / f"probs_{'x' + str(args.cross_encoder) + '_' if cross else ''}{slug}.npz",
                fold=pool["fold"], label=labels, patient_id=pids,
                prob_param=pool["prob_param"], prob_img=pool["p_img"],
                prob_slide=pool["p_slide"], prob_final=gate_out["p_final"],
                prob_final_loso=gate_out["p_final_loso"],
            )

        logger.info(
            f"    dim={row['geometry']['dim']:5.0f} eff_rank={row['geometry']['effective_rank']:7.2f} "
            f"| p_img auc={row['p_img']['image_auc']:.4f} acc={row['p_img']['image_acc']:.4f} "
            f"| p_final auc={row['p_final']['image_auc']:.4f} ({row['delta_auc_final_vs_param']:+.4f}) "
            f"| LOSO auc={row['p_final_loso']['image_auc']:.4f} "
            f"({row['delta_auc_final_loso_vs_param']:+.4f}) "
            f"acc={row['p_final_loso']['image_acc']:.4f} "
            f"({row['delta_acc_final_loso_vs_param']:+.4f}) "
            f"pat={row['p_final_loso']['patient_acc']:.4f}"
        )
        logger.info(
            f"    corr(param,img)={row['complementarity']['corr_param_img']:.3f} "
            f"auc|param wrong={row['complementarity']['auc_where_param_wrong']:.3f} "
            f"best blend auc={row['complementarity']['best_blend_auc']:.4f} "
            f"(alpha={row['complementarity']['best_fixed_alpha']:.2f}, "
            f"{row['complementarity']['blend_gain_auc']:+.4f}) "
            f"| w={row['gate_mean_weights']} "
            f"| mag_lock={row['neighbourhood']['same_mag_rate_route_all']:.2f} "
            f"subtype_lift={row['neighbourhood']['subtype_lift']:+.3f}"
        )

    out_path = out_dir / (f"key_ablation_cross_{args.cross_encoder}.json"
                          if args.cross_encoder else "key_ablation.json")
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "experiment": experiment, "folds": list(folds),
            "protocol": {
                "surface": "pooled out-of-fold validation (5 folds, 66 CV patients)",
                "held_fixed": "route/k/cap/temperature/two-level ranking (D3-D7) and the "
                              "gate fit protocol (pooled OOF, 5->16->3, same seed/epochs/lr)",
                "varied": "retrieval.key and retrieval.key_transform only",
                "selection_bias": "every row is scored on the surface the gate is fitted on; "
                                  "the winner must be confirmed once on the held-out test set",
            },
            "created": datetime.now().isoformat(timespec="seconds"),
            "results": results,
        }, fh, indent=2)
    logger.info(f"\nSaved -> {out_path}")
    _summary_table(logger, results)


def _summary_table(logger, results: List[Dict]) -> None:
    ok = [r for r in results if "error" not in r]
    if not ok:
        return
    header = (f"{'key / transform':34s} {'dim':>5s} {'eff':>6s} {'pImgAUC':>8s} "
              f"{'LOSOauc':>8s} {'dAUC':>7s} {'dAcc':>7s} {'dPatAcc':>8s} {'corr':>6s}")
    logger.info("\nRanked by leave-one-fold-out fused AUC (the honest ordering)")
    logger.info(header)
    logger.info("-" * len(header))
    for r in sorted(ok, key=lambda x: -x["p_final_loso"]["image_auc"]):
        prefix = "" if r["name"][:5] != "X-enc" else f"[{r.get('key_encoder', '?')}] "
        label = f"{prefix}{r['key']} / {r['key_transform']}"
        logger.info(
            f"{label[:34]:34s} {r['geometry']['dim']:5.0f} {r['geometry']['effective_rank']:6.2f} "
            f"{r['p_img']['image_auc']:8.4f} {r['p_final_loso']['image_auc']:8.4f} "
            f"{r['delta_auc_final_loso_vs_param']:+7.4f} "
            f"{r['delta_acc_final_loso_vs_param']:+7.4f} "
            f"{r['delta_patacc_final_loso_vs_param']:+8.4f} "
            f"{r['complementarity']['corr_param_img']:6.3f}"
        )


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Ablate what the Retrieval Memory keys on.")
    ap.add_argument("--stage", choices=("cache", "eval", "all"), default="all")
    ap.add_argument("--experiment", default=None, choices=experiment_names())
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="analysis/retrieval_keys")
    ap.add_argument("--only", default=None, help="Comma-separated key specs to evaluate.")
    ap.add_argument("--cross-encoder", default=None,
                    help="Also run the CROSS_GRID with keys from this experiment's cache "
                         "(e.g. exp1) while p_param still comes from the base encoder.")
    ap.add_argument("--overwrite", action="store_true", help="Re-encode existing caches.")
    ap.add_argument("--dump-probs", action="store_true",
                    help="Also write per-row probability columns per configuration.")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    if retrieval_cfg(base_cfg) is None:
        raise SystemExit(f"No `retrieval:` block in {args.config}.")
    experiment = args.experiment or str(getattr(retrieval_cfg(base_cfg), "base_experiment", "exp3n"))
    encoder = encoder_experiment(experiment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_root = Path(base_cfg.logging.log_root)
    folds = args.folds if args.folds else available_folds(encoder, log_root)
    if not folds:
        raise SystemExit(f"No trained folds for encoder '{encoder}' under {log_root}.")

    out_dir = Path(args.out) / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("rap_mst.retrieval_key_ablation", out_dir / "key_ablation.log")
    # The D5 "one bank -> two rankings" line is verified per RetrievalMemory
    # instance; across 30 configurations x 5 folds that is 150 identical lines.
    # The assertions still run -- only the confirmation chatter is muted.
    logging.getLogger("rap_mst.retrieval.memory").setLevel(logging.WARNING)

    env = reporting.collect_env(device)
    reporting.section(logger, "RETRIEVAL KEY ABLATION", [
        reporting.kv("experiment", f"{experiment}  (encoder: {encoder})"),
        reporting.kv("folds", folds),
        reporting.kv("stage", args.stage),
        reporting.kv("grid size", f"{len(GRID)} (key, key_transform) configurations"),
        reporting.kv("held fixed", "route/k/cap/T (D3-D7) + the Stage B2 gate protocol"),
        reporting.kv("surface", "pooled out-of-fold validation -- the test set is NOT touched"),
        reporting.kv("output dir", str(out_dir)),
        reporting.kv("device", f"{env['device']}  ({env['gpu_name']})"),
    ])

    if args.stage in ("cache", "all"):
        stage_cache(args, base_cfg, experiment, encoder, folds, out_dir, device, logger)
    if args.stage in ("eval", "all"):
        stage_eval(args, base_cfg, experiment, folds, out_dir, device, logger)


if __name__ == "__main__":
    main()
