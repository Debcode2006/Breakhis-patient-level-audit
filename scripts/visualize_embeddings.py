"""Direct evidence that SupCon (exp3) reshapes the embedding space vs the baseline
(exp1) — UMAP projections + quantitative cluster-separation metrics on the
held-out test set.

Why this exists
---------------
`docs/results/classifier_ladder.md` / `docs/results/threshold_calibration.md` showed exp3's advantage is a
*ranking* gain (AUC 0.9607, best) that a fixed 0.5 threshold hides. AUC is only
*indirect* evidence that SupCon shaped the representation. This script gives the
*direct* evidence: it extracts the per-image `embeddings` vector — the exact vector
a future Retrieval/Prototype module will index (`docs/results/classifier_ladder.md` §8) — for both exp1 and
exp3 on the 16-patient test set, and shows (a) UMAP scatters and (b) hard numbers
for how benign/malignant-separable and subtype-structured each space is.

If SupCon works, exp3's embedding space should be measurably more class-separable
(higher silhouette, higher patient-blocked kNN accuracy) than exp1's — which is
precisely the property that makes a non-parametric retrieval module worth building.

What it produces (under --out, default analysis/embeddings/)
------------------------------------------------------------
* ``umap_binary.png``   — folds x {exp1 emb, exp3 emb, exp3 proj}, coloured
                          benign/malignant.
* ``umap_subtype.png``  — same panels, coloured by the 8 tumour subtypes (shows
                          the rare-subtype collapse that motivates retrieval).
* ``separation_metrics.json`` / a printed table — silhouette (binary & subtype)
                          and patient-blocked kNN accuracy per fold + mean.
* ``embeddings/<exp>_fold<k>.npz`` — the raw high-dim vectors + metadata, so the
                          retrieval module can reuse them without re-running.

Usage
-----
    python scripts/visualize_embeddings.py                       # exp1 vs exp3, all folds
    python scripts/visualize_embeddings.py --folds 0 4
    python scripts/visualize_embeddings.py --experiments exp1 exp2 exp3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rap_mst.data.datamodule import BreaKHisDataModule  # noqa: E402
from rap_mst.experiments import EXPERIMENT_DIRS  # noqa: E402
from rap_mst.models.builder import build_model  # noqa: E402
from rap_mst.utils.config import load_config  # noqa: E402
from rap_mst.utils.seed import seed_everything  # noqa: E402

_FOLD_RE = re.compile(r"_train_fold(\d+)$")
_SUBTYPE_RE = re.compile(r"^[A-Z]+_[A-Z]+_(?P<ttype>[A-Z]+)-")

# Derived from the preset registry so a new experiment (e.g. exp3n) needs no edit.
DEFAULT_EXPERIMENTS = dict(EXPERIMENT_DIRS)

# Subtype tokens grouped by class, in a fixed order so colours/legends are stable.
BENIGN_SUBTYPES = ["A", "F", "TA", "PT"]      # adenosis, fibroadenoma, tubular ad., phyllodes
MALIGNANT_SUBTYPES = ["DC", "LC", "MC", "PC"]  # ductal, lobular, mucinous, papillary carcinoma
SUBTYPE_ORDER = BENIGN_SUBTYPES + MALIGNANT_SUBTYPES
SUBTYPE_NAME = {
    "A": "adenosis", "F": "fibroadenoma", "TA": "tubular adenoma", "PT": "phyllodes",
    "DC": "ductal ca.", "LC": "lobular ca.", "MC": "mucinous ca.", "PC": "papillary ca.",
}
# Okabe-Ito colour-blind-safe palette (benign = cool, malignant = warm).
SUBTYPE_COLOR = {
    "A": "#56B4E9", "F": "#0072B2", "TA": "#009E73", "PT": "#00C7B1",
    "DC": "#D55E00", "LC": "#E69F00", "MC": "#CC79A7", "PC": "#F0428A",
}
BINARY_COLOR = {0: "#0072B2", 1: "#D55E00"}  # benign, malignant


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
def resolve_runs_root(experiment: str, log_root: Path) -> Path:
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


def has_projection_head(experiment: str, log_root: Path) -> bool:
    """Does this experiment's checkpoint carry a SupCon projection head?

    Read from the config saved *inside* the checkpoint rather than from the
    experiment name, so any future SupCon preset gets its projections column for
    free and a non-SupCon one never gets an empty column.
    """
    runs = find_runs(resolve_runs_root(experiment, log_root))
    if not runs:
        return False
    cfg_path = runs[sorted(runs)[0]] / "config.yaml"   # saved by ExperimentDir
    if not cfg_path.exists():
        return False
    import yaml
    with cfg_path.open("r", encoding="utf-8") as fh:
        saved = yaml.safe_load(fh) or {}
    return bool(saved.get("model", {}).get("projection", {}).get("enabled", False))


def find_runs(runs_root: Path) -> Dict[int, Path]:
    found: Dict[int, Path] = {}
    for d in sorted(runs_root.glob("*_train_fold*")):
        m = _FOLD_RE.search(d.name)
        if not m or not (d / "checkpoints" / "best.pt").exists():
            continue
        found[int(m.group(1))] = d
    return found


def subtype_of(patient_id: str) -> str:
    m = _SUBTYPE_RE.match(patient_id)
    return m.group("ttype") if m else "??"


# --------------------------------------------------------------------------- #
# Embedding extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_test_vectors(run_dir: Path, device: torch.device) -> Dict[str, np.ndarray]:
    """Run the fold's best.pt over the test set, returning the named vectors +meta.

    ``embeddings`` is always present (the canonical retrieval vector); ``projections``
    only when the model has a SupCon projection head (exp3).
    """
    cfg = load_config(str(run_dir / "config.yaml"))
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    dm = BreaKHisDataModule(cfg)
    dm.setup_test()
    loader = dm.test_dataloader()

    model = build_model(cfg).to(device)
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    amp = bool(cfg.train.mixed_precision) and device.type == "cuda"
    autocast = torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda")

    emb, proj, labels, pids, mags, probs = [], [], [], [], [], []
    has_proj = model.uses_projection
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        mag_index = batch["mag_index"].to(device, non_blocking=True)
        with autocast:
            out = model(images, mag_index)
        emb.append(out["embeddings"].float().cpu().numpy())
        if has_proj and out["projections"] is not None:
            proj.append(out["projections"].float().cpu().numpy())
        probs.append(F.softmax(out["logits"].float(), dim=1)[:, 1].cpu().numpy())
        labels.extend(batch["label"].tolist())
        pids.extend(batch["patient_id"])
        mags.extend(batch["magnification"])

    result = {
        "embeddings": np.concatenate(emb, axis=0),
        "label": np.asarray(labels, dtype=int),
        "patient_id": np.asarray(pids),
        "subtype": np.asarray([subtype_of(p) for p in pids]),
        "magnification": np.asarray(mags),
        "prob": np.concatenate(probs, axis=0),
    }
    if proj:
        result["projections"] = np.concatenate(proj, axis=0)
    return result


# --------------------------------------------------------------------------- #
# Quantitative separation metrics (on the high-dim vectors, not the 2-D UMAP)
# --------------------------------------------------------------------------- #
def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def separation_metrics(vecs: np.ndarray, labels: np.ndarray, subtypes: np.ndarray,
                        patient_ids: np.ndarray, k: int = 15) -> Dict[str, float]:
    """Cluster-separation metrics that a retrieval module would care about.

    * silhouette (cosine) by binary class and by subtype — global geometry.
    * patient-blocked kNN accuracy — for each image, majority label of its k nearest
      neighbours that belong to *other patients*. This is the retrieval premise made
      literal (neighbourhood vote), and blocking same-patient neighbours stops
      near-duplicate images from a slide trivially inflating it.
    """
    from sklearn.metrics import silhouette_score

    x = _l2(vecs.astype(np.float64))
    out: Dict[str, float] = {}
    try:
        out["silhouette_binary"] = float(silhouette_score(x, labels, metric="cosine"))
    except Exception:
        out["silhouette_binary"] = float("nan")
    # Subtype silhouette only meaningful with >=2 subtypes present.
    if len(np.unique(subtypes)) > 1:
        try:
            out["silhouette_subtype"] = float(silhouette_score(x, subtypes, metric="cosine"))
        except Exception:
            out["silhouette_subtype"] = float("nan")
    else:
        out["silhouette_subtype"] = float("nan")

    # Patient-blocked kNN accuracy via cosine similarity (x is L2-normalised).
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    same_patient = patient_ids[:, None] == patient_ids[None, :]
    sim[same_patient] = -np.inf  # block same-patient neighbours
    n = x.shape[0]
    correct = 0
    for i in range(n):
        order = np.argpartition(-sim[i], kth=min(k, n - 1))[:k]
        vote = labels[order]
        pred = 1 if vote.mean() >= 0.5 else 0
        correct += int(pred == labels[i])
    out["knn_blocked_acc"] = correct / n
    out["k"] = k
    return out


# --------------------------------------------------------------------------- #
# UMAP + plotting
# --------------------------------------------------------------------------- #
def umap_2d(vecs: np.ndarray, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    import umap  # local import; heavy dependency
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        metric="cosine", random_state=seed,
    )
    return reducer.fit_transform(_l2(vecs.astype(np.float64)))


def _scatter(ax, coords, labels, mode: str):
    if mode == "binary":
        for cls in (0, 1):
            m = labels == cls
            ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.55,
                       c=BINARY_COLOR[cls], linewidths=0, rasterized=True)
    else:  # subtype
        for st in SUBTYPE_ORDER:
            m = labels == st
            if not m.any():
                continue
            ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.6,
                       c=SUBTYPE_COLOR[st], linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.25)


def render_grid(panels: List[dict], folds: List[int], col_titles: List[str],
                mode: str, out_path: Path, metrics_by_cell: dict) -> None:
    """panels: flat list of dicts {fold, col, coords, label(binary), subtype}."""
    nrows, ncols = len(folds), len(col_titles)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    fold_row = {f: i for i, f in enumerate(folds)}
    for p in panels:
        r, c = fold_row[p["fold"]], p["col"]
        ax = axes[r][c]
        labels = p["label"] if mode == "binary" else p["subtype"]
        _scatter(ax, p["coords"], labels, mode)
        cell = metrics_by_cell.get((p["fold"], p["col"]))
        if cell is not None:
            tag = (f"sil={cell['silhouette_binary']:.2f}  kNN={cell['knn_blocked_acc']:.2f}"
                   if mode == "binary"
                   else f"sil_sub={cell['silhouette_subtype']:.2f}")
            ax.set_title(tag, fontsize=8, pad=2)
    for c, t in enumerate(col_titles):
        axes[0][c].annotate(t, xy=(0.5, 1.18), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=11, fontweight="bold")
    for r, f in enumerate(folds):
        axes[r][0].set_ylabel(f"fold {f}", fontsize=11, fontweight="bold")

    if mode == "binary":
        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                          markerfacecolor=BINARY_COLOR[c], markeredgewidth=0,
                          label=lbl) for c, lbl in [(0, "benign"), (1, "malignant")]]
    else:
        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                          markerfacecolor=SUBTYPE_COLOR[st], markeredgewidth=0,
                          label=f"{st} ({SUBTYPE_NAME[st]})") for st in SUBTYPE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    title = ("Test-set embedding UMAP — benign vs malignant"
             if mode == "binary"
             else "Test-set embedding UMAP — by tumour subtype (rare-subtype structure)")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.98))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Headline figure — the held-out test set embedded by ONE representative fold
# --------------------------------------------------------------------------- #
def render_headline(cache: Dict[tuple, Dict[str, np.ndarray]], experiments: List[str],
                    fold: int, out_dir: Path, seed: int, n_neighbors: int,
                    min_dist: float, knn: int) -> None:
    """One clean UMAP of the held-out 16-patient test set, exp1 vs exp3 (+proj).

    NOT a pool of folds — there is no single 'ensemble' coordinate frame, so we
    show the same test_patients set as embedded by a *single* representative
    fold's model, side by side per experiment. This is the direct-proof money
    figure; the 5-fold grid backs it up for robustness.
    """
    cols = [(f"{e} · embeddings", e, "embeddings") for e in experiments]
    # Add a projections column for every experiment that actually produced one
    # (exp3, exp3n, ...) -- decided by what is in the cache, not by name.
    for e in experiments:
        if (e, fold) in cache and "projections" in cache[(e, fold)]:
            cols.append((f"{e} · projections", e, "projections"))

    for mode in ("binary", "subtype"):
        fig, axes = plt.subplots(1, len(cols), figsize=(4.4 * len(cols), 4.6),
                                 squeeze=False)
        for c, (title, exp, key) in enumerate(cols):
            data = cache[(exp, fold)]
            coords = umap_2d(data[key], seed, n_neighbors, min_dist)
            ax = axes[0][c]
            _scatter(ax, coords, data["label"] if mode == "binary" else data["subtype"], mode)
            met = separation_metrics(data[key], data["label"], data["subtype"],
                                     data["patient_id"], k=knn)
            tag = (f"sil={met['silhouette_binary']:.2f}  kNN={met['knn_blocked_acc']:.2f}"
                   if mode == "binary" else f"sil_sub={met['silhouette_subtype']:.2f}")
            ax.annotate(title, xy=(0.5, 1.09), xycoords="axes fraction", ha="center",
                        va="bottom", fontsize=12, fontweight="bold")
            ax.set_title(tag, fontsize=9, pad=2)
        if mode == "binary":
            handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                              markerfacecolor=BINARY_COLOR[c], markeredgewidth=0,
                              label=lbl) for c, lbl in [(0, "benign"), (1, "malignant")]]
        else:
            handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                              markerfacecolor=SUBTYPE_COLOR[st], markeredgewidth=0,
                              label=f"{st} ({SUBTYPE_NAME[st]})") for st in SUBTYPE_ORDER]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
        cap = ("benign vs malignant" if mode == "binary"
               else "by tumour subtype (rare-subtype structure)")
        fig.suptitle(f"Held-out test set (16 patients, fold-{fold} model) — {cap}",
                     fontsize=13, fontweight="bold", y=1.03)
        fig.tight_layout(rect=(0.01, 0.03, 1, 0.96))
        out_path = out_dir / f"umap_testset_{mode}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_path}")


def load_cache_from_npz(out_dir: Path, experiments: List[str],
                        folds: Optional[List[int]]) -> tuple:
    """Rebuild the (exp, fold) -> vectors cache from saved .npz (no GPU needed)."""
    emb_dir = out_dir / "embeddings"
    cache: Dict[tuple, Dict[str, np.ndarray]] = {}
    found_folds = set()
    for npz in sorted(emb_dir.glob("*_fold*.npz")):
        m = re.search(r"(?P<exp>.+)_fold(?P<fold>\d+)\.npz$", npz.name)
        exp, fold = m.group("exp"), int(m.group("fold"))
        if exp not in experiments or (folds is not None and fold not in folds):
            continue
        d = np.load(npz, allow_pickle=True)
        entry = {k: d[k] for k in ("embeddings", "label", "patient_id", "subtype",
                                   "magnification", "prob")}
        if d["projections"].size:
            entry["projections"] = d["projections"]
        cache[(exp, fold)] = entry
        found_folds.add(fold)
    return cache, sorted(found_folds)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="UMAP + separation metrics on test embeddings.")
    ap.add_argument("--experiments", nargs="+", default=["exp1", "exp3"])
    ap.add_argument("--folds", nargs="+", type=int, default=None,
                    help="Subset of folds (default: all found).")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--out", default="analysis/embeddings")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-neighbors", type=int, default=25)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--knn", type=int, default=15, help="k for patient-blocked kNN accuracy.")
    ap.add_argument("--headline-fold", type=int, default=0,
                    help="Fold whose model draws the single held-out-test-set headline figure.")
    ap.add_argument("--headline-only", action="store_true",
                    help="Skip GPU extraction; rebuild only the headline test-set figure from saved .npz.")
    args = ap.parse_args()

    out_dir = Path(args.out)

    # Offline path: rebuild the headline test-set figure straight from saved .npz.
    if args.headline_only:
        cache, folds = load_cache_from_npz(out_dir, args.experiments, args.folds)
        if not cache:
            raise SystemExit(f"No .npz found under {out_dir/'embeddings'}; run a full pass first.")
        hf = args.headline_fold if args.headline_fold in folds else folds[0]
        print(f"headline-only: experiments={args.experiments} fold={hf}")
        render_headline(cache, args.experiments, hf, out_dir,
                        args.seed, args.n_neighbors, args.min_dist, args.knn)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    log_root = Path(cfg.logging.log_root)
    (out_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    print(f"device={device}  experiments={args.experiments}")

    # Build the column layout: <exp> emb for every experiment, then <exp> proj for
    # every experiment that HAS a projection head (exp3, exp3n, ...). Keyed on the
    # checkpoint's own config, not on the experiment name.
    columns: List[tuple] = []  # (col_title, exp, vector_key)
    for exp in args.experiments:
        columns.append((f"{exp} · embeddings", exp, "embeddings"))
    proj_experiments = [e for e in args.experiments
                        if has_projection_head(e, log_root)]
    for exp in proj_experiments:
        # SupCon-optimised space, shown alongside the indexed space.
        columns.append((f"{exp} · projections", exp, "projections"))
    col_titles = [c[0] for c in columns]

    # Discover folds (intersection across the requested experiments).
    runs_by_exp = {e: find_runs(resolve_runs_root(e, log_root)) for e in args.experiments}
    all_folds = sorted(set.intersection(*[set(r) for r in runs_by_exp.values()]))
    folds = [f for f in all_folds if args.folds is None or f in args.folds]
    if not folds:
        raise SystemExit("No common folds with best.pt across the requested experiments.")
    print(f"folds={folds}")

    panels: List[dict] = []
    metrics_by_cell: Dict[tuple, dict] = {}
    metrics_records: List[dict] = []

    # Cache extraction per (exp, fold) so exp3 emb+proj reuse one forward pass.
    cache: Dict[tuple, Dict[str, np.ndarray]] = {}
    for f in folds:
        for exp in args.experiments:
            print(f"extracting {exp} fold {f} ...", flush=True)
            data = extract_test_vectors(runs_by_exp[exp][f], device)
            cache[(exp, f)] = data
            np.savez_compressed(
                out_dir / "embeddings" / f"{exp}_fold{f}.npz",
                embeddings=data["embeddings"],
                projections=data.get("projections", np.empty((0,))),
                label=data["label"], patient_id=data["patient_id"],
                subtype=data["subtype"], magnification=data["magnification"],
                prob=data["prob"],
            )

    for ci, (title, exp, key) in enumerate(columns):
        for f in folds:
            data = cache[(exp, f)]
            if key not in data:
                continue
            vecs = data[key]
            met = separation_metrics(vecs, data["label"], data["subtype"],
                                     data["patient_id"], k=args.knn)
            metrics_by_cell[(f, ci)] = met
            metrics_records.append({"experiment": exp, "vector": key, "fold": f, **met})
            coords = umap_2d(vecs, args.seed, args.n_neighbors, args.min_dist)
            panels.append({"fold": f, "col": ci, "coords": coords,
                           "label": data["label"], "subtype": data["subtype"]})
            print(f"  {exp}/{key} fold {f}: "
                  f"sil_bin={met['silhouette_binary']:.3f} "
                  f"sil_sub={met['silhouette_subtype']:.3f} "
                  f"kNN={met['knn_blocked_acc']:.3f}")

    render_grid(panels, folds, col_titles, "binary", out_dir / "umap_binary.png", metrics_by_cell)
    render_grid(panels, folds, col_titles, "subtype", out_dir / "umap_subtype.png", metrics_by_cell)

    # ---- headline: held-out test set via one representative fold model ---- #
    hf = args.headline_fold if args.headline_fold in folds else folds[0]
    render_headline(cache, args.experiments, hf, out_dir,
                    args.seed, args.n_neighbors, args.min_dist, args.knn)

    # ---- aggregate metric table ---- #
    with (out_dir / "separation_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_records, fh, indent=2)

    print("\n" + "=" * 82)
    print("SEPARATION METRICS  (mean over folds; higher = better-structured space)")
    print(f"  {'column':22} {'sil_binary':>11} {'sil_subtype':>12} {'kNN_blocked':>12}")
    for ci, (title, exp, key) in enumerate(columns):
        rows = [m for m in metrics_records if m["experiment"] == exp and m["vector"] == key]
        if not rows:
            continue
        sb = float(np.nanmean([r["silhouette_binary"] for r in rows]))
        ss = float(np.nanmean([r["silhouette_subtype"] for r in rows]))
        kn = float(np.nanmean([r["knn_blocked_acc"] for r in rows]))
        print(f"  {title:22} {sb:>11.3f} {ss:>12.3f} {kn:>12.3f}")
    print(f"\nSaved figures + metrics + per-fold .npz under {out_dir}")


if __name__ == "__main__":
    main()
