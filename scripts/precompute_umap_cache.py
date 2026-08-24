"""Fit the four UMAPs of Figure 4 once and cache them.

Figure 4 is drawn twice -- once as a standalone raw panel and once inside the
assembled figure -- and the assembler must not re-fit UMAP: a second fit is a
second random initialisation in all but name, and the two renderings would then
disagree. Fitting once here and caching the 2-D coordinates makes the raw panel
and the assembled panel the *same picture*, and drops the cost of a redraw from
minutes to milliseconds.

The separation metrics are cached alongside because they are computed on the
1024-d vectors (an O(n^2) cosine matrix), not on the projection.

Note on imports
---------------
``scripts/visualize_embeddings.py`` owns the canonical definitions of the three
routines below, but it imports ``torch`` at module scope for the GPU path this
script does not use, and this environment has no torch. They are therefore
transcribed verbatim rather than imported; keep them in sync with that module.

    python scripts/precompute_umap_cache.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

LADDER = ["exp1", "exp2", "exp3", "exp3n"]
FOLD = 0
SEED, N_NEIGHBORS, MIN_DIST, KNN = 42, 25, 0.1, 15
OUT = ROOT / "analysis" / "embeddings" / "fig4_umap_cache.npz"


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def umap_2d(vecs: np.ndarray, seed: int, n_neighbors: int, min_dist: float):
    """Verbatim from ``visualize_embeddings.umap_2d``."""
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, metric="cosine", random_state=seed)
    return reducer.fit_transform(_l2(vecs.astype(np.float64)))


def separation_metrics(vecs, labels, subtypes, patient_ids, k: int = 15):
    """Verbatim from ``visualize_embeddings.separation_metrics``.

    Cosine silhouette by class and by subtype, plus patient-blocked kNN
    accuracy -- all on the 1024-d vectors, never on the 2-D projection.
    """
    from sklearn.metrics import silhouette_score

    x = _l2(vecs.astype(np.float64))
    out: dict[str, float] = {}
    out["silhouette_binary"] = float(silhouette_score(x, labels, metric="cosine"))
    out["silhouette_subtype"] = (
        float(silhouette_score(x, subtypes, metric="cosine"))
        if len(np.unique(subtypes)) > 1 else float("nan"))

    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    sim[patient_ids[:, None] == patient_ids[None, :]] = -np.inf
    n = x.shape[0]
    correct = 0
    for i in range(n):
        order = np.argpartition(-sim[i], kth=min(k, n - 1))[:k]
        correct += int((1 if labels[order].mean() >= 0.5 else 0) == labels[i])
    out["knn_blocked_acc"] = correct / n
    return out


def load_cache(emb_root: Path):
    """Verbatim behaviour of ``visualize_embeddings.load_cache_from_npz``."""
    cache = {}
    for npz in sorted((emb_root / "embeddings").glob("*_fold*.npz")):
        m = re.search(r"(?P<exp>.+)_fold(?P<fold>\d+)\.npz$", npz.name)
        exp, fold = m.group("exp"), int(m.group("fold"))
        if exp not in LADDER:
            continue
        d = np.load(npz, allow_pickle=True)
        cache[(exp, fold)] = {k: d[k] for k in
                              ("embeddings", "label", "patient_id", "subtype")}
    return cache


def main() -> None:
    cache = load_cache(ROOT / "analysis" / "embeddings")
    missing = [e for e in LADDER if (e, FOLD) not in cache]
    if missing:
        raise SystemExit(f"no cached embeddings for {missing} at fold {FOLD}")

    store: dict[str, np.ndarray] = {}
    for exp in LADDER:
        d = cache[(exp, FOLD)]
        print(f"  {exp}: UMAP over {d['embeddings'].shape} ...", flush=True)
        store[f"{exp}/coords"] = umap_2d(d["embeddings"], SEED, N_NEIGHBORS, MIN_DIST)
        met = separation_metrics(d["embeddings"], d["label"], d["subtype"],
                                 d["patient_id"], k=KNN)
        store[f"{exp}/label"] = d["label"]
        store[f"{exp}/subtype"] = d["subtype"]
        store[f"{exp}/sil"] = np.float64(met["silhouette_binary"])
        store[f"{exp}/sil_sub"] = np.float64(met["silhouette_subtype"])
        store[f"{exp}/knn"] = np.float64(met["knn_blocked_acc"])
        print(f"     sil {met['silhouette_binary']:.3f}  "
              f"sil_sub {met['silhouette_subtype']:.3f}  "
              f"kNN {met['knn_blocked_acc']:.3f}", flush=True)

    store["fold"] = np.int64(FOLD)
    store["knn_k"] = np.int64(KNN)
    store["n_images"] = np.int64(len(cache[(LADDER[0], FOLD)]["label"]))
    store["n_patients"] = np.int64(len(np.unique(cache[(LADDER[0], FOLD)]["patient_id"])))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **store)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
