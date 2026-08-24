"""KeyTransform -- a fitted, invertible-in-spirit reshaping of the key space.

Motivation (``docs/retrieval.md`` Part VII §3). The exp3n ``features`` space is a
narrow cone: effective rank 1.19 / 1024, one singular direction holding 84-96% of
the variance, ‖mean unit key‖ = 0.84, random-pair cosine 0.70. Two consequences
follow, and both are geometric rather than architectural:

1. every cosine is dominated by the single dominant axis, which *is* the
   malignancy axis the linear head already reads optimally -- so the kNN vote is a
   noisy copy of ``p_param`` rather than independent evidence;
2. the nearest neighbour sits at cosine ~0.999 for *every* query, so "similarity"
   carries almost no ranking information.

Both are fixable **without retraining anything**, by whitening the stored space or
by deleting the directions the classifier already consumes. That is what this
module does. A transform is fitted **once, on the bank's own image rows** (i.e. on
training patients only -- it never sees a validation or test image), stored inside
the bank ``.npz``, and applied identically to every query. Fitting on the bank is
what keeps it leak-free: it is a property of the memory, not of the query set.

Spec grammar (``retrieval.key_transform``), a comma-separated pipeline::

    none                 identity (D1 behaviour, the default)
    center               subtract the bank mean key
    pca_drop:<n>         center, then remove the top-<n> principal directions
    whiten:<n>           center, project onto the top-<n> PCs, scale by 1/sqrt(lambda)
    drop_dirs            remove externally-supplied directions (the classifier's
                         decision direction -- see ``fit(..., aux_dirs=...)``)

Examples::

    retrieval.key_transform: "pca_drop:1"          # delete the collapsed cone axis
    retrieval.key_transform: "center,drop_dirs"    # delete what the head reads
    retrieval.key_transform: "whiten:128"          # isotropise the top-128 subspace

Every step is applied in the order written, and the result is L2-normalised by
the bank (``MemoryBank.add``), so downstream cosine/vote/cap logic is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

STEPS = ("center", "pca_drop", "whiten", "drop_dirs")

_EPS = 1e-12


def pca(centred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(singular_values, components)`` of an already-centred matrix.

    Via the ``d x d`` covariance eigendecomposition rather than a full SVD of the
    ``n x d`` matrix: with n ~ 5,000 and d ~ 1,024 that is ~15x faster, and the
    ablation sweep fits ~70 of these.
    """
    cov = centred.T @ centred
    lam, vecs = np.linalg.eigh(cov)          # ascending
    order = np.argsort(lam)[::-1]
    lam = np.clip(lam[order], 0.0, None)
    return np.sqrt(lam), vecs[:, order].T    # (sv [d], components [d, d] row-wise)


def parse_transform_spec(spec: Optional[str]) -> List[Tuple[str, Optional[int]]]:
    """``"pca_drop:1,center"`` -> ``[("pca_drop", 1), ("center", None)]``."""
    text = (spec or "none").strip()
    if text.lower() in ("", "none", "identity", "null"):
        return []
    steps: List[Tuple[str, Optional[int]]] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        name, _, arg = token.partition(":")
        name = name.strip()
        if name not in STEPS:
            raise ValueError(f"Unknown key_transform step {name!r}; expected one of {STEPS}.")
        if name in ("pca_drop", "whiten"):
            if not arg:
                raise ValueError(f"Step {name!r} needs an integer argument, e.g. '{name}:1'.")
            steps.append((name, int(arg)))
        else:
            if arg:
                raise ValueError(f"Step {name!r} takes no argument (got {arg!r}).")
            steps.append((name, None))
    return steps


@dataclass
class KeyTransform:
    """A fitted linear key-space transform: ``x -> (x - mu) @ W``.

    Every supported step composes into exactly that form, so the fitted object is
    two small arrays and application is one matmul -- cheap enough to run on every
    query without changing the module's non-parametric character (nothing here is
    learned from labels; it is fitted from the bank's geometry alone).
    """

    spec: str = "none"
    mean: Optional[np.ndarray] = None        # [D]
    matrix: Optional[np.ndarray] = None      # [D, D_out]
    info: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def is_identity(self) -> bool:
        return self.matrix is None and self.mean is None

    @property
    def out_dim(self) -> Optional[int]:
        return None if self.matrix is None else int(self.matrix.shape[1])

    # ------------------------------------------------------------------ #
    @classmethod
    def fit(
        cls,
        keys: torch.Tensor,
        spec: Optional[str] = "none",
        aux_dirs: Optional[torch.Tensor] = None,
    ) -> "KeyTransform":
        """Fit on the bank's own (already L2-normalised) image-level keys.

        Parameters
        ----------
        keys:
            ``[N, D]`` bank keys -- **training patients only**.
        aux_dirs:
            ``[M, D]`` directions consumed by the ``drop_dirs`` step. For the
            classifier-direction ablation this is ``W[1] - W[0]`` of the linear
            head: the one direction a binary linear readout actually uses.
        """
        steps = parse_transform_spec(spec)
        x = torch.as_tensor(keys).detach().to(torch.float64).cpu().numpy()
        dim = x.shape[1]
        if not steps:
            return cls(spec="none")

        mu = np.zeros(dim, dtype=np.float64)
        mat = np.eye(dim, dtype=np.float64)
        info: Dict = {"steps": []}

        for name, arg in steps:
            cur = (x - mu) @ mat
            if name == "center":
                shift = cur.mean(axis=0)
                # Fold the shift back into (mu, mat): (x - mu)W - s == (x - mu')W
                # only when W is invertible, so instead track it exactly by
                # updating mu through the pseudo-inverse of the current matrix.
                mu = mu + shift @ np.linalg.pinv(mat)
                info["steps"].append({"step": "center", "shift_norm": float(np.linalg.norm(shift))})

            elif name == "pca_drop":
                cen = cur - cur.mean(axis=0)
                sv, vt = pca(cen)
                n = int(arg)
                if n >= vt.shape[0]:
                    raise ValueError(f"pca_drop:{n} would delete every direction ({vt.shape[0]} available).")
                drop = vt[:n]                                  # [n, d_cur]
                proj = np.eye(mat.shape[1]) - drop.T @ drop    # remove those directions
                mu = mu + cur.mean(axis=0) @ np.linalg.pinv(mat)
                mat = mat @ proj
                var = sv ** 2
                info["steps"].append({
                    "step": "pca_drop", "n": n,
                    "variance_removed": float(var[:n].sum() / max(var.sum(), _EPS)),
                })

            elif name == "whiten":
                cen = cur - cur.mean(axis=0)
                sv, vt = pca(cen)
                n = min(int(arg), vt.shape[0])
                lam = np.maximum(sv[:n] ** 2 / max(len(cen) - 1, 1), 1e-8)
                w = vt[:n].T / np.sqrt(lam)                    # [d_cur, n]
                mu = mu + cur.mean(axis=0) @ np.linalg.pinv(mat)
                mat = mat @ w
                var = sv ** 2
                info["steps"].append({
                    "step": "whiten", "n": n,
                    "variance_kept": float(var[:n].sum() / max(var.sum(), _EPS)),
                })

            elif name == "drop_dirs":
                if aux_dirs is None:
                    raise ValueError(
                        "key_transform step 'drop_dirs' needs aux_dirs (e.g. the classifier's "
                        "decision direction). Pass aux_dirs= to KeyTransform.fit()."
                    )
                d = torch.as_tensor(aux_dirs).detach().to(torch.float64).cpu().numpy()
                if d.ndim == 1:
                    d = d.reshape(1, -1)
                if d.shape[1] != dim:
                    raise ValueError(f"aux_dirs dim {d.shape[1]} != key dim {dim}.")
                # Directions live in the ORIGINAL key space; map them through the
                # current matrix so a preceding whiten/pca_drop stays consistent.
                cur_dirs = d @ mat
                q, _ = np.linalg.qr(cur_dirs.T)                # orthonormal basis [d_cur, m]
                proj = np.eye(mat.shape[1]) - q @ q.T
                mat = mat @ proj
                info["steps"].append({"step": "drop_dirs", "n_dirs": int(d.shape[0])})

        info["in_dim"] = int(dim)
        info["out_dim"] = int(mat.shape[1])
        info["n_fit_rows"] = int(x.shape[0])
        return cls(spec=str(spec), mean=mu.astype(np.float32), matrix=mat.astype(np.float32), info=info)

    # ------------------------------------------------------------------ #
    def apply(self, keys: torch.Tensor) -> torch.Tensor:
        """``[N, D] -> [N, D_out]``. Identity transforms are a no-op passthrough."""
        if self.is_identity:
            return keys
        x = torch.as_tensor(keys)
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        mat = torch.as_tensor(self.matrix, dtype=x.dtype, device=x.device)
        if x.shape[1] != mean.shape[0]:
            raise ValueError(
                f"Key dim {x.shape[1]} != transform input dim {mean.shape[0]} "
                f"(transform '{self.spec}'). Rebuild the bank."
            )
        return (x - mean) @ mat

    # ------------------------------------------------------------------ #
    def state(self) -> Dict[str, np.ndarray]:
        """Arrays for the bank ``.npz`` (empty dict for the identity transform)."""
        if self.is_identity:
            return {}
        return {"transform_mean": self.mean, "transform_matrix": self.matrix}

    @classmethod
    def from_state(cls, spec: str, data, info: Optional[Dict] = None) -> "KeyTransform":
        if "transform_matrix" not in data:
            if parse_transform_spec(spec):
                raise ValueError(
                    f"Bank declares key_transform={spec!r} but stores no transform arrays. "
                    "Rebuild the bank."
                )
            return cls(spec="none")
        return cls(
            spec=spec,
            mean=np.asarray(data["transform_mean"], dtype=np.float32),
            matrix=np.asarray(data["transform_matrix"], dtype=np.float32),
            info=dict(info or {}),
        )


# --------------------------------------------------------------------------- #
# The one aux direction that matters
# --------------------------------------------------------------------------- #
def classifier_direction(model, key_spec: str = "features") -> Optional[torch.Tensor]:
    """``W[1] - W[0]`` of a **linear** classification head: what ``p_param`` reads.

    For a two-class linear readout ``logit_1 - logit_0 = x·(w1 - w0) + b``, so this
    single direction is the entire parametric decision function. Projecting it out
    of the key yields, exactly, "the part of the representation the classifier does
    not use" -- the sharpest possible test of whether retrieval has anything
    orthogonal to contribute. Returns ``None`` when the head is not a plain linear
    layer over the key (a hidden layer, or a key that is not the head's input).
    """
    if key_spec not in ("features", "embeddings"):
        return None
    head = getattr(model, "classifier", None)
    net = getattr(head, "net", head)
    linear_layers = [m for m in getattr(net, "modules", lambda: [])() if isinstance(m, torch.nn.Linear)]
    if len(linear_layers) != 1:
        return None
    w = linear_layers[0].weight.detach()
    if w.shape[0] != 2:
        return None
    return (w[1] - w[0]).reshape(1, -1).float().cpu()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + _EPS))
