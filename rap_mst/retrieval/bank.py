"""The unified memory bank: ONE table, one file, two views.

This is the store described in ``docs/retrieval.md`` Part V and ``docs/retrieval.md``
D5. Every row -- individual training image *and* derived per-patient centroid --
lives in the same table, distinguished only by a ``level`` column::

    ┌────────┬─────────────────┬───────┬─────────┬────────────┬──────┐
    │ level  │ key [D] (L2)    │ label │ subtype │ patient_id │ mag  │
    ├────────┼─────────────────┼───────┼─────────┼────────────┼──────┤
    │ image  │ f(x_1)          │  1    │  DC     │ SOB_M_DC.. │ 100  │
    │  ...   │                 │       │         │            │      │
    │ slide  │ centroid(p_1)   │  1    │  DC     │ SOB_M_DC.. │ ALL  │  <- derived
    └────────┴─────────────────┴───────┴─────────┴────────────┴──────┘

The single hard rule the measurements produced:

    **Unify the store. Never unify the ranking.**

A separate top-k *per level* reproduces the two-store design bit-identically
(max |Δp| = 0.00000000), whereas one top-k over the union silently degenerates
into image-only retrieval -- centroid rows win only ~1.2% of top-k slots -- and
loses exactly the PC-9146 / TA-16184 rescues the slide level exists to provide.
``MemoryBank`` therefore never merges levels on its own; ``RetrievalMemory``
queries it once per level, and the merged behaviour is reachable only through the
explicitly-labelled ``merge_levels`` **ablation**.

Design decisions implemented here
---------------------------------
D1  the key is the pre-magnification ``features`` vector, L2-normalised.
    ``retrieval.key`` makes that choice **configurable** (see ``retrieval/keys.py``)
    and ``retrieval.key_transform`` reshapes the fitted key space
    (see ``retrieval/transform.py``); both default to the D1 behaviour and both
    are recorded in the bank so an encoder/key/transform mismatch fails loudly.
D3  magnification is a metadata column used for *routing*, never a key dimension
D4  similarity-softmax vote at T=0.07 with a hard per-patient cap, no frequency
    re-weighting
D5  one table, a ``level`` column, one centroid per *patient* (not per
    patient×magnification), slide rows carry ``mag = ALL``
D7  plain cosine on L2-normalised keys; no learned projection, no CSLS
D8  subtypes are stored for diagnostics only -- they never touch the vote

Pure tensor ops; no learned parameters live here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from rap_mst.constants import IDX_TO_MAG
from rap_mst.retrieval.keys import extract_key
from rap_mst.retrieval.transform import KeyTransform

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
LEVEL_IMAGE = "image"
LEVEL_SLIDE = "slide"
LEVELS = (LEVEL_IMAGE, LEVEL_SLIDE)

#: Sentinel ``mag_index`` for slide rows. Centroids are per patient, not per
#: (patient, magnification) -- the whole point of the slide level is to average
#: *away* individual-field ambiguity -- so they are magnification-agnostic and
#: match any query magnification under every routing mode.
MAG_ALL = -1

ROUTES = ("same_mag", "all", "cross_mag")

#: v2 added ``key_transform`` (+ its two arrays). v1 banks load unchanged and are
#: treated as the identity transform, which is exactly what they were built with.
BANK_FORMAT_VERSION = 2
SUPPORTED_BANK_VERSIONS = (1, 2)

_NEG_INF = float("-inf")


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalisation (same epsilon as the probes, so keys match)."""
    return x / (x.norm(dim=1, keepdim=True) + 1e-12)


# --------------------------------------------------------------------------- #
# Rows and results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BankEntry:
    """One row of the unified value store (the key itself is passed separately)."""

    level: str          # "image" | "slide" -> which view owns it (D5)
    label: int          # 0 benign / 1 malignant -> the vote
    subtype: str        # A,F,TA,PT,DC,LC,MC,PC -> diagnostics only (D8)
    patient_id: str     # per-patient cap + leakage blocking
    mag_index: int      # 0..3 for image rows; MAG_ALL for slide rows (D3)


@dataclass
class RetrievalResult:
    """Everything one level's ranking produced, for the gate and for diagnostics."""

    prob: torch.Tensor                  # [B]    softmax-weighted P(malignant)
    sim: torch.Tensor                   # [B, k] cosine similarities, descending
    idx: torch.Tensor                   # [B, k] bank row indices (-1 where invalid)
    labels: torch.Tensor                # [B, k] neighbour labels (0 where invalid)
    weights: torch.Tensor               # [B, k] vote weights (0 where invalid)
    valid: torch.Tensor                 # [B, k] bool -- fewer than k may survive masks
    n_distinct_patients: torch.Tensor   # [B]    gate input + cap diagnostic
    top1_sim: torch.Tensor              # [B]    gate input
    subtypes: List[List[str]] = field(default_factory=list)     # diagnostics (D8)
    patient_ids: List[List[str]] = field(default_factory=list)  # diagnostics
    magnifications: List[List[int]] = field(default_factory=list)  # raw 40/100/... or -1

    @property
    def agreement(self) -> torch.Tensor:
        """Neighbourhood consensus in [0, 1] -- ``2*|p - 0.5|`` (gate input)."""
        return (self.prob - 0.5).abs() * 2.0

    def n_valid(self) -> torch.Tensor:
        return self.valid.sum(dim=1)


# --------------------------------------------------------------------------- #
# The bank
# --------------------------------------------------------------------------- #
class MemoryBank:
    """ONE sharded, L2-normalised key store holding every level (D5).

    Parameters
    ----------
    dim:
        Key dimensionality (1024 for the default ``features`` key).
    key:
        The key *spec* the keys were taken from (``rap_mst/retrieval/keys.py``:
        ``features``, ``projections``, ``fpn.std``, ``features+fpn.std``, ...).
        Persisted and asserted at load time so an ``embeddings``-keyed bank can
        never be queried with ``features`` vectors (the D1 ablation must be
        explicit).
    key_transform:
        The fitted key-space transform spec (``rap_mst/retrieval/transform.py``).
        ``"none"`` reproduces D1 exactly. Also persisted and asserted.
    meta:
        Free-form provenance (experiment, fold, checkpoint, split hash, ...).
    """

    def __init__(self, dim: Optional[int] = None, key: str = "features",
                 key_transform: str = "none", meta: Optional[dict] = None) -> None:
        # dim=None: infer from the first batch of keys, so a caller does not have
        # to know in advance whether `key` is features (1024) / embeddings (1088)
        # / projections (128) / a pyramid pooling.
        self.dim = int(dim) if dim is not None else 0
        self.key = str(key)
        self.key_transform = str(key_transform or "none")
        #: Unfitted until :meth:`fit_key_transform` runs, so the build path stores
        #: raw keys first and reshapes the whole table in one pass afterwards.
        self.transform = KeyTransform(spec="none")
        self.meta: Dict = dict(meta or {})

        self.keys = torch.zeros((0, self.dim), dtype=torch.float32)
        self.labels = torch.zeros(0, dtype=torch.long)
        self.mag_index = torch.zeros(0, dtype=torch.long)
        self.levels = np.empty(0, dtype="<U8")
        self.subtypes = np.empty(0, dtype="<U8")
        self.patient_ids = np.empty(0, dtype="<U64")

        self._pcode: Optional[torch.Tensor] = None      # patient -> contiguous int
        self._level_masks: Dict[str, torch.Tensor] = {}
        self._max_rows_per_patient: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def add(self, keys: torch.Tensor, entries: Sequence[BankEntry]) -> None:
        """Append rows. Keys are L2-normalised here so callers cannot forget."""
        keys = torch.as_tensor(keys).detach().to(torch.float32).cpu()
        if self.dim == 0 and keys.ndim == 2 and len(self) == 0:
            self.dim = int(keys.shape[1])
            self.keys = torch.zeros((0, self.dim), dtype=torch.float32)
        if keys.ndim != 2 or keys.shape[1] != self.dim:
            raise ValueError(f"Expected keys of shape [n, {self.dim}], got {tuple(keys.shape)}.")
        if len(entries) != keys.shape[0]:
            raise ValueError(f"{keys.shape[0]} keys but {len(entries)} entries.")
        for e in entries:
            if e.level not in LEVELS:
                raise ValueError(f"Unknown level {e.level!r}; expected one of {LEVELS}.")

        self.keys = torch.cat([self.keys, l2_normalize(keys)], dim=0)
        self.labels = torch.cat([self.labels, torch.tensor([e.label for e in entries], dtype=torch.long)])
        self.mag_index = torch.cat(
            [self.mag_index, torch.tensor([e.mag_index for e in entries], dtype=torch.long)]
        )
        self.levels = np.concatenate([self.levels, np.asarray([e.level for e in entries], dtype="<U8")])
        self.subtypes = np.concatenate([self.subtypes, np.asarray([e.subtype for e in entries], dtype="<U8")])
        self.patient_ids = np.concatenate(
            [self.patient_ids, np.asarray([e.patient_id for e in entries], dtype="<U64")]
        )
        self._invalidate_caches()

    def fit_key_transform(self, aux_dirs: Optional[torch.Tensor] = None) -> KeyTransform:
        """Fit ``self.key_transform`` on the bank's own image rows and rewrite the table.

        Call order matters and is enforced: image rows -> ``fit_key_transform()`` ->
        ``derive_slide_rows()``. Fitting *before* the centroids exist means the
        centroids are means in the transformed space (the space the vote actually
        ranks in), and fitting on the bank's own rows means the transform only ever
        sees training patients -- it cannot leak a validation or test image.
        """
        if (self.levels == LEVEL_SLIDE).any():
            raise RuntimeError("Fit the key transform before deriving slide rows.")
        if not self.transform.is_identity:
            raise RuntimeError("Key transform already fitted; call fit_key_transform() once.")
        img_mask = self.levels == LEVEL_IMAGE
        if not img_mask.any():
            raise RuntimeError("No image rows to fit the key transform on.")

        self.transform = KeyTransform.fit(
            self.keys[torch.as_tensor(img_mask, device=self.keys.device)],
            spec=self.key_transform,
            aux_dirs=aux_dirs,
        )
        if not self.transform.is_identity:
            self.keys = l2_normalize(self.transform.apply(self.keys))
            self.dim = int(self.keys.shape[1])
            self._invalidate_caches()
        return self.transform

    def derive_slide_rows(self) -> int:
        """Group the image rows by patient, average, re-normalise, append as slide rows.

        Centroids are **derived data, not a second artefact** (Part 2 §2.4): they
        cannot go stale and there is no second file to version. One centroid per
        *patient* -- the per-(patient, magnification) variant was measured and is
        no better on exp3n's predecessor and worse on exp1.
        """
        if (self.levels == LEVEL_SLIDE).any():
            raise RuntimeError("Slide rows already derived; call derive_slide_rows() once.")
        img_mask = self.levels == LEVEL_IMAGE
        if not img_mask.any():
            raise RuntimeError("No image rows to derive slide centroids from.")

        img_idx = np.nonzero(img_mask)[0]
        pids = self.patient_ids[img_idx]
        centroids: List[torch.Tensor] = []
        entries: List[BankEntry] = []
        for pid in sorted(set(pids.tolist())):
            rows_np = img_idx[pids == pid]
            rows = torch.as_tensor(rows_np, dtype=torch.long, device=self.keys.device)
            labels = self.labels[rows]
            subtypes = set(self.subtypes[rows_np].tolist())
            if labels.min().item() != labels.max().item() or len(subtypes) != 1:
                raise AssertionError(
                    f"Patient {pid} has mixed labels/subtypes in the bank -- dataset assumption violated."
                )
            centroids.append(self.keys[rows].mean(dim=0))
            entries.append(
                BankEntry(
                    level=LEVEL_SLIDE,
                    label=int(labels[0].item()),
                    subtype=subtypes.pop(),
                    patient_id=str(pid),
                    mag_index=MAG_ALL,
                )
            )
        self.add(torch.stack(centroids), entries)  # add() re-normalises the mean
        return len(entries)

    @torch.no_grad()
    def build_from_loader(
        self,
        model,
        loader,
        device,
        key: Optional[str] = None,
        amp: bool = False,
        logger=None,
        log_every: int = 25,
    ) -> int:
        """Encode a split with the **frozen** encoder and store ``extract_key(out, key)`` (D1).

        Only image rows are added; call :meth:`fit_key_transform` and then
        :meth:`derive_slide_rows` afterwards. The loader must use *eval*
        transforms, ``shuffle=False`` and ``drop_last=False`` -- see
        ``BreaKHisDataModule.bank_dataloader()``.
        """
        from rap_mst.data.breakhis import subtype_from_patient_id
        from rap_mst.retrieval.keys import assert_gap_matches_features

        key = key or self.key
        if key != self.key:
            raise ValueError(f"Bank was created for key={self.key!r} but asked to store {key!r}.")

        model.eval()
        autocast = torch.autocast(device_type="cuda", enabled=bool(amp) and device.type == "cuda")
        n = 0
        for i, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            mag_index = batch["mag_index"].to(device, non_blocking=True)
            with autocast:
                out = model(images, mag_index)
            if i == 0 and "fpn." in key:
                assert_gap_matches_features(out)
            vecs = extract_key(out, key).detach().float().cpu()   # AMP -> fp16; store fp32
            entries = [
                BankEntry(
                    level=LEVEL_IMAGE,
                    label=int(batch["label"][j].item()),
                    subtype=subtype_from_patient_id(batch["patient_id"][j]),
                    patient_id=batch["patient_id"][j],
                    mag_index=int(batch["mag_index"][j].item()),
                )
                for j in range(vecs.shape[0])
            ]
            self.add(vecs, entries)
            n += vecs.shape[0]
            if logger is not None and log_every and (i % log_every == 0):
                logger.info(f"  encoded {n} images ...")
        return n

    # ------------------------------------------------------------------ #
    # Introspection / safety
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self.keys.shape[0])

    @property
    def device(self) -> torch.device:
        return self.keys.device

    def to(self, device) -> "MemoryBank":
        device = torch.device(device)
        self.keys = self.keys.to(device)
        self.labels = self.labels.to(device)
        self.mag_index = self.mag_index.to(device)
        self._invalidate_caches()
        return self

    def unique_patients(self) -> List[str]:
        return sorted(set(self.patient_ids.tolist()))

    def assert_disjoint(self, patient_ids: Sequence[str], what: str = "query") -> None:
        """Hard leakage guard: fail loudly if any of these patients is in the bank.

        This is *the* highest-risk failure mode of the whole module -- a val or
        test patient in the bank makes every number downstream worthless. Never
        work around a firing assertion (``docs/retrieval.md`` Stage B1).
        """
        overlap = sorted(set(self.patient_ids.tolist()) & set(patient_ids))
        if overlap:
            raise AssertionError(
                f"LEAKAGE: {len(overlap)} {what} patient(s) present in the memory bank: "
                f"{overlap[:5]}{' ...' if len(overlap) > 5 else ''}"
            )

    def level_mask(self, level: Optional[str]) -> torch.Tensor:
        """Boolean row mask for a level; ``None`` selects every row (merge ablation)."""
        cache_key = level or "__all__"
        if cache_key not in self._level_masks:
            if level is None:
                mask = torch.ones(len(self), dtype=torch.bool)
            else:
                if level not in LEVELS:
                    raise ValueError(f"Unknown level {level!r}; expected one of {LEVELS}.")
                mask = torch.as_tensor(self.levels == level)
            self._level_masks[cache_key] = mask.to(self.keys.device)
        return self._level_masks[cache_key]

    def patient_codes(self) -> torch.Tensor:
        if self._pcode is None:
            _, inv = np.unique(self.patient_ids, return_inverse=True)
            self._pcode = torch.as_tensor(inv, dtype=torch.long, device=self.keys.device)
        return self._pcode

    def max_rows_per_patient(self, level: Optional[str]) -> int:
        cache_key = level or "__all__"
        if cache_key not in self._max_rows_per_patient:
            mask = self.level_mask(level).cpu().numpy()
            if not mask.any():
                self._max_rows_per_patient[cache_key] = 0
            else:
                _, counts = np.unique(self.patient_ids[mask], return_counts=True)
                self._max_rows_per_patient[cache_key] = int(counts.max())
        return self._max_rows_per_patient[cache_key]

    def summary(self) -> Dict:
        """Row counts per level and per magnification shard (Stage B1 sanity table)."""
        img = self.levels == LEVEL_IMAGE
        mags = self.mag_index.cpu().numpy()
        shards = {
            str(IDX_TO_MAG.get(int(mi), int(mi))): int(((mags == mi) & img).sum())
            for mi in sorted(set(mags[img].tolist()))
        }
        norms = self.keys.norm(dim=1) if len(self) else torch.zeros(1)
        return {
            "rows": len(self),
            "image_rows": int(img.sum()),
            "slide_rows": int((self.levels == LEVEL_SLIDE).sum()),
            "patients": len(self.unique_patients()),
            "dim": self.dim,
            "key": self.key,
            "key_transform": self.key_transform,
            "mag_shard_sizes": shards,
            "key_norm_min": float(norms.min().item()),
            "key_norm_max": float(norms.max().item()),
            "max_image_rows_per_patient": self.max_rows_per_patient(LEVEL_IMAGE),
        }

    def _invalidate_caches(self) -> None:
        self._pcode = None
        self._level_masks = {}
        self._max_rows_per_patient = {}

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #
    def view(self, level: Optional[str]) -> "BankView":
        """A level-restricted, independently-ranked slice.

        The two views are **never** merged into one top-k -- that collapses to
        image-only retrieval (D5 / Part 2 §2.3).
        """
        return BankView(self, level)

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def query(
        self,
        keys: torch.Tensor,
        mag_index: Optional[torch.Tensor] = None,
        level: Optional[str] = LEVEL_IMAGE,
        k: int = 15,
        route: str = "same_mag",
        per_patient_cap: int = 3,
        temperature: float = 0.07,
        block_patients: Optional[Sequence[str]] = None,
        query_patient_ids: Optional[Sequence[str]] = None,
        candidate_pool_multiplier: int = 8,
    ) -> RetrievalResult:
        """Rank one level and vote (D3 routing, D4 cap + softmax, D7 cosine).

        Parameters
        ----------
        keys:
            ``[B, dim]`` query vectors (normalised here; callers need not).
        mag_index:
            ``[B]`` contiguous magnification indices; required unless ``route='all'``.
        level:
            ``"image"`` / ``"slide"``, or ``None`` for the merge_levels **ablation**.
        block_patients / query_patient_ids:
            Rows from these patients are removed before ranking -- a global list
            and a per-query list respectively. Defence in depth on top of the
            build-time leakage assertion.
        """
        if route not in ROUTES:
            raise ValueError(f"Unknown route {route!r}; expected one of {ROUTES}.")
        if k <= 0:
            raise ValueError("k must be positive.")

        device = self.keys.device
        # Exactly the bank's own build path: L2 -> fitted transform -> L2. Doing it
        # here (rather than in the caller) is what makes it impossible to query a
        # transformed bank with untransformed vectors.
        q = l2_normalize(torch.as_tensor(keys).detach().to(device, torch.float32))
        if not self.transform.is_identity:
            q = l2_normalize(self.transform.apply(q))
        if q.shape[1] != self.dim:
            raise ValueError(
                f"Query dim {q.shape[1]} != bank dim {self.dim}. The bank was built on "
                f"key={self.key!r} key_transform={self.key_transform!r} -- rebuild it or "
                "fix retrieval.key / retrieval.key_transform."
            )
        batch = q.shape[0]

        sel = self.level_mask(level)
        sel_idx = torch.nonzero(sel, as_tuple=False).squeeze(1)          # [M] global rows
        if sel_idx.numel() == 0:
            raise RuntimeError(f"Memory bank holds no rows for level={level!r}.")

        sim = q @ self.keys[sel_idx].t()                                  # [B, M] cosine

        # --- D3: magnification routing (metadata, never a key dimension) --- #
        bank_mag = self.mag_index[sel_idx]                                # [M]
        if route != "all":
            if mag_index is None:
                raise ValueError(f"route={route!r} needs mag_index.")
            qm = torch.as_tensor(mag_index).to(device, torch.long).view(-1, 1)
            same = bank_mag.view(1, -1) == qm
            agnostic = bank_mag.view(1, -1) == MAG_ALL   # slide rows match any zoom
            drop = (~same if route == "same_mag" else same) & ~agnostic
            sim = sim.masked_fill(drop, _NEG_INF)

        # --- leakage / patient blocking ------------------------------------ #
        # NB: never force the query ids into the bank's dtype -- a narrower/wider
        # fixed-width unicode dtype would silently truncate an id and could then
        # match the wrong patient. numpy picks a common width on its own.
        bank_pid = self.patient_ids[sel_idx.cpu().numpy()]                # [M] str
        if block_patients is not None and len(block_patients):   # len(): arrays too
            blocked = np.isin(bank_pid, np.asarray(list(block_patients)))
            if blocked.any():
                sim = sim.masked_fill(torch.as_tensor(blocked, device=device).view(1, -1), _NEG_INF)
        if query_patient_ids is not None:
            qp = np.asarray(list(query_patient_ids)).reshape(-1, 1)
            same_patient = torch.as_tensor(bank_pid.reshape(1, -1) == qp, device=device)
            sim = sim.masked_fill(same_patient, _NEG_INF)

        # --- D4: top-k with a hard per-patient cap -------------------------- #
        pcode_sel = self.patient_codes()[sel_idx]                         # [M]
        cap_active = 0 < per_patient_cap < min(k, self.max_rows_per_patient(level))
        pool = min(sim.shape[1], k * max(candidate_pool_multiplier, 1)) if cap_active else min(sim.shape[1], k)
        top_sim, top_col = torch.topk(sim, pool, dim=1)                   # descending

        if cap_active:
            chosen = self._apply_patient_cap(top_sim, top_col, pcode_sel, k, per_patient_cap)
        else:
            chosen = torch.full((batch, k), -1, dtype=torch.long, device=device)
            width = min(k, top_col.shape[1])
            chosen[:, :width] = torch.where(
                torch.isfinite(top_sim[:, :width]), top_col[:, :width],
                torch.full_like(top_col[:, :width], -1),
            )

        valid = chosen >= 0
        safe = chosen.clamp(min=0)
        n_sim = torch.gather(sim, 1, safe)
        n_sim = torch.where(valid, n_sim, torch.zeros_like(n_sim))
        n_lab = self.labels[sel_idx][safe].to(sim.dtype)
        n_lab = torch.where(valid, n_lab, torch.zeros_like(n_lab))

        # --- D4: similarity-softmax vote at T (do not tune) ------------------ #
        masked = torch.where(valid, n_sim, torch.full_like(n_sim, _NEG_INF))
        shift = masked.max(dim=1, keepdim=True).values
        shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
        weights = torch.exp((n_sim - shift) / float(temperature)) * valid.to(n_sim.dtype)
        wsum = weights.sum(dim=1)
        prob = torch.where(wsum > 0, (weights * n_lab).sum(dim=1) / (wsum + 1e-12),
                           torch.full_like(wsum, 0.5))

        global_idx = torch.where(valid, sel_idx[safe], torch.full_like(safe, -1))
        n_distinct = self._count_distinct(pcode_sel[safe], valid)
        top1 = torch.where(valid[:, 0], n_sim[:, 0], torch.zeros_like(prob))

        gi = global_idx.cpu().numpy()
        vi = valid.cpu().numpy()
        all_mags = self.mag_index.cpu().numpy()
        return RetrievalResult(
            prob=prob,
            sim=n_sim,
            idx=global_idx,
            labels=n_lab,
            weights=weights,
            valid=valid,
            n_distinct_patients=n_distinct,
            top1_sim=top1,
            subtypes=[self.subtypes[gi[b][vi[b]]].tolist() for b in range(batch)],
            patient_ids=[self.patient_ids[gi[b][vi[b]]].tolist() for b in range(batch)],
            magnifications=[
                [IDX_TO_MAG.get(int(m), -1) for m in all_mags[gi[b][vi[b]]]] for b in range(batch)
            ],
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_patient_cap(
        top_sim: torch.Tensor, top_col: torch.Tensor, pcode_sel: torch.Tensor,
        k: int, cap: int,
    ) -> torch.Tensor:
        """Greedy descending scan keeping at most ``cap`` rows per bank patient.

        BreaKHis slides contribute 60-235 near-identical fields each, so an
        uncapped top-15 is routinely 15 views of *one* patient. The cap turns that
        into "evidence from >=5 patients" and is worth +0.6 image points and
        +2.6 specificity (D4).
        """
        sims = top_sim.cpu().numpy()
        cols = top_col.cpu().numpy()
        codes = pcode_sel.cpu().numpy()
        out = np.full((sims.shape[0], k), -1, dtype=np.int64)
        for b in range(sims.shape[0]):
            seen: Dict[int, int] = {}
            n = 0
            for pos in range(cols.shape[1]):
                if not np.isfinite(sims[b, pos]):
                    break                      # descending: everything after is masked
                c = int(codes[cols[b, pos]])
                if seen.get(c, 0) >= cap:
                    continue
                seen[c] = seen.get(c, 0) + 1
                out[b, n] = cols[b, pos]
                n += 1
                if n == k:
                    break
        return torch.as_tensor(out, device=top_col.device)

    @staticmethod
    def _count_distinct(codes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        counts = []
        c = codes.cpu().numpy()
        v = valid.cpu().numpy()
        for b in range(c.shape[0]):
            counts.append(len(np.unique(c[b][v[b]])) if v[b].any() else 0)
        return torch.as_tensor(counts, dtype=torch.long, device=codes.device)

    # ------------------------------------------------------------------ #
    # Persistence -- ONE .npz holds every level (D5)
    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta)
        meta.update({
            "format_version": BANK_FORMAT_VERSION,
            "dim": self.dim,
            "key": self.key,
            "key_transform": self.key_transform,
            "key_transform_info": self.transform.info,
        })
        np.savez_compressed(
            path,
            keys=self.keys.cpu().numpy().astype(np.float32),
            labels=self.labels.cpu().numpy().astype(np.int64),
            mag_index=self.mag_index.cpu().numpy().astype(np.int64),
            levels=self.levels,
            subtypes=self.subtypes,
            patient_ids=self.patient_ids,
            meta=np.asarray(json.dumps(meta)),
            **self.transform.state(),
        )

    @classmethod
    def load(cls, path) -> "MemoryBank":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Memory bank not found: {path}. Run scripts/build_memory_bank.py first."
            )
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        version = int(meta.get("format_version", -1))
        if version not in SUPPORTED_BANK_VERSIONS:
            raise ValueError(
                f"Bank format v{meta.get('format_version')} not in {SUPPORTED_BANK_VERSIONS}: {path}"
            )
        # v1 predates configurable key transforms and was always built untransformed.
        transform_spec = str(meta.get("key_transform", "none")) if version >= 2 else "none"
        bank = cls(dim=int(meta["dim"]), key=str(meta["key"]),
                   key_transform=transform_spec, meta=meta)
        bank.transform = KeyTransform.from_state(
            transform_spec, d, info=meta.get("key_transform_info")
        )
        bank.keys = torch.as_tensor(d["keys"], dtype=torch.float32)
        bank.labels = torch.as_tensor(d["labels"], dtype=torch.long)
        bank.mag_index = torch.as_tensor(d["mag_index"], dtype=torch.long)
        bank.levels = d["levels"]
        bank.subtypes = d["subtypes"]
        bank.patient_ids = d["patient_ids"]
        bank._invalidate_caches()
        return bank

    def assert_compatible(self, *, key: Optional[str] = None, key_transform: Optional[str] = None,
                          experiment: Optional[str] = None, fold: Optional[int] = None) -> None:
        """Refuse silently-wrong combinations (bank/encoder/key/transform mismatch)."""
        if key is not None and key != self.key:
            raise ValueError(
                f"Bank was built on key={self.key!r} but retrieval.key={key!r}. "
                "Rebuild the bank or change the config -- do not mix."
            )
        if key_transform is not None and str(key_transform or "none") != self.key_transform:
            raise ValueError(
                f"Bank was built with key_transform={self.key_transform!r} but "
                f"retrieval.key_transform={key_transform!r}. The transform is fitted into the "
                "stored keys -- rebuild the bank or change the config, do not mix."
            )
        if experiment is not None and self.meta.get("experiment") not in (None, experiment):
            raise ValueError(
                f"Bank belongs to experiment {self.meta.get('experiment')!r}, not {experiment!r}. "
                "The bank, the gate and the test run must all come from the same encoder."
            )
        if fold is not None and self.meta.get("fold") not in (None, fold):
            raise ValueError(f"Bank is fold {self.meta.get('fold')}, not fold {fold}.")


@dataclass
class BankView:
    """A level-restricted handle on the one bank. Ranks independently, always."""

    bank: MemoryBank
    level: Optional[str]

    def query(self, keys: torch.Tensor, **kwargs) -> RetrievalResult:
        kwargs.pop("level", None)
        return self.bank.query(keys, level=self.level, **kwargs)

    def __len__(self) -> int:
        return int(self.bank.level_mask(self.level).sum().item())
