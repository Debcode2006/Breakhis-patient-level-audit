"""RetrievalMemory -- orchestrates the two views over the ONE bank.

Non-parametric by construction: it holds no learned weights, never mutates the
model's forward dict, and only *returns* evidence. Everything learned lives in
:class:`~rap_mst.retrieval.gate.FusionGate` (~100 parameters).

Why two queries and not one
---------------------------
``docs/retrieval.md`` D5 / ``docs/retrieval.md`` Part V: image-level kNN and
slide-level prototypes fix **disjoint** failure cases -- image kNN rescues the
low-grade ductal patient (DC-12312: 0.577 -> 0.73) while slide prototypes rescue
the two architecture-mimicry cases (PC-9146 0.639 -> 0.68, TA-16184 0.316 ->
0.43) and give the best specificity of any pure retrieval rule (0.851 vs 0.805).
Ranking them together destroys that: centroid rows take only 1.2% of top-k slots,
so a merged index *is* the image view with the slide evidence statistically
deleted. Hence:

    Unify the store. Never unify the ranking.

``merge_levels=True`` reproduces that failure on purpose -- it exists only as the
pre-registered ablation and logs a warning when used.

Diagnostics returned alongside the probabilities (``agreement``, ``top1_sim``,
``n_distinct_patients``) are exactly the gate's inputs, so the gate never needs
to re-query the bank.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from rap_mst.retrieval.bank import (
    LEVEL_IMAGE,
    LEVEL_SLIDE,
    MemoryBank,
    RetrievalResult,
)

# A library logger: no handlers of its own, propagates to the `rap_mst` package
# logger that get_logger() mirrors the run's console + file sinks onto.
logger = logging.getLogger(__name__)


@dataclass
class LevelConfig:
    """Per-level ranking knobs. Separate ``k``/``temperature`` are mandatory, not
    optional: centroid similarities are far more dispersed than image ones, so a
    shared temperature would be wrong even though both optima land at 0.07."""

    enabled: bool = True
    route: str = "same_mag"
    k: int = 15
    per_patient_cap: int = 3
    temperature: float = 0.07
    candidate_pool_multiplier: int = 8

    @classmethod
    def from_cfg(cls, node, **defaults) -> "LevelConfig":
        out = cls(**defaults)
        if node is None:
            return out
        for f in ("enabled", "route", "k", "per_patient_cap", "temperature",
                  "candidate_pool_multiplier"):
            val = node.get(f, None) if hasattr(node, "get") else getattr(node, f, None)
            if val is not None:
                setattr(out, f, val)
        return out


class RetrievalMemory(nn.Module):
    """Query the one bank once per level and turn neighbourhoods into evidence.

    Returns (all tensors on the bank's device)::

        p_img                softmax(sim/T)-weighted mean of neighbour labels   (D4)
        p_slide              same over the top-`slide_k` patient centroids      (D5)
        agreement            2*|p_img - 0.5| in [0,1] -- neighbourhood consensus
        top1_sim             best image-level cosine similarity
        n_distinct_patients  how many bank patients contributed (cap diagnostic)
        image_result/slide_result  full RetrievalResult for interpretability
    """

    def __init__(
        self,
        bank: MemoryBank,
        key: str = "features",
        key_transform: str = "none",
        image: Optional[LevelConfig] = None,
        slide: Optional[LevelConfig] = None,
        merge_levels: bool = False,
        block_query_patients: bool = True,
    ) -> None:
        super().__init__()
        self.bank = bank
        self.key = key
        self.key_transform = str(key_transform or "none")
        self.image = image or LevelConfig()
        self.slide = slide or LevelConfig(route="all", k=5, per_patient_cap=1)
        self.merge_levels = bool(merge_levels)
        self.block_query_patients = bool(block_query_patients)
        self._views_verified = False

        bank.assert_compatible(key=key, key_transform=self.key_transform)
        if not self.image.enabled and not self.slide.enabled and not self.merge_levels:
            raise ValueError("retrieval: both levels disabled -- nothing to retrieve.")
        if self.merge_levels:
            logger.warning(
                "retrieval.merge_levels=true: ONE top-k over the union of levels. "
                "This is the pre-registered ABLATION -- it degenerates to image-only "
                "retrieval (centroid share ~1%) and loses the PC/TA rescues (D5)."
            )

    # ------------------------------------------------------------------ #
    @property
    def uses_slide(self) -> bool:
        return self.slide.enabled and not self.merge_levels

    @property
    def uses_image(self) -> bool:
        return self.image.enabled or self.merge_levels

    def term_mask(self) -> Sequence[int]:
        """(param, image, slide) availability -- consumed by the fusion gate."""
        return (1, int(self.uses_image), int(self.uses_slide))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(
        self,
        features: torch.Tensor,
        mag_index: Optional[torch.Tensor] = None,
        query_patient_ids: Optional[Sequence[str]] = None,
        block_patients: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        device = self.bank.device
        batch = features.shape[0]
        blocked = query_patient_ids if self.block_query_patients else None

        def _run(level: Optional[str], cfg: LevelConfig) -> RetrievalResult:
            return self.bank.query(
                features,
                mag_index=mag_index,
                level=level,
                k=cfg.k,
                route=cfg.route,
                per_patient_cap=cfg.per_patient_cap,
                temperature=cfg.temperature,
                block_patients=block_patients,
                query_patient_ids=blocked,
                candidate_pool_multiplier=cfg.candidate_pool_multiplier,
            )

        neutral = torch.full((batch,), 0.5, dtype=torch.float32, device=device)
        image_result: Optional[RetrievalResult] = None
        slide_result: Optional[RetrievalResult] = None

        if self.merge_levels:
            # Ablation only: one ranking over every row of the table (level=None).
            image_result = _run(None, self.image)
        else:
            if self.image.enabled:
                image_result = _run(LEVEL_IMAGE, self.image)
            if self.slide.enabled:
                slide_result = _run(LEVEL_SLIDE, self.slide)

        self._verify_views_once(image_result, slide_result)

        p_img = image_result.prob if image_result is not None else neutral
        p_slide = slide_result.prob if slide_result is not None else neutral
        agreement = (p_img - 0.5).abs() * 2.0
        top1 = image_result.top1_sim if image_result is not None else torch.zeros_like(neutral)
        n_distinct = (
            image_result.n_distinct_patients if image_result is not None
            else torch.zeros(batch, dtype=torch.long, device=device)
        )

        return {
            "p_img": p_img,
            "p_slide": p_slide,
            "agreement": agreement,
            "top1_sim": top1,
            "n_distinct_patients": n_distinct,
            "image_result": image_result,
            "slide_result": slide_result,
        }

    # ------------------------------------------------------------------ #
    def _verify_views_once(self, image_result, slide_result) -> None:
        """Assert, on the first batch, that the ONE bank really served TWO rankings.

        The D5 invariant is easy to break in a future refactor and impossible to
        spot in the aggregate metrics -- a merged ranking silently *becomes* the
        image view (centroid share ~1.2%) and quietly loses the PC-9146 / TA-16184
        rescues. So the run checks it structurally instead of trusting the code:
        every row the image view returned must be a ``level='image'`` row, every row
        the slide view returned must be a ``level='slide'`` row, and the slide view
        must return one row per *distinct patient*.
        """
        if self._views_verified:
            return
        self._views_verified = True

        levels = self.bank.levels
        for name, result, expected in (("image", image_result, LEVEL_IMAGE),
                                       ("slide", slide_result, LEVEL_SLIDE)):
            if result is None or self.merge_levels:
                continue
            idx = result.idx[result.valid].cpu().numpy()
            if idx.size and not (levels[idx] == expected).all():
                got = sorted(set(levels[idx].tolist()))
                raise AssertionError(
                    f"D5 violation: the '{name}' view returned rows of level(s) {got}, "
                    f"expected only '{expected}'. Unify the store, never the ranking."
                )

        if slide_result is not None and not self.merge_levels:
            for pids in slide_result.patient_ids:
                if len(pids) != len(set(pids)):
                    raise AssertionError(
                        "D5 violation: the slide view returned two rows for one patient; "
                        "centroids must be one per patient (not per patient x magnification)."
                    )

        if self.merge_levels:
            logger.warning(
                "retrieval: ONE merged ranking over both levels (ablation) -- the slide "
                "evidence is statistically deleted; p_slide is fixed at 0.5."
            )
        else:
            logger.info(
                "retrieval: ONE bank -> TWO independent rankings (D5 verified on the first "
                f"batch) | image view: level=image route={self.image.route} k={self.image.k} "
                f"cap={self.image.per_patient_cap} T={self.image.temperature} | "
                f"slide view: level=slide route={self.slide.route} k={self.slide.k} "
                f"cap={self.slide.per_patient_cap} T={self.slide.temperature}"
                + ("" if self.slide.enabled else "  [slide DISABLED]")
            )


@dataclass
class RetrievalRunner:
    """Memory + gate assembled for inference (what the evaluator actually calls).

    Keeping the composition here means ``scripts/test.py`` and
    ``scripts/train_gate.py`` share one code path, and the gate can be absent
    (fixed-alpha ablation) without either script branching.
    """

    memory: RetrievalMemory
    gate: Optional[nn.Module] = None
    key: str = "features"
    fixed_weights: Sequence[float] = (0.5, 0.5, 0.0)
    thresholds: Optional[Dict] = None

    @torch.no_grad()
    def __call__(
        self,
        features: torch.Tensor,
        mag_index: Optional[torch.Tensor],
        p_param: torch.Tensor,
        patient_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        ev = self.memory(features, mag_index, query_patient_ids=patient_ids)
        p_param = p_param.to(ev["p_img"].device, torch.float32)
        n_frac = ev["n_distinct_patients"].to(torch.float32) / max(self.memory.image.k, 1)

        if self.gate is not None:
            p_final, weights = self.gate(
                p_param, ev["p_img"], ev["p_slide"], ev["agreement"], ev["top1_sim"], n_frac
            )
        else:
            w = torch.tensor(self.fixed_weights, dtype=torch.float32, device=p_param.device)
            mask = torch.tensor(self.memory.term_mask(), dtype=torch.float32, device=p_param.device)
            w = w * mask
            w = w / w.sum().clamp(min=1e-12)
            weights = w.view(1, 3).expand(p_param.shape[0], 3)
            p_final = (
                weights[:, 0] * p_param + weights[:, 1] * ev["p_img"] + weights[:, 2] * ev["p_slide"]
            )

        ev.update({"p_param": p_param, "p_final": p_final, "weights": weights,
                   "n_distinct_frac": n_frac})
        return ev
