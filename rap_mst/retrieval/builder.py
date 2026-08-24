"""Config-driven construction of the retrieval stack (mirrors ``models/builder.py``).

Nothing about the module is hardcoded in the scripts: paths, key, per-level
routing/k/cap/temperature, the gate and the ablation switches all come from the
``retrieval:`` block of ``config/config.yaml`` (or a ``--set`` override).

    build_memory(cfg, bank)          -> RetrievalMemory
    build_retrieval(cfg, ...)        -> RetrievalRunner (memory + optional gate)
    resolve_bank_path / gate_path    -> the fold's artefacts

``build_retrieval`` returns ``None`` when ``retrieval.enabled`` is false, so every
caller can stay a one-liner and exp1-exp3n behaviour is byte-identical when the
block is off.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import torch

from rap_mst.retrieval.bank import MemoryBank
from rap_mst.retrieval.gate import FusionGate
from rap_mst.retrieval.memory import LevelConfig, RetrievalMemory, RetrievalRunner

# Library logger -- propagates to the `rap_mst` package logger (see logging_utils).
logger = logging.getLogger(__name__)

#: Defaults mirror config/config.yaml so a missing/partial block still builds the
#: measured configuration rather than something arbitrary.
DEFAULT_IMAGE = dict(enabled=True, route="same_mag", k=15, per_patient_cap=3, temperature=0.07)
DEFAULT_SLIDE = dict(enabled=True, route="all", k=5, per_patient_cap=1, temperature=0.07)


def retrieval_cfg(cfg):
    """The ``retrieval:`` node, or ``None`` when the block is absent."""
    return getattr(cfg, "retrieval", None)


def is_enabled(cfg) -> bool:
    rcfg = retrieval_cfg(cfg)
    return bool(rcfg is not None and getattr(rcfg, "enabled", False))


def _fmt(template: str, experiment: str, fold: Optional[int]) -> str:
    return str(template).format(experiment=experiment, fold="?" if fold is None else fold)


def resolve_bank_path(cfg, experiment: str, fold: Optional[int]) -> Path:
    rcfg = retrieval_cfg(cfg)
    template = getattr(rcfg, "bank_path", "analysis/retrieval/{experiment}/bank_fold{fold}.npz")
    return Path(_fmt(template, experiment, fold))


def resolve_gate_path(cfg, experiment: str) -> Path:
    rcfg = retrieval_cfg(cfg)
    template = getattr(rcfg, "gate_path", "analysis/retrieval/{experiment}/gate.pt")
    return Path(_fmt(template, experiment, None))


def build_memory(cfg, bank: MemoryBank) -> RetrievalMemory:
    """Assemble the two views over an already-loaded bank."""
    rcfg = retrieval_cfg(cfg)
    levels = getattr(rcfg, "levels", None)
    return RetrievalMemory(
        bank=bank,
        key=getattr(rcfg, "key", "features"),
        key_transform=getattr(rcfg, "key_transform", "none"),
        image=LevelConfig.from_cfg(getattr(levels, "image", None), **DEFAULT_IMAGE),
        slide=LevelConfig.from_cfg(getattr(levels, "slide", None), **DEFAULT_SLIDE),
        merge_levels=bool(getattr(rcfg, "merge_levels", False)),
        block_query_patients=bool(getattr(rcfg, "block_query_patients", True)),
    )


def build_retrieval(
    cfg,
    experiment: str,
    fold: Optional[int] = None,
    device: Optional[torch.device] = None,
    bank_path: Optional[str] = None,
    gate_path: Optional[str] = None,
    require_gate: Optional[bool] = None,
    assert_disjoint_with=None,
) -> Optional[RetrievalRunner]:
    """Load the fold's bank (+ the shared gate) and return an inference runner.

    Parameters
    ----------
    assert_disjoint_with:
        Patient ids that must NOT be in the bank (the split being evaluated).
        The hard leakage guard -- see ``docs/retrieval.md`` Stage B1.
    """
    if not is_enabled(cfg):
        return None
    rcfg = retrieval_cfg(cfg)
    device = device or torch.device("cpu")
    key = getattr(rcfg, "key", "features")
    key_transform = getattr(rcfg, "key_transform", "none")

    path = Path(bank_path) if bank_path else resolve_bank_path(cfg, experiment, fold)
    bank = MemoryBank.load(path)
    bank.assert_compatible(key=key, key_transform=key_transform,
                           experiment=experiment, fold=fold)
    if assert_disjoint_with is not None:
        bank.assert_disjoint(assert_disjoint_with, what="evaluated")
    bank.to(device)
    summary = bank.summary()
    logger.info(
        f"retrieval: bank {path.name} loaded from {path.parent} -- "
        f"{summary['image_rows']} image + {summary['slide_rows']} slide rows, "
        f"{summary['patients']} patients, key={summary['key']}[{summary['dim']}]"
        f"{'' if summary['key_transform'] == 'none' else ' transform=' + summary['key_transform']}, "
        f"encoder={bank.meta.get('encoder_experiment', '?')} fold={bank.meta.get('fold', '?')} "
        f"built={bank.meta.get('created', '?')}"
    )
    if assert_disjoint_with is not None:
        logger.info(
            f"retrieval: leakage guard passed -- none of the {len(assert_disjoint_with)} "
            "evaluated patients is in the bank"
        )

    memory = build_memory(cfg, bank)

    gate_cfg = getattr(rcfg, "gate", None)
    want_gate = bool(getattr(gate_cfg, "enabled", True)) if require_gate is None else require_gate
    gate = None
    gate_state: Dict = {}
    if want_gate:
        gpath = Path(gate_path) if gate_path else resolve_gate_path(cfg, experiment)
        gate, gate_state = FusionGate.load(gpath, map_location=device)
        gate.to(device)
        if list(gate.mask.tolist()) != [float(m) for m in memory.term_mask()]:
            raise ValueError(
                f"Gate term mask {gate.mask.tolist()} != current level configuration "
                f"{memory.term_mask()} (param, img, slide).\n"
                "An ablation that removes an evidence term changes what the gate was "
                "fitted on, so reusing the old gate would be dishonest. Either:\n"
                "  (a) refit for this ablation -- run scripts/train_gate.py with the SAME "
                "--set flags and point --gate at the new gate.pt, or\n"
                "  (b) use the fixed-alpha path -- add --set retrieval.gate.enabled=false "
                "(weights are re-normalised over the active terms)."
            )
        mw = gate_state.get("mean_weights")
        logger.info(
            f"retrieval: gate {gpath.name} loaded from {gpath.parent} -- "
            f"fitted on {gate_state.get('n_rows', '?')} rows / "
            f"{gate_state.get('n_patients', '?')} patients ({gate_state.get('fit_on', '?')}), "
            f"mean weights={tuple(round(w, 3) for w in mw) if mw else '?'}, "
            f"thresholds={'locked' if gate_state.get('thresholds') else 'absent'}"
        )
    else:
        logger.info("retrieval: gate disabled -- blending at fixed weights (ablation)")

    fixed = tuple(getattr(gate_cfg, "fixed_weights", None) or getattr(gate_cfg, "init_weights", None)
                  or (0.5, 0.5, 0.0)) if gate_cfg is not None else (0.5, 0.5, 0.0)

    return RetrievalRunner(
        memory=memory,
        gate=gate,
        key=key,
        fixed_weights=fixed,
        thresholds=gate_state.get("thresholds"),
    )


def new_gate(cfg, term_mask) -> FusionGate:
    """A freshly-initialised gate matching the config (used by Stage B2)."""
    gate_cfg = getattr(retrieval_cfg(cfg), "gate", None)
    return FusionGate(
        hidden=int(getattr(gate_cfg, "hidden", 16)),
        init_weights=tuple(getattr(gate_cfg, "init_weights", (0.5, 0.5, 0.0))),
        init_weight_floor=float(getattr(gate_cfg, "init_weight_floor", 0.01)),
        term_mask=term_mask,
    )
