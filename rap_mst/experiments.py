"""Experiment presets.

The three research experiments differ *only* in which optional modules and loss
terms are active. Rather than duplicate whole configs, each preset is a small set
of dotted-key overrides applied on top of the base ``config.yaml``. This keeps
the YAML the single source of truth for hyper-parameters while making experiment
selection a one-token change (``--experiment exp2``).

    exp1   Swin backbone + classifier
    exp2   + magnification embedding
    exp3   + magnification embedding + projection head + SupCon loss
    exp3n  exp3 *without* the magnification embedding -- the missing cell of the
           ladder (SupCon alone) and the **chosen encoder for the retrieval
           module**.  See `docs/results/magnification_audit.md` Part 1: the magnification block is a
           4-value per-magnification logit offset for the linear head (worth
           +0.0002 AUC) but it makes the retrieval key and the SupCon projection
           ~100% magnification-locked.  exp1-exp3 are left byte-identical.
    exp5   exp3n + the Retrieval Memory module (`retrieval.enabled: true`).
    expfm  CTransPath (frozen pathology foundation model) + a **linear** head --
           the foundation-model baseline row of the paper's main table
           (the paper).
    expfm_mlp  same, with a one-hidden-layer MLP head instead (the "optionally a
           small MLP" variant of the same experiment).

The two ``expfm*`` presets are **not** rungs of the exp1->exp3n ladder: they do not
share the Swin/FPN architecture, so they are not an ablation of one model but an
external baseline. They are registered here anyway so that `EXPERIMENT_DIRS` -- and
therefore every analysis script that resolves ``runs/<experiment>`` through it
(`diagnose_folds.py`, `threshold_calibration.py`, ...) -- picks them up with no
edits. They are trained by `scripts/train_linear_probe.py`, never by
`scripts/train.py`; :func:`assert_not_foundation` is what makes that a loud error
instead of a silently-wrong Swin run.

`exp5` needs **no new encoder training**: it is exp3n's frozen checkpoints plus
the bank + gate built on top of them (`docs/retrieval.md` §10.5, two-stage by design).
Use `--experiment exp3n` to train, then the retrieval scripts with
`--experiment exp5`; :func:`encoder_experiment` maps exp5 -> exp3n so they load
exp3n's `runs/` directory. Training `--experiment exp5` directly would just
re-train an exp3n-equivalent encoder into a second run directory -- 10 GPU hours
for nothing.
"""

from __future__ import annotations

from typing import Dict

EXPERIMENT_PRESETS: Dict[str, Dict] = {
    "exp1": {
        "experiment.name": "exp1_swin_cls",
        "model.magnification.enabled": False,
        "model.projection.enabled": False,
        "loss.ce_weight": 1.0,
        "loss.supcon_weight": 0.0,
        "train.epochs": 10,  # lighter baseline; exp2/exp3 keep the config default (15)
    },
    "exp2": {
        "experiment.name": "exp2_swin_mag_cls",
        "model.magnification.enabled": True,
        "model.projection.enabled": False,
        "loss.ce_weight": 1.0,
        "loss.supcon_weight": 0.0,
    },
    "exp3": {
        "experiment.name": "exp3_swin_mag_supcon_cls",
        "model.magnification.enabled": True,
        "model.projection.enabled": True,
        "loss.ce_weight": 1.0,
        "loss.supcon_weight": 0.4,
    },
    "exp3n": {
        "experiment.name": "exp3n_swin_supcon_cls",
        "model.magnification.enabled": False,
        "model.projection.enabled": True,
        "loss.ce_weight": 1.0,
        "loss.supcon_weight": 0.4,
    },
    "exp5": {
        # Same encoder as exp3n (identical model/loss flags) + the memory module.
        "experiment.name": "exp5_retrieval",
        "model.magnification.enabled": False,
        "model.projection.enabled": True,
        "loss.ce_weight": 1.0,
        "loss.supcon_weight": 0.4,
        "retrieval.enabled": True,
    },
    "expfm": {
        # Frozen CTransPath + linear head. The `model.*` flags below are NOT used
        # to build anything (the probe never touches `build_model`); they are set
        # so the saved config reads honestly -- no Swin component is active.
        "experiment.name": "expfm_ctranspath_linear",
        "model.magnification.enabled": False,
        "model.projection.enabled": False,
        "loss.ce_weight": 1.0,
        "loss.supcon_weight": 0.0,
        "foundation.probe.head": "linear",
    },
    "expfm_mlp": {
        "experiment.name": "expfm_ctranspath_mlp",
        "model.magnification.enabled": False,
        "model.projection.enabled": False,
        "loss.ce_weight": 1.0,
        "loss.supcon_weight": 0.0,
        "foundation.probe.head": "mlp",
    },
}


#: Presets trained by ``scripts/train_linear_probe.py`` on cached frozen features,
#: **not** by ``scripts/train.py``. Guarded by :func:`assert_not_foundation`.
FOUNDATION_EXPERIMENTS = frozenset({"expfm", "expfm_mlp"})


#: Retrieval presets reuse a *frozen* encoder trained under another preset, so the
#: bank / gate / test scripts must look for checkpoints under that experiment's
#: run directory. Anything not listed encodes with its own checkpoints.
ENCODER_EXPERIMENT: Dict[str, str] = {"exp5": "exp3n"}


#: Short experiment key -> the ``runs/<dir>`` name that preset writes into.
#: Derived from the presets so a new experiment needs **no** edits in the
#: analysis scripts (train / test / threshold calibration / UMAP / probes all
#: resolve run directories through this).
EXPERIMENT_DIRS: Dict[str, str] = {
    key: preset["experiment.name"] for key, preset in EXPERIMENT_PRESETS.items()
}


def experiment_names() -> list:
    """Valid ``--experiment`` values, for argparse ``choices``."""
    return list(EXPERIMENT_PRESETS)


def is_foundation(name: str) -> bool:
    """Is ``name`` a frozen-foundation-model preset (trained by the probe scripts)?"""
    return name.strip().lower() in FOUNDATION_EXPERIMENTS


def assert_not_foundation(name: str | None, script: str) -> None:
    """Refuse a foundation preset in a Swin-pipeline script.

    ``expfm`` has no Swin backbone, no FPN and no checkpoint ``build_model`` can
    reconstruct. Running it through ``train.py`` would happily train a Swin-Tiny
    into ``runs/expfm_ctranspath_linear/`` and every downstream table would then be
    comparing the wrong thing -- so fail loudly and name the right script.
    """
    if name and is_foundation(name):
        raise SystemExit(
            f"'{name}' is a frozen foundation-model baseline and cannot be run by "
            f"{script}: it has no Swin backbone to build or load.\n"
            f"  Stage F1: python scripts/extract_foundation_features.py\n"
            f"  Stage F2: python scripts/train_linear_probe.py --experiment {name} --fold <k>\n"
            f"  Stage F3: python scripts/test_linear_probe.py  --experiment {name} --fold <k>"
        )


def resolve_experiment_dir(name: str) -> str:
    """``'exp3n'`` -> ``'exp3n_swin_supcon_cls'``; a full dir name passes through."""
    return EXPERIMENT_DIRS.get(name.strip().lower(), name)


def encoder_experiment(name: str) -> str:
    """Which experiment's checkpoints does ``name`` encode with? (``exp5`` -> ``exp3n``)."""
    return ENCODER_EXPERIMENT.get(name.strip().lower(), name.strip().lower())


def experiment_key_from_dir(dir_name: str) -> str:
    """``'exp3n_swin_supcon_cls'`` -> ``'exp3n'``; an unknown name passes through."""
    for key, value in EXPERIMENT_DIRS.items():
        if value == dir_name:
            return key
    return dir_name


def get_experiment_preset(name: str) -> Dict:
    key = name.strip().lower()
    if key not in EXPERIMENT_PRESETS:
        raise ValueError(
            f"Unknown experiment '{name}'. Choose from {list(EXPERIMENT_PRESETS)}."
        )
    return EXPERIMENT_PRESETS[key]
