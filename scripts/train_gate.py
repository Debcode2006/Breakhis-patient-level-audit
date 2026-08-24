"""Stage B2 -- fit the fusion gate, once, on pooled out-of-fold validation data.

What this does, in words
------------------------
The gate answers one question: *for this image, how much should I trust the
classifier versus the memory?* It is a 5 -> 16 -> 3 MLP (~100 parameters) that
reads five cheap signals -- the head's probability, how confident the head is, how
much the neighbourhood agrees, how similar the top match is, and how many distinct
patients contributed -- and outputs three weights that sum to 1::

    p_final = w_param * p_param + w_img * p_img + w_slide * p_slide

Why once, on pooled data
------------------------
The five folds' validation sets are a disjoint partition of all 66 CV patients, so
concatenating them gives full out-of-fold coverage -- every patient scored by a
model that never trained on it. ``docs/results/threshold_calibration.md`` §4 is the
cautionary tale: thresholds fitted per fold on ~13 patients swung from 0.05 to
0.91 and *hurt* test. Do not fit this per fold.

This script also locks the **pooled-OOF decision thresholds** for p_param, p_img,
p_slide and p_final (image and patient level) using the same sweep as
``scripts/threshold_calibration.py``, and stores them inside ``gate.pt`` so
Stage C can report test metrics at a threshold that never saw the test set.

Output: ``analysis/retrieval/<experiment>/gate.pt`` (+ ``gate_fit.json``).

Examples
--------
    python scripts/train_gate.py --experiment exp3n
    python scripts/train_gate.py --experiment exp3n --set retrieval.levels.slide.enabled=false
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rap_mst.data.datamodule import BreaKHisDataModule  # noqa: E402
from rap_mst.data.splits import get_fold, load_splits  # noqa: E402
from rap_mst.engine.evaluator import evaluate  # noqa: E402
from rap_mst.experiments import encoder_experiment, experiment_names  # noqa: E402
from rap_mst.models.builder import build_model  # noqa: E402
from rap_mst.retrieval.builder import (  # noqa: E402
    build_retrieval, new_gate, resolve_bank_path, resolve_gate_path, retrieval_cfg,
)
from rap_mst.retrieval.gate import FusionGate, fit_gate  # noqa: E402
from rap_mst.utils import reporting  # noqa: E402
from rap_mst.utils.checkpoint import load_checkpoint  # noqa: E402
from rap_mst.utils.config import (  # noqa: E402
    apply_overrides, load_config, merge_section, parse_set_overrides,
)
from rap_mst.utils.logging_utils import get_logger  # noqa: E402
from rap_mst.utils.metrics import best_accuracy_threshold, compute_metrics  # noqa: E402
from rap_mst.utils.runs import available_folds, find_fold_run  # noqa: E402
from rap_mst.utils.seed import seed_everything  # noqa: E402

PROB_COLUMNS = ("prob_param", "prob_img", "prob_slide", "prob_final")


def resolve_experiment(cli_value, base_cfg) -> str:
    """CLI wins; otherwise the config's declared base encoder (``retrieval.base_experiment``)."""
    name = cli_value or str(getattr(retrieval_cfg(base_cfg), "base_experiment", "exp3n"))
    if name not in experiment_names():
        raise SystemExit(
            f"Unknown experiment '{name}'. Set retrieval.base_experiment to one of "
            f"{experiment_names()} or pass --experiment."
        )
    return name


def fold_config(args, base_cfg, encoder: str, fold: int):
    """The run's own config (matches the weights) + the current `retrieval:` block."""
    run_dir = find_fold_run(encoder, Path(base_cfg.logging.log_root), fold)
    cfg = load_config(str(run_dir / "config.yaml"))
    cfg = merge_section(cfg, "retrieval", base_cfg)
    cfg = apply_overrides(cfg, parse_set_overrides(args.set))
    return run_dir, cfg


def collect_fold(args, base_cfg, experiment: str, encoder: str, fold: int,
                 device: torch.device, logger) -> Tuple[Dict[str, np.ndarray], Dict[str, list]]:
    """Score fold `fold`'s validation split against *its own* bank.

    Returns ``(columns_for_the_fit, raw_evaluate_output)`` -- the second is kept so
    the pooled retrieval diagnostics / health checks run on all five folds at once.
    """
    run_dir, cfg = fold_config(args, base_cfg, encoder, fold)
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    logger.info(f"fold {fold}: loading {run_dir.name} and scanning the dataset ...")

    dm = BreaKHisDataModule(cfg)
    dm.setup_fold(fold)
    val_patients = get_fold(dm.splits, fold)["val_patients"]

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # require_gate=False: the gate is what we are about to fit.
    runner = build_retrieval(
        cfg, experiment=experiment, fold=fold, device=device,
        bank_path=args.bank_dir and str(Path(args.bank_dir) / f"bank_fold{fold}.npz"),
        require_gate=False,
        assert_disjoint_with=val_patients,     # hard leakage guard (Stage B1 rule)
    )
    if runner is None:
        raise SystemExit("retrieval.enabled is false -- enable it (or use --experiment exp5).")

    amp = bool(cfg.train.mixed_precision) and device.type == "cuda"
    logger.info(f"fold {fold}: scoring {len(dm.val_set)} validation images "
                f"({len(val_patients)} patients) against a {len(runner.memory.bank)}-row bank ...")
    _, raw = evaluate(model, dm.val_dataloader(), device, amp=amp, retrieval=runner)

    k_img = max(runner.memory.image.k, 1)
    diag = reporting.retrieval_diagnostics(
        raw, k=k_img, log_subtypes=False, route=runner.memory.image.route
    )
    logger.info(
        f"fold {fold}: {len(raw['label'])} images / {len(set(raw['patient_id']))} patients"
        f" | top1={diag['mean_top1_similarity']:.4f}"
        f" neighbours={diag['mean_neighbours']:.1f}"
        f" distinct_patients={diag['mean_distinct_patients']:.1f}"
        f" same_mag={diag['same_magnification_rate']:.3f}"
        f" img/slide disagreement={(diag.get('levels') or {}).get('img_slide_disagreement', float('nan')):.1%}"
    )
    part = {
        "fold": np.full(len(raw["label"]), fold, dtype=int),
        "label": np.asarray(raw["label"], dtype=int),
        "patient_id": np.asarray(raw["patient_id"]),
        "prob_param": np.asarray(raw["prob_param"], dtype=float),
        "prob_img": np.asarray(raw["prob_img"], dtype=float),
        "prob_slide": np.asarray(raw["prob_slide"], dtype=float),
        "agreement": np.asarray(raw["agreement"], dtype=float),
        "top1_sim": np.asarray(raw["top1_sim"], dtype=float),
        "n_distinct_frac": np.asarray(raw["n_distinct_patients"], dtype=float) / k_img,
        "term_mask": np.asarray(runner.memory.term_mask(), dtype=float),
    }
    return part, raw


def to_patient_level(pids: np.ndarray, labels: np.ndarray, probs: np.ndarray):
    """Mean-pool image probabilities per patient (the headline BreaKHis metric)."""
    uniq = np.unique(pids)
    p_lab = np.asarray([int(labels[pids == p][0]) for p in uniq], dtype=int)
    p_prob = np.asarray([float(probs[pids == p].mean()) for p in uniq], dtype=float)
    return p_lab, p_prob


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage B2: fit the retrieval fusion gate.")
    ap.add_argument("--experiment", default=None, choices=experiment_names(),
                    help="Retrieval identity (paths + bank metadata). "
                         "Default: retrieval.base_experiment from the config (exp3n, D9).")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--bank-dir", default=None, help="Override the directory holding bank_fold*.npz.")
    ap.add_argument("--out", default=None, help="Override the gate.pt output path.")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="Default: every trained fold.")
    ap.add_argument("--set", action="append", default=[],
                    help="Override any config key (repeatable).")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    if retrieval_cfg(base_cfg) is None:
        raise SystemExit(f"No `retrieval:` block in {args.config}.")
    base_cfg = apply_overrides(base_cfg, {"retrieval.enabled": True})

    experiment = resolve_experiment(args.experiment, base_cfg)
    encoder = encoder_experiment(experiment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_root = Path(base_cfg.logging.log_root)

    folds = args.folds if args.folds else available_folds(encoder, log_root)
    if not folds:
        raise SystemExit(f"No trained folds found for encoder '{encoder}' under {log_root}.")

    gate_path = Path(args.out) if args.out else resolve_gate_path(base_cfg, experiment)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger("rap_mst.train_gate", gate_path.parent / "train_gate.log")

    fit_cfg = retrieval_cfg(base_cfg).gate
    if str(getattr(fit_cfg, "fit_on", "pooled_oof")) != "pooled_oof":
        raise SystemExit(
            "retrieval.gate.fit_on must be 'pooled_oof'. Per-fold fitting on ~13 patients "
            "overfits the operating point (docs/results/threshold_calibration.md §4)."
        )

    env = reporting.collect_env(device)
    reporting.report_retrieval_config(
        logger, retrieval_cfg(base_cfg), title="STAGE B2 -- FUSION GATE (pooled out-of-fold)",
        extra=[
            reporting._rule(),
            reporting.kv("experiment", f"{experiment}  (encoder: {encoder})"),
            reporting.kv("folds", folds),
            reporting.kv("bank", str(args.bank_dir
                                     or resolve_bank_path(base_cfg, experiment, "{fold}"))),
            reporting.kv("output", str(gate_path)),
            reporting.kv("fit", f"epochs={getattr(fit_cfg, 'epochs', 400)} "
                                f"lr={getattr(fit_cfg, 'lr', 0.01)} "
                                f"weight_decay={getattr(fit_cfg, 'weight_decay', 0.0)}"),
            reporting.kv("device", f"{env['device']}  ({env['gpu_name']})"),
        ],
    )

    # ----- collect out-of-fold validation rows ----- #
    parts: List[Dict[str, np.ndarray]] = []
    pooled_raw: Dict[str, list] = {}
    for f in folds:
        part, raw = collect_fold(args, base_cfg, experiment, encoder, f, device, logger)
        parts.append(part)
        for key, values in raw.items():
            pooled_raw.setdefault(key, []).extend(values)
    term_masks = {tuple(p["term_mask"].tolist()) for p in parts}
    if len(term_masks) != 1:
        raise SystemExit(f"Folds disagree on which retrieval terms are active: {term_masks}")
    term_mask = list(term_masks.pop())

    pool = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "term_mask"}
    labels = pool["label"]
    pids = pool["patient_id"]

    # ----- sanity checks before fitting ----- #
    splits = load_splits(fold_config(args, base_cfg, encoder, folds[0])[1].data.splits_path)
    test_patients = set(splits["test_patients"])
    n_patients = len(set(pids.tolist()))
    leaked = sorted(set(pids.tolist()) & test_patients)
    if leaked:
        raise AssertionError(f"LEAKAGE: test patients present in the pooled OOF set: {leaked}")
    if n_patients != 66:
        reporting.warn_banner(
            logger, "Pooled OOF coverage is not the full 66 CV patients",
            f"got {n_patients} distinct patients from folds {folds};",
            "a fold's validation set was skipped -- the gate is being fitted on partial coverage.",
        )

    reporting.section(logger, "POOLED OOF SANITY", [
        reporting.kv("rows", len(labels)),
        reporting.kv("distinct patients", f"{n_patients}  (expected 66)"),
        reporting.kv("any test patient", "no ✓"),
        reporting.kv("class balance", f"benign {(labels == 0).sum()} / malignant {(labels == 1).sum()}"),
        reporting.kv("active terms (param, img, slide)", term_mask),
        reporting.kv("folds pooled", f"{folds}  (disjoint validation partition)"),
    ])

    # Retrieval quality on the pooled OOF set, independent of any classifier metric.
    dcfg = getattr(retrieval_cfg(base_cfg), "diagnostics", None)
    image_cfg = getattr(getattr(retrieval_cfg(base_cfg), "levels", None), "image", None)
    k_img = int(getattr(image_cfg, "k", 15))
    pooled_diag = reporting.retrieval_diagnostics(
        pooled_raw, k=k_img,
        log_subtypes=bool(getattr(dcfg, "log_retrieved_subtypes", True)),
        route=str(getattr(image_cfg, "route", "same_mag")),
    )
    reporting.report_retrieval_diagnostics(
        logger, pooled_diag, title="RETRIEVAL DIAGNOSTICS (pooled OOF validation)"
    )
    reporting.check_retrieval_health(
        logger, pooled_diag, k=k_img,
        route=str(getattr(image_cfg, "route", "same_mag")),
        gate_learned=False,   # the gate is fitted below; its collapse check follows
    )

    # ----- fit ----- #
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    features = FusionGate.build_features(
        t(pool["prob_param"]), t(pool["agreement"]), t(pool["top1_sim"]), t(pool["n_distinct_frac"])
    )
    probs3 = torch.stack([t(pool["prob_param"]), t(pool["prob_img"]), t(pool["prob_slide"])], dim=1)

    torch.manual_seed(base_cfg.seed)   # before init: the hidden layer is random
    gate = new_gate(base_cfg, term_mask=term_mask)
    stats = fit_gate(
        gate, features, probs3, t(labels),
        epochs=int(getattr(fit_cfg, "epochs", 400)),
        lr=float(getattr(fit_cfg, "lr", 0.01)),
        weight_decay=float(getattr(fit_cfg, "weight_decay", 0.0)),
        logger=logger,
    )

    with torch.no_grad():
        weights = gate.weights_from_features(features)
        p_final = (weights * probs3).sum(dim=1).numpy()
    pool["prob_final"] = p_final

    # ----- pooled-OOF thresholds (locked here, applied in Stage C) ----- #
    thresholds: Dict[str, Dict] = {"image": {}, "patient": {}}
    val_report: Dict[str, Dict] = {}
    for col in PROB_COLUMNS:
        probs = pool[col]
        thresholds["image"][col] = best_accuracy_threshold(labels, probs)
        p_lab, p_prob = to_patient_level(pids, labels, probs)
        thresholds["patient"][col] = best_accuracy_threshold(p_lab, p_prob)
        img_m = compute_metrics(labels, (probs >= 0.5).astype(int), probs)
        pat_m = compute_metrics(p_lab, (p_prob >= 0.5).astype(int), p_prob)
        val_report[col] = {
            "image_acc@0.5": img_m["accuracy"], "image_auc": img_m["auc"],
            "image_sens": img_m["sensitivity"], "image_spec": img_m["specificity"],
            "patient_acc@0.5": pat_m["accuracy"],
            "image_threshold": thresholds["image"][col]["threshold"],
            "patient_threshold": thresholds["patient"][col]["threshold"],
        }

    mean_w = weights.mean(dim=0).tolist()
    gain = val_report["prob_final"]["image_acc@0.5"] - val_report["prob_param"]["image_acc@0.5"]
    reporting.section(logger, "GATE RESULT (pooled OOF validation)", [
        reporting.kv("BCE", f"{stats['initial_bce']:.5f} -> {stats['final_bce']:.5f}"),
        reporting.kv("mean weights (param, img, slide)",
                     f"({mean_w[0]:.3f}, {mean_w[1]:.3f}, {mean_w[2]:.3f})"),
        reporting.kv("val acc  p_param", f"{val_report['prob_param']['image_acc@0.5']:.4f}"),
        reporting.kv("val acc  p_img", f"{val_report['prob_img']['image_acc@0.5']:.4f}"),
        reporting.kv("val acc  p_slide", f"{val_report['prob_slide']['image_acc@0.5']:.4f}"),
        reporting.kv("val acc  p_final", f"{val_report['prob_final']['image_acc@0.5']:.4f} "
                                         f"({gain:+.4f} vs p_param)"),
        reporting.kv("locked image threshold (p_final)",
                     f"{thresholds['image']['prob_final']['threshold']:.3f}"),
        reporting.kv("locked patient threshold (p_final)",
                     f"{thresholds['patient']['prob_final']['threshold']:.3f}"),
    ])
    if gain <= 0:
        reporting.warn_banner(
            logger, "The memory is not helping on validation",
            f"p_final ({val_report['prob_final']['image_acc@0.5']:.4f}) does not beat "
            f"p_param ({val_report['prob_param']['image_acc@0.5']:.4f}).",
            "Report this as 'the module does not help on this encoder' -- do not tune until it does.",
        )
    if mean_w[0] > 0.9:
        reporting.warn_banner(
            logger, "Gate collapsed toward the parametric head",
            f"mean w_param = {mean_w[0]:.3f}.",
            "docs/retrieval.md §6 predicts alpha~0.5 for SupCon encoders and ~0.8 for exp1, so a high "
            "w_param is a real possible outcome, not necessarily a bug.",
        )

    gate.save(gate_path, extra={
        "experiment": experiment,
        "encoder_experiment": encoder,
        "folds": list(folds),
        "fit_on": "pooled_oof",
        "n_rows": int(len(labels)),
        "n_patients": int(n_patients),
        "mean_weights": mean_w,
        "fit_stats": stats,
        "thresholds": thresholds,
        "val_report": val_report,
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    logger.info(f"Saved gate -> {gate_path}")

    with (gate_path.parent / "gate_fit.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "experiment": experiment, "encoder": encoder, "folds": list(folds),
            "n_rows": int(len(labels)), "n_patients": int(n_patients),
            "term_mask": term_mask, "mean_weights": mean_w,
            "fit_stats": stats, "val_report": val_report, "thresholds": thresholds,
            "retrieval_diagnostics": pooled_diag,
        }, fh, indent=2)
    logger.info(f"Saved fit report -> {gate_path.parent / 'gate_fit.json'}")


if __name__ == "__main__":
    main()
