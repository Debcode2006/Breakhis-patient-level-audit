"""Evaluate a trained checkpoint on the permanent 16-patient held-out test set.

The test set is loaded from the saved splits and is NEVER seen during training or
model selection. The model architecture is reconstructed from the config stored
inside the checkpoint, so you only need to point at the checkpoint.

Stage C (retrieval)
-------------------
``--retrieval`` additionally loads that fold's memory bank and the shared fusion
gate and reports ``p_param`` / ``p_img`` / ``p_slide`` / ``p_final``, each at
threshold 0.5 **and** at the locked pooled-OOF threshold, so the numbers stay
comparable with ``docs/results/classifier_ladder.md`` and ``docs/results/threshold_calibration.md``. It also
dumps the retrieved exemplars for the named hard patients -- the interpretability
deliverable, which goes in the paper regardless of the accuracy delta.

Examples
--------
    python scripts/test.py --checkpoint runs/exp1_swin_cls/<run>/checkpoints/best.pt
    python scripts/test.py --checkpoint runs/exp3n_swin_supcon_cls/<run>/checkpoints/best.pt --retrieval
    python scripts/test.py --checkpoint <ckpt> --retrieval --set retrieval.levels.image.route=all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rap_mst.data.datamodule import BreaKHisDataModule  # noqa: E402
from rap_mst.engine.evaluator import RETRIEVAL_KEYS, evaluate  # noqa: E402
from rap_mst.experiments import (  # noqa: E402
    assert_not_foundation, experiment_key_from_dir, experiment_names,
)
from rap_mst.models.builder import build_model  # noqa: E402
from rap_mst.retrieval.builder import build_retrieval, retrieval_cfg  # noqa: E402
from rap_mst.utils import reporting  # noqa: E402
from rap_mst.utils.checkpoint import load_checkpoint  # noqa: E402
from rap_mst.utils.config import (  # noqa: E402
    Config, apply_overrides, load_config, merge_section, parse_set_overrides,
)
from rap_mst.utils.logging_utils import get_logger  # noqa: E402
from rap_mst.utils.metrics import compute_metrics, patient_level_metrics  # noqa: E402
from rap_mst.utils.seed import seed_everything  # noqa: E402

PROB_COLUMNS = ("prob_param", "prob_img", "prob_slide", "prob_final")


def _check_config_mismatch(logger, override_cfg, ckpt_cfg: dict) -> None:
    """Warn if a user-supplied config disagrees with the checkpoint's model config.

    Evaluating with a different architecture than the one that produced the
    weights silently loads a mismatched model; surface it loudly.
    """
    ckpt_model = ckpt_cfg.get("model", {})
    override_model = override_cfg.to_dict().get("model", {})
    if override_model != ckpt_model:
        reporting.warn_banner(
            logger, "Config mismatch between --config and checkpoint",
            "the model section differs from the config the checkpoint was trained with;",
            "state_dict loading may fail or evaluate a different architecture.",
        )


def _scored(labels: np.ndarray, probs: np.ndarray, pids: np.ndarray, thr: float) -> dict:
    """Image- and patient-level metrics for one probability column at one threshold."""
    preds = (probs >= thr).astype(int)
    out = compute_metrics(labels, preds, probs)
    out.update(patient_level_metrics(pids, labels, probs, threshold=thr))
    out["threshold"] = float(thr)
    return out


def retrieval_metrics(raw: dict, thresholds: dict | None) -> dict:
    """Every probability column at 0.5 and at its locked pooled-OOF threshold."""
    labels = np.asarray(raw["label"], dtype=int)
    pids = np.asarray(raw["patient_id"])
    report: dict = {}
    for col in PROB_COLUMNS:
        probs = np.asarray(raw[col], dtype=float)
        entry = {"at_0.5": _scored(labels, probs, pids, 0.5)}
        if thresholds:
            img_thr = thresholds.get("image", {}).get(col, {}).get("threshold")
            pat_thr = thresholds.get("patient", {}).get(col, {}).get("threshold")
            if img_thr is not None:
                entry["at_locked_image_threshold"] = _scored(labels, probs, pids, float(img_thr))
            if pat_thr is not None:
                pat = patient_level_metrics(pids, labels, probs, threshold=float(pat_thr))
                pat["threshold"] = float(pat_thr)
                entry["at_locked_patient_threshold"] = pat
        report[col] = entry
    return report


def _neighbours(raw: dict, i: int, prefix: str, top_k: int) -> list:
    """The top-k retrieved rows of one level for image ``i``: who, which subtype, how close."""
    pid_key = "neighbour_patients" if prefix == "neighbour" else "slide_patients"
    sub_key = "neighbour_subtypes" if prefix == "neighbour" else "slide_subtypes"
    sim_key = "neighbour_sims" if prefix == "neighbour" else "slide_sims"
    pids = [p for p in raw[pid_key][i].split(";") if p][:top_k]
    subs = [s for s in raw[sub_key][i].split(";") if s][:top_k]
    sims = [s for s in raw[sim_key][i].split(";") if s][:top_k]
    return [
        {"patient_id": p, "subtype": s, "similarity": float(sim)}
        for p, s, sim in zip(pids, subs, sims)
    ]


def dump_exemplars(raw: dict, cfg, out_dir: Path, logger) -> None:
    """Top-k retrieved neighbours for the named hard patients (paper deliverable)."""
    dcfg = getattr(retrieval_cfg(cfg), "diagnostics", None)
    if dcfg is None or not bool(getattr(dcfg, "dump_exemplars", True)):
        return
    wanted = list(getattr(dcfg, "dump_exemplars_for", []) or [])
    if not wanted:
        return
    top_k = int(getattr(dcfg, "exemplars_top_k", 5))
    per_patient = int(getattr(dcfg, "exemplars_per_patient", 5))

    ex_dir = out_dir / "exemplars"
    ex_dir.mkdir(parents=True, exist_ok=True)
    pids = np.asarray(raw["patient_id"])
    for pid in wanted:
        rows = np.nonzero(pids == pid)[0]
        if rows.size == 0:
            logger.info(f"  exemplars: {pid} not in the test split -- skipped")
            continue
        # Spread the sampled queries across the patient's images / magnifications.
        picks = rows[np.linspace(0, rows.size - 1, min(per_patient, rows.size)).astype(int)]
        payload = {
            "patient_id": pid,
            "label": int(raw["label"][picks[0]]),
            "n_test_images": int(rows.size),
            "queries": [
                {
                    "image_path": raw["image_path"][i],
                    "magnification": raw["magnification"][i],
                    "p_param": round(float(raw["prob_param"][i]), 6),
                    "p_img": round(float(raw["prob_img"][i]), 6),
                    "p_slide": round(float(raw["prob_slide"][i]), 6),
                    "p_final": round(float(raw["prob_final"][i]), 6),
                    "weights": {
                        "param": round(float(raw["w_param"][i]), 4),
                        "img": round(float(raw["w_img"][i]), 4),
                        "slide": round(float(raw["w_slide"][i]), 4),
                    },
                    "top1_sim": round(float(raw["top1_sim"][i]), 6),
                    "n_distinct_patients": int(raw["n_distinct_patients"][i]),
                    "retrieved_images": _neighbours(raw, i, "neighbour", top_k),
                    "retrieved_slides": _neighbours(raw, i, "slide", top_k),
                }
                for i in picks
            ],
        }
        path = ex_dir / f"{pid}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    logger.info(f"Saved exemplars -> {ex_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the held-out test set.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint (best.pt).")
    parser.add_argument("--config", default=None,
                        help="Optional config override; defaults to the config saved in the checkpoint.")
    parser.add_argument("--out", default=None, help="Directory for predictions/metrics (default: next to checkpoint).")
    parser.add_argument("--retrieval", action="store_true",
                        help="Stage C: blend the memory module's evidence into the prediction.")
    parser.add_argument("--retrieval-config", default="config/config.yaml",
                        help="Where the `retrieval:` block comes from (checkpoints predate it).")
    parser.add_argument("--experiment", default=None, choices=experiment_names(),
                        help="Retrieval identity for bank/gate paths (default: inferred from the checkpoint).")
    parser.add_argument("--bank", default=None, help="Explicit bank .npz (default: retrieval.bank_path).")
    parser.add_argument("--gate", default=None, help="Explicit gate .pt (default: retrieval.gate_path).")
    parser.add_argument("--set", action="append", default=[],
                        help="Override any config key, e.g. --set retrieval.levels.slide.enabled=false.")
    args = parser.parse_args()
    assert_not_foundation(args.experiment, "scripts/test.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out) if args.out else Path(args.checkpoint).parent.parent / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("rap_mst.test", out_dir / "test.log")

    ckpt = load_checkpoint(args.checkpoint, map_location=device)  # raises if missing

    # Reconstruct the exact config the checkpoint was trained with.
    if args.config:
        cfg = load_config(args.config)
        _check_config_mismatch(logger, cfg, ckpt.get("config", {}))
    else:
        cfg = Config(ckpt["config"])

    # The `retrieval:` block did not exist when these checkpoints were trained, so
    # it always comes from the current config file (model/data still come from the
    # checkpoint -- the architecture must match the weights).
    if args.retrieval:
        cfg = merge_section(cfg, "retrieval", load_config(args.retrieval_config))
        cfg = apply_overrides(cfg, {"retrieval.enabled": True})
    overrides = parse_set_overrides(args.set)
    if any(k.startswith("model.") for k in overrides):
        reporting.warn_banner(
            logger, "--set is overriding a model.* key",
            "the architecture must match the checkpoint's weights;",
            "state_dict loading may fail or evaluate a different model.",
        )
    cfg = apply_overrides(cfg, overrides)
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    # Data: held-out test set only.
    dm = BreaKHisDataModule(cfg)
    dm.setup_test()
    test_loader = dm.test_dataloader()

    n_patients = len(dm.splits["test_patients"])
    n_images = len(dm.test_set)
    reporting.report_test_header(
        logger, checkpoint=args.checkpoint, cfg=cfg, fold=cfg.train.fold,
        n_patients=n_patients, n_images=n_images,
    )

    # Model.
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    # Retrieval (Stage C) -- fold-matched bank + the one shared gate.
    runner = None
    if args.retrieval:
        experiment = args.experiment or experiment_key_from_dir(cfg.experiment.name)
        runner = build_retrieval(
            cfg, experiment=experiment, fold=cfg.train.fold, device=device,
            bank_path=args.bank, gate_path=args.gate,
            assert_disjoint_with=dm.splits["test_patients"],  # hard leakage guard
        )
        mem = runner.memory
        summary = mem.bank.summary()
        reporting.report_retrieval_config(
            logger, retrieval_cfg(cfg), title="RETRIEVAL MEMORY (Stage C)",
            extra=[
                reporting._rule(),
                reporting.kv("experiment", experiment),
                reporting.kv("bank rows", f"{summary['image_rows']} image + "
                                          f"{summary['slide_rows']} slide "
                                          f"({summary['patients']} patients)"),
                reporting.kv("bank key dim", f"{summary['key']}[{summary['dim']}]"),
                reporting.kv("mag shards", summary["mag_shard_sizes"]),
                reporting.kv("gate", "learned" if runner.gate is not None
                             else f"fixed {tuple(runner.fixed_weights)}"),
                reporting.kv("thresholds", "locked (pooled OOF)" if runner.thresholds
                             else "none -- 0.5 only"),
            ],
        )
        base = str(getattr(retrieval_cfg(cfg), "base_experiment", experiment))
        if base != experiment:
            reporting.warn_banner(
                logger, "Encoder is not the configured retrieval base",
                f"testing '{experiment}' but retrieval.base_experiment is '{base}'.",
                "The bank, the gate and the checkpoint must all come from one encoder.",
            )

    amp = bool(cfg.train.mixed_precision) and device.type == "cuda"
    logger.info(f"Evaluating {n_images} test images"
                f"{' with retrieval' if runner is not None else ''} ...")
    metrics, raw = evaluate(model, test_loader, device, amp=amp, retrieval=runner)

    reporting.report_test_metrics(logger, metrics, out_dir)

    if runner is None:
        with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        metrics_path, pred_path = out_dir / "test_metrics.json", out_dir / "test_predictions.csv"
        columns = ["image_path", "patient_id", "magnification", "label", "pred", "prob_malignant"]
        rows = lambda i: [  # noqa: E731
            raw["image_path"][i], raw["patient_id"][i], raw["magnification"][i],
            raw["label"][i], raw["pred"][i], f"{raw['prob'][i]:.6f}",
        ]
    else:
        dcfg = getattr(retrieval_cfg(cfg), "diagnostics", None)
        diagnostics = reporting.retrieval_diagnostics(
            raw,
            k=runner.memory.image.k,
            log_subtypes=bool(getattr(dcfg, "log_retrieved_subtypes", True)),
            route=runner.memory.image.route,
        )
        reporting.report_retrieval_diagnostics(logger, diagnostics)
        reporting.check_retrieval_health(
            logger, diagnostics, k=runner.memory.image.k,
            route=runner.memory.image.route, gate_learned=runner.gate is not None,
        )
        report = {
            "fold": cfg.train.fold,
            "retrieval": cfg.to_dict()["retrieval"],
            "gate_weights_mean": {
                "param": float(np.mean(raw["w_param"])),
                "img": float(np.mean(raw["w_img"])),
                "slide": float(np.mean(raw["w_slide"])),
            },
            "retrieval_diagnostics": diagnostics,
            "metrics": retrieval_metrics(raw, runner.thresholds),
        }
        metrics_path = out_dir / "test_metrics_retrieval.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        _log_retrieval_summary(logger, report)

        pred_path = out_dir / "test_predictions_retrieval.csv"
        columns = ["image_path", "patient_id", "magnification", "label", "pred", "prob_malignant",
                   *RETRIEVAL_KEYS]
        rows = lambda i: [  # noqa: E731
            raw["image_path"][i], raw["patient_id"][i], raw["magnification"][i],
            raw["label"][i], raw["pred"][i], f"{raw['prob'][i]:.6f}",
            *[raw[k][i] for k in RETRIEVAL_KEYS],
        ]
        dump_exemplars(raw, cfg, out_dir, logger)

    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(len(raw["label"])):
            writer.writerow(rows(i))

    logger.info(f"Saved metrics -> {metrics_path}")
    logger.info(f"Saved preds   -> {pred_path}")


def _log_retrieval_summary(logger, report: dict) -> None:
    rows = []
    for col, entry in report["metrics"].items():
        m = entry["at_0.5"]
        rows.append(
            f"{col:<11} acc={m['accuracy']:.4f}  auc={m['auc']:.4f}  "
            f"sens={m['sensitivity']:.3f}  spec={m['specificity']:.3f}  "
            f"pat_acc={m['patient_accuracy']:.4f}"
        )
    w = report["gate_weights_mean"]
    reporting.section(logger, "RETRIEVAL RESULTS (threshold 0.5)", rows + [
        "",
        reporting.kv("mean gate weights",
                     f"param={w['param']:.3f}  img={w['img']:.3f}  slide={w['slide']:.3f}"),
    ])


if __name__ == "__main__":
    main()
