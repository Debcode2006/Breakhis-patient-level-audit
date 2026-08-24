"""Train one cross-validation fold.

Assembles config -> data -> model -> loss -> optimizer -> trainer and runs the
fit loop. Experiment selection, fold, and any config key can be set from the CLI
without editing the YAML.

Examples
--------
    python scripts/train.py --experiment exp1 --fold 0
    python scripts/train.py --experiment exp3 --fold 2 --set train.epochs=30
    python scripts/train.py --experiment exp3n --fold 0     # SupCon without magnification
    python scripts/train.py --resume runs/exp1_swin_cls/<run>/checkpoints/last.pt

``--experiment`` accepts any key registered in ``rap_mst/experiments.py``; adding a
preset there is enough to make it trainable here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rap_mst.data.datamodule import BreaKHisDataModule  # noqa: E402
from rap_mst.engine.optim import build_optimizer, build_scheduler  # noqa: E402
from rap_mst.engine.trainer import Trainer  # noqa: E402
from rap_mst.experiments import (  # noqa: E402
    assert_not_foundation, experiment_names, get_experiment_preset,
)
from rap_mst.losses.builder import build_loss  # noqa: E402
from rap_mst.models.builder import build_model  # noqa: E402
from rap_mst.utils import reporting  # noqa: E402
from rap_mst.utils.config import apply_overrides, load_config, parse_set_overrides  # noqa: E402
from rap_mst.utils.experiment import ExperimentDir  # noqa: E402
from rap_mst.utils.logging_utils import get_logger  # noqa: E402
from rap_mst.utils.seed import seed_everything  # noqa: E402


def build_config(args):
    cfg = load_config(args.config)
    overrides = {}
    if args.experiment:
        overrides.update(get_experiment_preset(args.experiment))
    if args.fold is not None:
        overrides["train.fold"] = args.fold
    overrides.update(parse_set_overrides(args.set))
    return apply_overrides(cfg, overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RAP-MST fold.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--experiment", default=None, choices=experiment_names(),
                        help="Apply an experiment preset (overrides model/loss flags). "
                             "Choices come from rap_mst/experiments.py.")
    parser.add_argument("--fold", type=int, default=None, help="CV fold index (0-4).")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--set", action="append", default=[],
                        help="Override any config key: --set train.epochs=30 (repeatable).")
    args = parser.parse_args()
    assert_not_foundation(args.experiment, "scripts/train.py")

    cfg = build_config(args)

    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold = cfg.train.fold
    expdir = ExperimentDir(cfg.logging.log_root, cfg.experiment.name, fold, tag="train")
    expdir.save_config(cfg)  # persist the EXACT config used

    # Logger created here (idempotent by name) so the startup/dataset/architecture
    # reports and the trainer share one console+file logger.
    logger = get_logger("rap_mst.train", expdir.log_path)
    reporting.report_startup(logger, cfg, device, fold)

    # Data (report + hard leakage guard before building the model).
    dm = BreaKHisDataModule(cfg)
    dm.setup_fold(fold)
    reporting.report_dataset(logger, dm, fold)

    # Model / loss / optim
    model = build_model(cfg)
    reporting.report_architecture(logger, model)
    class_weights = dm.class_weights().to(device)
    loss_fn = build_loss(cfg, class_weights=class_weights)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(dm.train_dataloader()))

    trainer = Trainer(cfg, model, dm, loss_fn, optimizer, scheduler, expdir, device)
    trainer.maybe_resume(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
