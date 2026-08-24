"""Per-run experiment directory management.

Every run gets its own timestamped directory so previous runs are never
overwritten. The layout is:

    <log_root>/<experiment_name>/<timestamp>_fold<k>/
        config.yaml          # exact config used for this run
        train.log            # full console log
        metrics.csv          # per-epoch metrics
        checkpoints/
            best.pt
            last.pt
        tensorboard/         # TensorBoard event files
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from rap_mst.utils.config import Config, save_config


class ExperimentDir:
    """Create and expose the standard run directory structure."""

    def __init__(self, log_root: str | Path, experiment_name: str, fold: int, tag: str = "train") -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = Path(log_root) / experiment_name / f"{timestamp}_{tag}_fold{fold}"
        self.checkpoints = self.root / "checkpoints"
        self.tensorboard = self.root / "tensorboard"

        for d in (self.root, self.checkpoints, self.tensorboard):
            d.mkdir(parents=True, exist_ok=True)

        self.config_path = self.root / "config.yaml"
        self.log_path = self.root / f"{tag}.log"
        self.metrics_path = self.root / "metrics.csv"

    @property
    def best_ckpt(self) -> Path:
        return self.checkpoints / "best.pt"

    @property
    def last_ckpt(self) -> Path:
        return self.checkpoints / "last.pt"

    def save_config(self, cfg: Config | Dict[str, Any]) -> None:
        save_config(cfg, self.config_path)

    def __str__(self) -> str:
        return str(self.root)
