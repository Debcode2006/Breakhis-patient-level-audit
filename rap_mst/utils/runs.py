"""Locating trained runs under ``runs/<experiment>/<timestamp>_train_fold<k>/``.

The three analysis scripts written before the retrieval module each carried their
own copy of this logic; new code imports it from here instead. Behaviour is
identical to those copies (latest run per fold, ``_``-anchored prefix fallback so
``exp3`` never resolves to ``exp3n_...``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

from rap_mst.experiments import EXPERIMENT_DIRS

_FOLD_RE = re.compile(r"_train_fold(\d+)$")


def find_runs(runs_root: Path) -> Dict[int, Path]:
    """Map fold index -> the latest run dir under ``runs_root`` that has a best.pt."""
    found: Dict[int, Path] = {}
    for d in sorted(Path(runs_root).glob("*_train_fold*")):
        m = _FOLD_RE.search(d.name)
        if not m or not (d / "checkpoints" / "best.pt").exists():
            continue
        found[int(m.group(1))] = d  # sorted() is chronological -> keep the latest
    return found


def resolve_runs_root(experiment: str, log_root: Path) -> Path:
    """Resolve ``runs/<experiment>``, tolerating short names (``exp3n``)."""
    log_root = Path(log_root)
    if experiment in EXPERIMENT_DIRS:
        cand = log_root / EXPERIMENT_DIRS[experiment]
        if cand.exists():
            return cand
    direct = log_root / experiment
    if direct.exists():
        return direct
    matches = sorted(log_root.glob(f"{experiment}_*"))
    if matches:
        return matches[0]
    raise SystemExit(f"Runs dir not found for experiment '{experiment}' under {log_root}")


def find_fold_run(experiment: str, log_root: Path, fold: int) -> Path:
    """The run directory holding fold ``fold``'s ``checkpoints/best.pt``."""
    runs = find_runs(resolve_runs_root(experiment, log_root))
    if fold not in runs:
        raise SystemExit(
            f"No completed run with checkpoints/best.pt for experiment '{experiment}' "
            f"fold {fold} (found folds: {sorted(runs)}). Train it first."
        )
    return runs[fold]


def available_folds(experiment: str, log_root: Path) -> list:
    return sorted(find_runs(resolve_runs_root(experiment, log_root)))


def checkpoint_path(run_dir: Path, which: str = "best") -> Optional[Path]:
    p = Path(run_dir) / "checkpoints" / f"{which}.pt"
    return p if p.exists() else None
