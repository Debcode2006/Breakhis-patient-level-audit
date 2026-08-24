"""Logging helpers: a console+file Python logger and a lightweight CSV metric log.

TensorBoard is handled separately (see `rap_mst/utils/experiment.py`), so these
helpers stay dependency-light and easy to reuse in scripts.

Two kinds of logger, one set of sinks
-------------------------------------
* **Script loggers** -- ``get_logger("rap_mst.train", path)`` -- own the console +
  file handlers for a run. They do not propagate, so nothing is duplicated.
* **Library loggers** -- ``logging.getLogger(__name__)`` inside ``rap_mst/*`` --
  have no handlers of their own. They propagate to the shared **package logger**
  ``rap_mst``, which :func:`get_logger` mirrors the run's sinks onto. Without that
  mirror a module-level ``logger.info(...)`` inside the package is silently
  discarded and a ``logger.warning(...)`` escapes to ``logging.lastResort``
  (bare stderr, never written to the run's log file) -- which is exactly how the
  retrieval module's bank/gate provenance lines and its ``merge_levels`` ablation
  warning went missing before this was added.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List

#: Parent of every logger inside the package; carries a mirror of the run's sinks.
PACKAGE_LOGGER = "rap_mst"

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _formatter() -> logging.Formatter:
    return logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)


def _force_utf8_stdout() -> None:
    """The reports use Unicode box-drawing / ✓ / → glyphs.

    On Windows stdout may be a legacy codepage (cp1252); force UTF-8 so those never
    raise UnicodeEncodeError mid-training. File handlers are already UTF-8. Guarded
    because a replaced/non-TextIO stdout (e.g. under pytest) has no reconfigure.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - environment dependent
        pass


def _mirror_to_package_logger(log_file: str | Path | None) -> None:
    """Give ``rap_mst.*`` library loggers the same console + file sinks.

    Idempotent: at most one console handler, and one file handler per distinct
    path, so repeated :func:`get_logger` calls never duplicate output.
    """
    pkg = logging.getLogger(PACKAGE_LOGGER)
    pkg.setLevel(logging.INFO)

    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in pkg.handlers
    )
    if not has_console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(_formatter())
        pkg.addHandler(stream)

    if log_file is not None:
        target = str(Path(log_file).resolve())
        attached = {
            str(Path(h.baseFilename).resolve())
            for h in pkg.handlers
            if isinstance(h, logging.FileHandler)
        }
        if target not in attached:
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setFormatter(_formatter())
            pkg.addHandler(handler)


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Create (or fetch) a logger that writes to stdout and optionally a file.

    Also mirrors the same sinks onto the ``rap_mst`` package logger so that
    module-level loggers inside the package (retrieval memory, future modules)
    land in the *same* console stream and the *same* run log file.
    """
    _force_utf8_stdout()
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured (idempotent)
        _mirror_to_package_logger(log_file)
        return logger

    logger.setLevel(logging.INFO)
    fmt = _formatter()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False  # the package logger below is the only other sink
    _mirror_to_package_logger(log_file)
    return logger


class CSVMetricLogger:
    """Append-only CSV logger for per-epoch metrics.

    Columns are locked on the first write. Rows are flushed immediately so a
    crashed run still leaves a readable metrics history.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: List[str] | None = None

    def log(self, row: Dict) -> None:
        write_header = not self.path.exists()
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})
