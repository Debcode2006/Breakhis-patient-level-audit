"""Configuration loading and attribute-style access.

The YAML config is the single source of truth for every experiment. We wrap the
parsed dict in a small recursive namespace so code can read ``cfg.model.backbone``
while still allowing ``cfg.to_dict()`` round-trips for saving the exact config a
run used.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Recursive, attribute-accessible view over a config dict."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data
        for key, value in data.items():
            setattr(self, key, self._wrap(value))

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep-copied plain dict (safe to mutate / serialize)."""
        out: Dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, dict):
                out[key] = Config(value).to_dict()
            else:
                out[key] = copy.deepcopy(value)
        return out

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"Config({self._data!r})"


def load_config(path: str | Path) -> Config:
    """Load a YAML file into a :class:`Config`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data)}: {path}")
    return Config(data)


def save_config(cfg: Config | Dict[str, Any], path: str | Path) -> None:
    """Persist a config (the exact one a run used) to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, Config) else cfg
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def parse_set_overrides(items) -> Dict[str, Any]:
    """Parse repeated ``--set key=value`` CLI flags into typed dotted overrides.

    Values go through the YAML loader, so ``true`` / ``0.07`` / ``[100]`` arrive as
    the types the config expects. Shared by every script that exposes ``--set``.
    """
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)  # infers int/float/bool/list/str
    return out


def merge_section(cfg: Config, section: str, source: Config) -> Config:
    """Return a copy of ``cfg`` whose ``section`` comes from ``source``.

    Used by the retrieval scripts: the model/data config must come from the run
    that produced the checkpoint (so the architecture always matches the weights),
    while a *new* section such as ``retrieval:`` -- which did not exist when those
    checkpoints were trained -- comes from the current ``config/config.yaml``.
    """
    data = cfg.to_dict()
    src = source.to_dict()
    if section not in src:
        raise KeyError(f"Config section '{section}:' not found in the source config.")
    data[section] = src[section]
    return Config(data)


def apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """Return a new Config with dotted-key overrides applied.

    Example: ``apply_overrides(cfg, {"train.fold": 2, "experiment.name": "exp2"})``.
    Used so command-line flags (e.g. ``--fold``) can override the YAML without
    editing the file.
    """
    data = cfg.to_dict()
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        keys = dotted_key.split(".")
        node = data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return Config(data)
