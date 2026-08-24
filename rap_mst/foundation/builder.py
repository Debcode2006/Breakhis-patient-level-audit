"""Config -> foundation-baseline objects.

Same role ``rap_mst/retrieval/builder.py`` plays for the memory module: every
lookup into the ``foundation:`` block goes through here, so a renamed key breaks in
one place instead of five.
"""

from __future__ import annotations

from pathlib import Path


def foundation_cfg(cfg):
    """The ``foundation:`` block, or a clear error naming what to add."""
    fcfg = getattr(cfg, "foundation", None)
    if fcfg is None:
        raise SystemExit(
            "No `foundation:` block in the config. Add it to config/config.yaml "
            "(see the block documented there) before running the baseline."
        )
    return fcfg


def encoder_spec(cfg, name: str):
    """The ``foundation.encoders.<name>`` sub-block."""
    encoders = getattr(foundation_cfg(cfg), "encoders", None)
    spec = getattr(encoders, name, None) if encoders is not None else None
    if spec is None:
        raise SystemExit(
            f"No `foundation.encoders.{name}:` block in the config. "
            "Declare the hub id / feature dim there."
        )
    return spec


def probe_cfg(cfg):
    pcfg = getattr(foundation_cfg(cfg), "probe", None)
    if pcfg is None:
        raise SystemExit("No `foundation.probe:` block in the config.")
    return pcfg


def resolve_cache_path(cfg, encoder: str | None = None) -> Path:
    """``foundation.cache_path`` with ``{encoder}`` filled in."""
    fcfg = foundation_cfg(cfg)
    encoder = (encoder or str(fcfg.encoder)).strip().lower()
    template = str(getattr(fcfg, "cache_path", "analysis/foundation/{encoder}/features.npz"))
    return Path(template.format(encoder=encoder))


def build_probe_head(cfg, in_dim: int, num_classes: int = 2):
    """Build the probe head described by ``foundation.probe`` (linear | mlp)."""
    from rap_mst.foundation.probe import ProbeHead

    pcfg = probe_cfg(cfg)
    return ProbeHead(
        in_dim=in_dim,
        num_classes=num_classes,
        head=str(getattr(pcfg, "head", "linear")),
        hidden_dim=int(getattr(pcfg, "hidden_dim", 512)),
        dropout=float(getattr(pcfg, "dropout", 0.0)),
    )
