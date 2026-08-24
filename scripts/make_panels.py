"""Render the six raw panels of Figures 3-5, one panel per canvas.

Each panel is written as SVG *and* PDF and is publication-ready on its own:
its own margins, its own legend where it needs one, its own axis labels. What a
raw panel does **not** carry is a ``(a)``/``(b)`` label -- that is a statement
about position within a figure, and it is added by ``assemble_figures.py``.

The panel boxes are sized in inches from ``figure_panels.GEOM``, the same table
the assembler uses, so a raw panel and its assembled copy contain axes of
identical size and identical type. Nothing is rescaled at any point.

    python scripts/make_panels.py                  # all six
    python scripts/make_panels.py --panels 3a 5b
    python scripts/make_panels.py --out images/panels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paper_style import (FS_ANNOT, FS_SMALL, apply_style, legend_row,  # noqa: E402
                         new_figure, save_panel)
import figure_panels as P                                             # noqa: E402


def _canvas(name: str):
    """A canvas exactly the size of the panel box, with the panel on it."""
    g = P.GEOM[name]
    fig = new_figure(width_in=g["w"], height_in=g["h"])
    return fig, P.cell(fig, 0.0, 0.0, g["w"], g["h"], g["pad"])


def panel_3a(out: Path, fmts):
    fig, spec = _canvas("fig3a")
    P.draw_fig3a(fig, spec)
    save_panel(fig, out, "fig3a", fmts)


def panel_3b(out: Path, fmts):
    fig, spec = _canvas("fig3b")
    P.draw_fig3b(fig, spec)
    save_panel(fig, out, "fig3b", fmts)


def _panel_fig4(out: Path, fmts, mode: str, stem: str):
    fig, spec = _canvas("fig4a" if mode == "binary" else "fig4b")
    P.draw_fig4(fig, spec, mode)
    # The key sits below the row, frameless and in one line -- eight subtype
    # swatches wrap into two ragged rows only if the codes are spelled out, and
    # the caption already carries the expansion.
    legend_row(fig, P.fig4_handles(mode), y=0.020,
               fontsize=FS_ANNOT if mode == "binary" else FS_SMALL)
    save_panel(fig, out, stem, fmts)


def panel_4a(out: Path, fmts):
    _panel_fig4(out, fmts, "binary", "fig4a")


def panel_4b(out: Path, fmts):
    _panel_fig4(out, fmts, "subtype", "fig4b")


def panel_5a(out: Path, fmts):
    fig, spec = _canvas("fig5a")
    P.draw_fig5a(fig, spec)
    save_panel(fig, out, "fig5a", fmts)


def panel_5b(out: Path, fmts, resamples: int, seed: int):
    fig, spec = _canvas("fig5b")
    P.draw_fig5b(fig, spec, resamples=resamples, seed=seed)
    save_panel(fig, out, "fig5b", fmts)


PANELS = {"3a": panel_3a, "3b": panel_3b, "4a": panel_4a,
          "4b": panel_4b, "5a": panel_5a}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panels", nargs="+", default=["3a", "3b", "4a", "4b", "5a", "5b"])
    ap.add_argument("--out", default="images/panels")
    ap.add_argument("--formats", nargs="+", default=["svg", "pdf"])
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    apply_style()
    out = ROOT / args.out
    for name in args.panels:
        print(f"panel {name}")
        if name == "5b":
            panel_5b(out, args.formats, args.resamples, args.seed)
        else:
            PANELS[name](out, args.formats)


if __name__ == "__main__":
    main()
