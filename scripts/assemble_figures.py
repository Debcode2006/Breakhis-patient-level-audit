"""Assemble the raw panels into Figure 3, Figure 4 and Figure 5.

This is the only stage that knows a panel's position in a figure, and therefore
the only stage that adds ``(a)`` / ``(b)``.

It does not paste the rendered panel files together. It calls the same drawing
functions ``make_panels.py`` calls, into cells laid out by the same
``figure_panels.GEOM`` table, measured in inches. A panel's *axes* therefore
occupy exactly as many inches here as they do standalone -- same type sizes,
same line weights, same marker sizes, nothing rescaled -- while the surrounding
canvas differs. Compositing the PDFs instead would have been equivalent only as
long as nobody ever touched a margin.

Every figure comes out exactly ``\\textwidth`` (4.803 in) wide, for
``\\includegraphics[width=\\textwidth]`` with no scaling.

    python scripts/assemble_figures.py
    python scripts/assemble_figures.py --figures 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paper_style import (FS_ANNOT, FS_SMALL, TEXTWIDTH_IN, apply_style,  # noqa: E402
                         legend_row, new_figure, panel_label, save_figure)
import figure_panels as P                                              # noqa: E402

#: The panel label sits this far to the left of, and this far above, the
#: panel's *axes* -- not its box. Panels differ in how much room their tick
#: labels need, so anchoring to the box would put the label a different
#: distance from every panel.
LABEL_DX, LABEL_DY = 0.30, 0.03
#: Room above the tallest panel for the label band.
LABEL_BAND = 0.17


def _label(fig, text: str, x0: float, pad_left: float, top_in: float) -> None:
    W, H = fig.get_size_inches()
    panel_label(fig, text, x=max(x0, x0 + pad_left - LABEL_DX) / W,
                y=(top_in + LABEL_DY + 0.115) / H)


# --------------------------------------------------------------------------- #
def figure3(out: Path, formats):
    """(a) the per-magnification logit offsets; (b) AUC under five head inputs."""
    a, b = P.GEOM["fig3a"], P.GEOM["fig3b"]
    h = a["h"] + LABEL_BAND
    fig = new_figure(width_in=TEXTWIDTH_IN, height_in=h)

    P.draw_fig3a(fig, P.cell(fig, 0.0, 0.0, a["w"], a["h"], a["pad"]))
    P.draw_fig3b(fig, P.cell(fig, a["w"], 0.0, b["w"], b["h"], b["pad"]))

    _label(fig, "(a)", 0.0, a["pad"][0], a["h"])
    _label(fig, "(b)", a["w"], b["pad"][0], b["h"])
    save_figure(fig, out, "Figure3", formats)


def figure4(out: Path, formats):
    """(a) coloured by class; (b) by tumour subtype -- the same projection.

    The two rows do not share a key, so they do not share a legend: each sits
    directly under the row it belongs to. Only the top row carries the column
    titles; repeating them over vertically aligned columns is redundant ink.
    """
    g = P.GEOM["fig4a"]
    pad_a, pad_b = g["pad"], P.FIG4B_ASSEMBLED_PAD
    h_a = g["h"]
    h_b = g["h"] - pad_a[2] + pad_b[2]
    H = h_a + h_b
    fig = new_figure(width_in=TEXTWIDTH_IN, height_in=H)

    P.draw_fig4(fig, P.cell(fig, 0.0, h_b, g["w"], h_a, pad_a),
                "binary", titles=True)
    P.draw_fig4(fig, P.cell(fig, 0.0, 0.0, g["w"], h_b, pad_b),
                "subtype", titles=False)

    legend_row(fig, P.fig4_handles("binary"), y=(h_b + 0.030) / H,
               fontsize=FS_ANNOT)
    legend_row(fig, P.fig4_handles("subtype"), y=0.030 / H, fontsize=FS_SMALL)

    # Figure 4's panels run the full text width, so there is no left gutter for
    # a label; it goes in the band each row already reserves above its axes.
    panel_label(fig, "(a)", x=0.02 / TEXTWIDTH_IN, y=(h_b + h_a - 0.02) / H)
    panel_label(fig, "(b)", x=0.02 / TEXTWIDTH_IN, y=(h_b - 0.02) / H)
    save_figure(fig, out, "Figure4", formats)


def figure5(out: Path, formats, resamples: int, seed: int):
    """(a) the three-dose mechanism; (b) five streams with clustered CIs."""
    a, b = P.GEOM["fig5a"], P.GEOM["fig5b"]
    h = a["h"] + LABEL_BAND
    fig = new_figure(width_in=TEXTWIDTH_IN, height_in=h)

    P.draw_fig5a(fig, P.cell(fig, 0.0, 0.0, a["w"], a["h"], a["pad"]))
    P.draw_fig5b(fig, P.cell(fig, a["w"], 0.0, b["w"], b["h"], b["pad"]),
                 resamples=resamples, seed=seed)

    _label(fig, "(a)", 0.0, a["pad"][0], a["h"])
    _label(fig, "(b)", a["w"], b["pad"][0], b["h"])
    save_figure(fig, out, "Figure5", formats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--figures", nargs="+", default=["3", "4", "5"],
                    choices=["3", "4", "5"])
    ap.add_argument("--out", default="images")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    apply_style()
    out = ROOT / args.out
    if "3" in args.figures:
        print("Figure 3"); figure3(out, args.formats)
    if "4" in args.figures:
        print("Figure 4"); figure4(out, args.formats)
    if "5" in args.figures:
        print("Figure 5"); figure5(out, args.formats, args.resamples, args.seed)


if __name__ == "__main__":
    main()
