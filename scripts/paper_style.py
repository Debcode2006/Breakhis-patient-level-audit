"""The paper's one plotting style. Every figure imports it; nothing overrides it.

This is the executable form of ``docs/FIGURE_STYLE.md``. If a figure script
needs a font size, a colour, a line weight or a margin, it takes it from here --
so the figures read as one publication rather than as three afternoons.

The governing constraint
------------------------
The paper is ``\\documentclass[runningheads]{llncs}``: single column,
``\\textwidth = 347.12 pt = 4.803 in``. Figures are authored *at* that width and
included with ``width=\\textwidth``, so they reproduce 1:1 and the point sizes
set here are the point sizes that reach the printed page. Authoring large and
letting LaTeX scale down is what turns an 8 pt label into a 5 pt one.

For the same reason nothing here is ever saved with ``bbox_inches="tight"``: a
tight box crops each figure to its own content, the three figures land at three
different widths, and ``width=\\textwidth`` then applies three different scale
factors. Explicit margins instead -- see ``MARGIN``.

Usage
-----
    from paper_style import *
    apply_style()
    fig = new_figure(height_in=2.3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from matplotlib.patches import Rectangle            # noqa: E402

__all__ = [
    "TEXTWIDTH_IN", "TEXTWIDTH_PT", "MARGIN",
    "INK", "MUTED", "RULE", "FILL", "ACCENT", "COOL", "PAPER",
    "BINARY_COLOR", "SUBTYPE_ORDER", "SUBTYPE_COLOR", "SUBTYPE_NAME",
    "FS_PANEL", "FS_TITLE", "FS_LABEL", "FS_TICK", "FS_ANNOT", "FS_SMALL",
    "LW_AXIS", "LW_DATA", "LW_ERROR", "LW_REF", "LW_GRID",
    "MS_MAIN", "MS_CONTEXT", "MS_LEGEND", "DASH_REF",
    "apply_style", "new_figure", "despine", "reference_line", "panel_label",
    "legend_row", "dot_handle", "patch_handle", "categorical_ylim",
    "save_panel", "save_figure",
]

# --------------------------------------------------------------------------- #
# Canvas
# --------------------------------------------------------------------------- #
TEXTWIDTH_PT = 347.12354              # llncs \textwidth
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27   # = 4.8031 in

#: One set of figure-fraction margins for the whole paper, so consecutive
#: figures align with each other and with the text block. Panel content is
#: placed inside these; tick labels and axis labels live in the gutters.
MARGIN = dict(left=0.105, right=0.995, top=0.955, bottom=0.150)

# --------------------------------------------------------------------------- #
# Colour -- Okabe-Ito derived, so every pairing survives deuteranopia,
# protanopia and a greyscale print run.
# --------------------------------------------------------------------------- #
INK = "#16181D"      # text; data marks that carry no category
MUTED = "#5C626B"    # ticks, secondary annotation, reference labels
RULE = "#C8CDD4"     # hairlines, grid, frames
FILL = "#E9EDF1"     # quantitative area fills (bars, bands)
PAPER = "#FFFFFF"

#: THE condition under test. One meaning per figure, and as far as possible one
#: meaning across the paper. Everything that is not the point is grey.
ACCENT = "#D55E00"   # vermillion
COOL = "#0072B2"     # blue -- the contrasting category

#: Binary class colouring, shared with ``scripts/visualize_embeddings.py``.
BINARY_COLOR = {0: COOL, 1: ACCENT}

#: Eight tumour subtypes as a *nested* palette: benign on a cool ramp, malignant
#: on a warm one, so the binary structure stays legible in the subtype panel
#: without that panel becoming a recolouring of the binary panel. MC and PC --
#: the two the argument is about, and the two that were near-identical in the
#: previous palette -- are given the widest separation in the warm ramp.
SUBTYPE_ORDER = ["A", "F", "TA", "PT", "DC", "LC", "MC", "PC"]
SUBTYPE_COLOR = {
    "A":  "#8FD3F4",   # adenosis          pale cyan
    "F":  "#0072B2",   # fibroadenoma      blue
    "TA": "#009E73",   # tubular adenoma   green
    "PT": "#1F3B73",   # phyllodes         deep navy
    "DC": "#E8A33D",   # ductal ca.        amber   (the ~55% mass: kept light)
    "LC": "#8C2D04",   # lobular ca.       dark rust
    "MC": "#CC79A7",   # mucinous ca.      pink
    "PC": "#6A2C8F",   # papillary ca.     violet
}
SUBTYPE_NAME = {
    "A": "adenosis", "F": "fibroadenoma", "TA": "tubular adenoma",
    "PT": "phyllodes", "DC": "ductal ca.", "LC": "lobular ca.",
    "MC": "mucinous ca.", "PC": "papillary ca.",
}

# --------------------------------------------------------------------------- #
# Type -- absolute points on the printed page (no rescaling ever happens).
# LNCS sets the caption at 9 pt; nothing in a figure may be louder than that.
# 6.5 pt is the floor: Nature's minimum is 5 pt at final size and we have no
# scale-up headroom to spend.
# --------------------------------------------------------------------------- #
FS_PANEL = 9.0    # (a) / (b) -- the only bold type in the paper's figures
FS_TITLE = 8.0    # nominal panel title
FS_LABEL = 8.0    # axis label
FS_TICK = 7.0     # tick label
FS_ANNOT = 7.0    # in-panel annotation, legend
FS_SMALL = 6.5    # secondary reference note -- the floor

# --------------------------------------------------------------------------- #
# Marks and rules
# --------------------------------------------------------------------------- #
LW_AXIS = 0.75    # publisher floor for a rendered line
LW_DATA = 1.20
LW_ERROR = 1.00
LW_REF = 0.80
LW_GRID = 0.50

MS_MAIN = 4.2     # emphasised marker
MS_CONTEXT = 3.4  # context marker
MS_LEGEND = 3.2

DASH_REF = (0, (3.6, 2.2))   # the one dash pattern in the paper

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    # Bind mathtext to the *same* family. matplotlib otherwise sets $p_{final}$
    # in DejaVu next to an Arial label -- two typefaces inside one tick label,
    # which is the clearest tell of an undesigned figure.
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "it",

    "font.size": FS_LABEL,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_ANNOT,

    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.edgecolor": MUTED,
    "axes.linewidth": LW_AXIS,
    "axes.facecolor": PAPER,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.axisbelow": True,

    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
    "xtick.major.width": LW_AXIS, "ytick.major.width": LW_AXIS,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.major.pad": 2.2, "ytick.major.pad": 2.2,
    "xtick.direction": "out", "ytick.direction": "out",

    "grid.color": RULE, "grid.linewidth": LW_GRID,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    "legend.borderpad": 0.0,
    "legend.labelspacing": 0.35,
    "legend.handlelength": 1.4,

    "lines.linewidth": LW_DATA,
    "lines.solid_capstyle": "round",
    "patch.linewidth": 0.0,

    "figure.dpi": 200,
    "savefig.dpi": 600,
    # TrueType, not Type-3. Springer asks for it and Type-3 breaks text search.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",   # keep SVG text as text, so it stays editable
}


def apply_style() -> None:
    """Install the house style. Call once, before any figure is created."""
    plt.rcParams.update(_RC)


# --------------------------------------------------------------------------- #
# Construction helpers
# --------------------------------------------------------------------------- #
def new_figure(height_in: float, width_in: float = TEXTWIDTH_IN):
    """A canvas exactly ``\\textwidth`` wide (or a stated fraction of it)."""
    return plt.figure(figsize=(width_in, height_in))


def despine(ax, keep: Iterable[str] = ("left", "bottom")) -> None:
    """Top and right spines go everywhere in this paper."""
    keep = set(keep)
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def reference_line(ax, value: float, axis: str = "y", color: str = RULE,
                   zorder: float = 1.0):
    """A level to read against -- dashed, hairline, always below the data."""
    fn = ax.axhline if axis == "y" else ax.axvline
    return fn(value, color=color, lw=LW_REF, ls=DASH_REF, zorder=zorder)


def categorical_ylim(ax, n: int) -> None:
    """One row-spacing rule for every categorical/forest panel in the paper."""
    ax.set_ylim(-0.62, n - 0.38)


def dot_handle(color: str, label: str, ms: float = MS_LEGEND) -> Line2D:
    return Line2D([0], [0], marker="o", linestyle="", markersize=ms,
                  markerfacecolor=color, markeredgewidth=0, label=label)


def patch_handle(color: str, label: str) -> Rectangle:
    return Rectangle((0, 0), 1, 1, fc=color, ec="none", label=label)


def legend_row(target, handles: Sequence, y: float, ncol: int | None = None,
               x: float = 0.5, fontsize: float = FS_ANNOT, **kw):
    """A frameless single-row legend, placed outside the data region."""
    return target.legend(handles=list(handles), loc="lower center",
                         bbox_to_anchor=(x, y), ncol=ncol or len(handles),
                         frameon=False, fontsize=fontsize,
                         handletextpad=0.35, columnspacing=1.1,
                         borderaxespad=0.0, **kw)


def panel_label(fig, text: str, x: float, y: float) -> None:
    """``(a)`` / ``(b)``, in *figure* coordinates.

    Deliberately not axes coordinates: an axes-relative offset puts the label a
    different distance from each panel depending on how wide that panel's tick
    labels are, which is how the previous figure set ended up with x = -0.02,
    -0.19 and -0.30 for the same mark. Added only at assembly -- a raw panel
    carries no label, so it can be reordered or promoted without an edit.
    """
    fig.text(x, y, text, fontsize=FS_PANEL, fontweight="bold", color=INK,
             ha="left", va="top")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _save(fig, out_dir: Path, stem: str, formats: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, format=fmt)          # no bbox_inches -- see module docstring
        print(f"  wrote {p}")
    plt.close(fig)


def save_panel(fig, out_dir: Path, stem: str,
               formats: Sequence[str] = ("svg", "pdf")) -> None:
    """Write one raw panel. Vector only; a panel is never a preview."""
    _save(fig, out_dir, stem, formats)


def save_figure(fig, out_dir: Path, stem: str,
                formats: Sequence[str] = ("pdf", "png")) -> None:
    """Write one assembled figure. PDF for LaTeX, PNG to look at."""
    _save(fig, out_dir, stem, formats)
