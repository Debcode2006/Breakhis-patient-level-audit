"""The six raw panels of Figures 3-5, as drawing functions.

Why drawing functions and not scripts
-------------------------------------
Each panel is rendered twice: once standalone (``make_panels.py``) and once
inside its assembled figure (``assemble_figures.py``). Both call the *same*
function here, so a panel and its assembled copy cannot drift. The functions
draw content only -- no ``(a)``/``(b)`` label, no figure-level title. Panel
labels are added by the assembler, which is the only place that knows the
panel's position in a figure.

Every function has the signature ``draw(fig, spec, ...)`` where ``spec`` is a
``SubplotSpec`` sized by ``cell()``. That is what makes the standalone and
assembled renderings pixel-identical: the *axes* occupy the same number of
inches in both, only the surrounding canvas differs.

Data provenance -- nothing here re-derives a number:
    fig3a/3b  analysis/retrieval_probe/probe4.json          (exp3, 5-fold means)
    fig4a/4b  analysis/embeddings/fig4_umap_cache.npz       (precompute_umap_cache.py)
    fig5a     docs/results/crossencoder_screen.md 4.1/5.1                (transcribed, see DOSE)
    fig5b     analysis/retrieval_crossencoder/{crossencoder_screen.json,
                                               probs_primary.npz}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib.transforms import blended_transform_factory

from paper_style import (ACCENT, BINARY_COLOR, COOL, DASH_REF, FILL, FS_ANNOT,
                         FS_LABEL, FS_SMALL, FS_TICK, FS_TITLE, INK, LW_AXIS,
                         LW_DATA, LW_ERROR, LW_REF, MS_CONTEXT, MS_MAIN, MUTED,
                         RULE, SUBTYPE_COLOR, SUBTYPE_ORDER, categorical_ylim,
                         despine, dot_handle, patch_handle, reference_line)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def cell(fig, x0: float, y0: float, w: float, h: float,
         pad: tuple[float, float, float, float]):
    """A panel box placed in *inches* on ``fig``, returning its SubplotSpec.

    ``x0, y0`` is the lower-left corner of the panel box and ``w, h`` its size;
    ``pad = (left, right, top, bottom)`` is the panel's own decoration margin --
    the room its tick labels, axis labels and annotation gutters need.

    Working in inches rather than figure fractions is what lets a panel be drawn
    standalone on a small canvas and again inside a wide figure with the *axes*
    coming out exactly the same size. Fractions would rescale with the canvas.
    """
    W, H = fig.get_size_inches()
    left, right, top, bottom = pad
    gs = fig.add_gridspec(1, 1,
                          left=(x0 + left) / W, right=(x0 + w - right) / W,
                          bottom=(y0 + bottom) / H, top=(y0 + h - top) / H)
    return gs[0, 0]


#: Panel box sizes and decoration pads, in inches. One table, so the standalone
#: panels and the assembled figures are laid out from the same numbers.
#: ``pad`` is (left, right, top, bottom).
GEOM = {
    "fig3a": dict(w=2.243, h=2.18, pad=(0.46, 0.05, 0.08, 0.42)),
    "fig3b": dict(w=2.560, h=2.18, pad=(0.66, 0.50, 0.08, 0.42)),
    "fig4a": dict(w=4.803, h=1.78, pad=(0.05, 0.05, 0.34, 0.39)),
    "fig4b": dict(w=4.803, h=1.78, pad=(0.05, 0.05, 0.34, 0.39)),
    "fig5a": dict(w=2.100, h=2.28, pad=(0.50, 0.05, 0.06, 0.62)),
    "fig5b": dict(w=2.703, h=2.28, pad=(0.56, 0.98, 0.08, 0.42)),
}
#: Figure 4's rows are stacked, so only the top row carries column titles; the
#: bottom row's box is shorter by exactly the title band it does not draw.
FIG4B_ASSEMBLED_PAD = (0.05, 0.05, 0.06, 0.39)


# =========================================================================== #
# Figure 3 -- the magnification audit
# =========================================================================== #
MAGS = ["40x", "100x", "200x", "400x"]
MAG_LABEL = ["40×", "100×", "200×", "400×"]


def load_fig3(probe4: Path = ROOT / "analysis/retrieval_probe/probe4.json") -> dict:
    return json.load(probe4.open(encoding="utf-8"))["exp3"]["A2_counterfactual"]


def draw_fig3a(fig, spec, audit: dict | None = None) -> None:
    """Per-magnification logit offset, against the image-driven logit s.d.

    The claim has two halves that live at scales fourteen times apart -- the
    offsets are *monotone in zoom* (span 0.61) and they are *small* (against an
    image-driven spread of +/-4.22). One axis cannot show both, so the panel
    uses the zoom-inset idiom properly: the magnified band is marked on the
    parent axes and joined to the inset by two leaders, rather than a second
    plot being parked on top of the first.
    """
    audit = audit or load_fig3()
    offsets = [audit["logit_offset_per_mag"][m] for m in MAGS]
    sd = audit["feature_logit_std"]
    x = np.arange(4)

    ax = fig.add_subplot(spec)
    ax.set_xlim(-0.62, 3.62)
    ax.set_ylim(-sd * 1.19, sd * 1.19)

    # The reference envelope: a bounded light region, never a solid slab that
    # outweighs the data it is there to put in proportion.
    ax.axhspan(-sd, sd, color=FILL, zorder=0, lw=0)
    for s in (-1, 1):
        reference_line(ax, s * sd, color=RULE, zorder=1)
    ax.axhline(0, color=RULE, lw=LW_AXIS, zorder=2)

    ax.bar(x, offsets, width=0.58, color=ACCENT, zorder=3, lw=0)
    ax.annotate(f"image-driven logit s.d.  ±{sd:.2f}", xy=(3.55, -sd),
                xytext=(0, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=FS_SMALL, color=MUTED)

    ax.set_xticks(x, MAG_LABEL)
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_xlabel("magnification")
    ax.set_ylabel("logit offset  (malignant − benign)")
    despine(ax)

    # The inset: same four bars at their own scale. It carries no text of its
    # own -- its two ticks are the scale, and its x categories are the parent's.
    ins = ax.inset_axes((0.045, 0.575, 0.86, 0.355))
    ins.set_xlim(-0.62, 3.62)
    ins.set_ylim(-0.42, 0.42)
    ins.axhline(0, color=RULE, lw=LW_AXIS, zorder=1)
    ins.bar(x, offsets, width=0.58, color=ACCENT, zorder=2, lw=0)
    ins.set_xticks([])
    ins.set_yticks([-0.3, 0.0, 0.3])
    ins.set_yticklabels(["−0.3", "0", "0.3"], fontsize=FS_SMALL)
    # Ticks on the right: on the left they land on top of the parent's own y
    # tick labels, which sit only a few points away.
    ins.yaxis.set_ticks_position("right")
    ins.tick_params(length=1.8, pad=1.4, width=LW_AXIS)
    ins.set_facecolor("white")
    # A hairline frame all round: the leaders have to land on something, and
    # the inset sits on the shaded band, where a white ground alone reads as a
    # hole rather than as a panel.
    for side, sp in ins.spines.items():
        sp.set(visible=True, color=MUTED if side == "right" else RULE,
               linewidth=LW_AXIS)

    # Marked source region + leaders: this is what makes it an inset rather
    # than a second plot sitting on the first.
    rect, leaders = ax.indicate_inset_zoom(ins, edgecolor=MUTED, alpha=1.0)
    rect.set(linewidth=0.6, facecolor="none")
    for ln in leaders:
        ln.set(color=RULE, linewidth=0.6, linestyle="-")


def draw_fig3b(fig, spec, audit: dict | None = None) -> None:
    """Image AUC of the same frozen weights under five head inputs.

    Four of the five rows are identical to five decimal places, so the panel is
    built to make that *coincidence* legible: the reference line sits exactly on
    a labelled tick, and the four counterfactual markers stack on it. Only the
    deployed row leaves the line, and the length it leaves by is the result.
    """
    audit = audit or load_fig3()
    forced = audit["forced_magnification"]
    rows: list[tuple[str, float]] = [
        ("true mag.", audit["auc_true_mag"]),
        ("block zeroed", audit["auc_block_zeroed"]),
        ("block = mean", audit["auc_block_table_mean"]),
        ("forced 40×", forced["forced_40x"]["auc"]),
        ("forced 400×", forced["forced_400x"]["auc"]),
    ]
    deployed, base = rows[0][1], rows[1][1]
    y = np.arange(len(rows))[::-1]

    ax = fig.add_subplot(spec)
    ax.set_xlim(0.960435, 0.960755)
    ax.set_xticks([0.96050, 0.96060, 0.96070],
                  ["0.96050", "0.96060", "0.96070"])
    categorical_ylim(ax, len(rows))
    ax.xaxis.grid(True, color=RULE, lw=LW_AXIS * 0.7)

    reference_line(ax, base, axis="x", color=MUTED, zorder=2)
    for yi, (_, v) in zip(y, rows):
        top = yi == y[0]
        if top:
            ax.plot([base, v], [yi, yi], color=ACCENT, lw=LW_DATA, zorder=3)
        ax.plot([v], [yi], marker="o", ms=MS_MAIN if top else MS_CONTEXT,
                color=ACCENT if top else MUTED, markeredgecolor="white",
                markeredgewidth=0.6, zorder=4)

    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("image AUC  (5-fold mean, test)")
    despine(ax, keep=("bottom",))

    ax.annotate(f"+{deployed - base:.5f}", xy=(deployed, y[0]), xytext=(5, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=FS_ANNOT, color=INK)
    ax.annotate("four counterfactuals", xy=(base, y[0] + 0.44), xytext=(4, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=FS_SMALL, color=MUTED)


# =========================================================================== #
# Figure 4 -- UMAP of the held-out test set, four configurations
# =========================================================================== #
LADDER = ["exp1", "exp2", "exp3", "exp3n"]
#: Two-tier column titles: the configuration, then what it adds. A single-line
#: title ("Exp3 (+ mag + SupCon)") overruns a 1.2 in column at 8 pt.
LADDER_TITLE = {"exp1": ("Exp1", "baseline"), "exp2": ("Exp2", "+ mag"),
                "exp3": ("Exp3", "+ mag + SupCon"), "exp3n": ("Exp3n", "+ SupCon")}


def load_fig4(cache: Path = ROOT / "analysis/embeddings/fig4_umap_cache.npz") -> dict:
    if not cache.exists():
        raise SystemExit(f"{cache} missing -- run scripts/precompute_umap_cache.py")
    return dict(np.load(cache, allow_pickle=True))


def _umap_axes(ax, fig, coords: np.ndarray) -> None:
    """A projection has no scale, so it gets no frame, no ticks and no axis.

    The window is centred on the cloud and given *the axes box's own aspect
    ratio*, sized by the cloud's larger extent. Equal x and y scaling then
    holds by construction, so no aspect machinery runs and the axes box never
    moves. That last part is what matters: ``set_aspect("equal")`` defaults to
    ``adjustable="box"``, which satisfies the aspect by silently resizing the
    axes -- and every annotation placed in axes coordinates goes with it, which
    is how four column titles end up at four different heights.
    """
    ax.set_xticks([]); ax.set_yticks([])
    despine(ax, keep=())
    box = ax.get_position()
    fw, fh = fig.get_size_inches()
    aspect = (box.width * fw) / (box.height * fh)

    x, y = coords[:, 0], coords[:, 1]
    half_y = 0.53 * max(x.max() - x.min(), y.max() - y.min()) / max(aspect, 1.0)
    half_x = half_y * aspect
    cx, cy = 0.5 * (x.max() + x.min()), 0.5 * (y.max() + y.min())
    ax.set_xlim(cx - half_x, cx + half_x)
    ax.set_ylim(cy - half_y, cy + half_y)


def draw_fig4(fig, spec, mode: str, data: dict | None = None,
              titles: bool = True) -> None:
    """One row of the ladder's UMAPs: ``mode`` is ``"binary"`` or ``"subtype"``.

    Both rows plot the *same* projection -- fitted once by
    ``precompute_umap_cache.py`` -- so a point can be followed from row to row.
    The per-panel separation metrics are computed on the 1024-d vectors and are
    set below the cloud at reference weight: they annotate the panel, they are
    not its result.
    """
    data = data or load_fig4()
    grid = spec.subgridspec(1, 4, wspace=0.055)

    for c, exp in enumerate(LADDER):
        ax = fig.add_subplot(grid[0, c])
        coords = data[f"{exp}/coords"]
        ax.set_rasterization_zorder(2)      # points raster, everything else vector

        if mode == "binary":
            lab = data[f"{exp}/label"]
            for cls in (0, 1):
                m = lab == cls
                ax.scatter(coords[m, 0], coords[m, 1], s=1.6, alpha=0.50,
                           c=BINARY_COLOR[cls], linewidths=0, zorder=1)
            note = (f"sil {float(data[f'{exp}/sil']):.2f}"
                    f"   kNN {float(data[f'{exp}/knn']):.2f}")
        else:
            sub = data[f"{exp}/subtype"]
            for st in SUBTYPE_ORDER:        # fixed order: colours never shuffle
                m = sub == st
                if m.any():
                    ax.scatter(coords[m, 0], coords[m, 1], s=1.6, alpha=0.55,
                               c=SUBTYPE_COLOR[st], linewidths=0, zorder=1)
            note = (f"sil$_{{sub}}$ "
                    f"{float(data[f'{exp}/sil_sub']):.2f}".replace("-", "−"))

        _umap_axes(ax, fig, coords)
        ax.annotate(note, xy=(0.5, -0.035), xycoords="axes fraction",
                    ha="center", va="top", fontsize=FS_SMALL, color=MUTED)
        if titles:
            head, tail = LADDER_TITLE[exp]
            ax.annotate(head, xy=(0.5, 1.105), xycoords="axes fraction",
                        ha="center", va="bottom", fontsize=FS_TITLE, color=INK)
            ax.annotate(tail, xy=(0.5, 1.020), xycoords="axes fraction",
                        ha="center", va="bottom", fontsize=FS_SMALL, color=MUTED)


def fig4_handles(mode: str) -> list:
    if mode == "binary":
        return [dot_handle(BINARY_COLOR[0], "benign"),
                dot_handle(BINARY_COLOR[1], "malignant")]
    return [dot_handle(SUBTYPE_COLOR[s], s) for s in SUBTYPE_ORDER]


# =========================================================================== #
# Figure 5 -- the retrieval boundary condition
# =========================================================================== #
#: docs/results/crossencoder_screen.md 5.1 (a_wrong and its range over every key configuration
#: measured at that dose: 39 same-encoder, 4 exp1, 3 CTransPath) and 4.1 (the
#: gate's mean weight on the memory, and the correlation between the two
#: probability streams, for the matched `features` key at each dose).
DOSE = [
    # label,          sublabel,       a_wrong, (lo, hi),         gate w, corr
    ("exp3n",       "shared",          0.070, (0.051, 0.100),    0.147, 0.948),
    ("exp1",        "other Swin",      0.131, (0.127, 0.136),    0.132, 0.908),
    ("CTransPath",  "foundation",      0.568, (0.568, 0.739),    0.404, 0.728),
]
PREREG_BAR = 0.136      # the threshold a_wrong was locked to clear

#: Row labels are the streams' plain names; the caption maps them to
#: p_param / p_img / p_final / p_probe / p_ens. Symbols in the tick column cost
#: 0.3 in of panel width and read no faster.
STREAMS = [
    ("prob_param",      "head",        False),
    ("prob_img",        "image kNN",   False),
    ("prob_final_loso", "gated",       True),
    ("prob_probe",      "probe",       False),
    ("prob_ens_half",   "½ average",   False),
]


def load_fig5(screen: Path = ROOT / "analysis/retrieval_crossencoder/crossencoder_screen.json",
              probs: Path = ROOT / "analysis/retrieval_crossencoder/probs_primary.npz") -> tuple:
    return (json.load(screen.open(encoding="utf-8")),
            np.load(probs, allow_pickle=True))


def draw_fig5a(fig, spec) -> None:
    """The three-dose response of the memory's error decorrelation.

    All three quantities are in [0, 1] and all three are monotone across the
    doses, so all three are *plotted*. The previous version typeset two of them
    as rows of text under the tick labels, which asked the reader to
    reconstruct the panel's own mechanism from six numbers.
    """
    ax = fig.add_subplot(spec)
    x = np.arange(len(DOSE))
    aw = [r[2] for r in DOSE]
    lo = [r[2] - r[3][0] for r in DOSE]
    hi = [r[3][1] - r[2] for r in DOSE]

    ax.bar(x, [r[4] for r in DOSE], width=0.60, color=FILL, zorder=0, lw=0)
    reference_line(ax, PREREG_BAR, color=RULE, zorder=1)
    ax.errorbar(x, aw, yerr=[lo, hi], color=ACCENT, lw=LW_DATA, marker="o",
                ms=MS_MAIN, capsize=2.0, elinewidth=LW_ERROR,
                markeredgecolor="white", markeredgewidth=0.6, zorder=2)
    # Drawn last so its white marker face masks the a_wrong whisker it crosses
    # at the third dose (upper range 0.739 against a correlation of 0.728).
    ax.plot(x, [r[5] for r in DOSE], color=MUTED, lw=LW_ERROR, ls=DASH_REF,
            marker="o", ms=MS_CONTEXT, markerfacecolor="white",
            markeredgecolor=MUTED, markeredgewidth=0.9, zorder=3)

    ax.set_xlim(-0.62, len(DOSE) - 0.38)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x, [f"{r[0]}\n{r[1]}" for r in DOSE])
    ax.tick_params(axis="x", length=0, pad=3, labelsize=FS_SMALL)
    ax.set_ylabel("value in [0, 1]")
    ax.set_xlabel("encoder that writes the memory", labelpad=4)
    despine(ax)

    ax.annotate(f"locked bar {PREREG_BAR:.3f}", xy=(len(DOSE) - 0.45,
                PREREG_BAR), xytext=(0, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=FS_SMALL, color=MUTED)
    # The empty mid-left band (corr sits above 0.90, a_wrong below 0.14 for the
    # first two doses) is the one place a key does not overlap the data.
    ax.legend(handles=[
        dot_handle(ACCENT, "error decorr. $a_{wrong}$"),
        dot_handle(MUTED, "stream correlation"),
        patch_handle(FILL, "gate weight on memory")],
        loc="center left", bbox_to_anchor=(-0.015, 0.545),
        fontsize=FS_SMALL, handlelength=1.0, handletextpad=0.35,
        labelspacing=0.42, borderpad=0.0)


def _stream_auc_ci(labels, pids, prob, resamples: int, seed: int):
    """Patient-clustered bootstrap of one stream's AUC.

    Same protocol as ``scripts/retrieval_crossencoder_screen.py``
    (``patient_bootstrap``): resample the 66 CV patients with replacement, never
    the images, and drop resamples that draw a single class.
    """
    from sklearn.metrics import roc_auc_score

    uniq = np.unique(pids)
    rows = {p: np.nonzero(pids == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(resamples):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([rows[uniq[i]] for i in pick])
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(roc_auc_score(labels[idx], prob[idx]))
    v = np.asarray(vals)
    return (float(roc_auc_score(labels, prob)),
            float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def draw_fig5b(fig, spec, screen: dict | None = None, probs=None,
               resamples: int = 2000, seed: int = 42) -> None:
    """Five probability streams with patient-clustered 95% intervals.

    A forest plot: rows ordered by effect, a dashed reference at the parametric
    head, and the two paired contrasts as thin brackets in the right gutter.
    The contrasts have to be *in* the panel because a paired difference and its
    interval cannot be recovered from the marginal intervals -- that is exactly
    what clustering the bootstrap over patients buys.
    """
    if screen is None or probs is None:
        screen, probs = load_fig5()
    labels, pids = probs["label"], probs["patient_id"]

    ax = fig.add_subplot(spec)
    y = np.arange(len(STREAMS))
    row_y, base = {}, None
    for yi, (key, label, hot) in zip(y, STREAMS):
        point, ci_lo, ci_hi = _stream_auc_ci(labels, pids, probs[key],
                                             resamples, seed)
        row_y[key] = yi
        if key == "prob_param":
            base = point
        colour = ACCENT if hot else MUTED
        ax.plot([ci_lo, ci_hi], [yi, yi], color=colour, lw=LW_ERROR,
                solid_capstyle="butt", zorder=3)
        for e in (ci_lo, ci_hi):
            ax.plot([e, e], [yi - 0.13, yi + 0.13], color=colour,
                    lw=LW_ERROR, zorder=3)
        ax.plot([point], [yi], marker="o", ms=MS_MAIN if hot else MS_CONTEXT,
                color=colour, markeredgecolor="white", markeredgewidth=0.6,
                zorder=4)
        ax.annotate(f"{point:.4f}", xy=(point, yi), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS_SMALL, color=INK if hot else MUTED)

    ax.set_xlim(0.795, 1.005)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00],
                  ["0.80", "0.85", "0.90", "0.95", "1.00"])
    categorical_ylim(ax, len(STREAMS))
    ax.set_yticks(y, [s[1] for s in STREAMS])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=RULE, lw=LW_AXIS * 0.7)
    ax.set_xlabel("image AUC  (95% CI, patient-clustered)")
    despine(ax, keep=("bottom",))
    reference_line(ax, base, axis="x", color=MUTED, zorder=2)

    # Contrast brackets, in a gutter *outside* the axes: extending xlim to make
    # room would drag the grid out with it and give the panel a gridded region
    # wider than its own axis.
    tr = blended_transform_factory(ax.transAxes, ax.transData)

    #: The shorter span nests inside the longer one, and both labels start at
    #: one x -- a text column reads as a table, staggered labels read as
    #: callouts.
    LABEL_X = 1.115

    def bracket(k_hi: str, k_lo: str, xb: float, delta: dict, hot: bool) -> None:
        y_hi, y_lo = row_y[k_hi], row_y[k_lo]
        col = ACCENT if hot else MUTED
        ax.plot([xb, xb], [y_lo, y_hi], color=col, lw=LW_REF,
                transform=tr, clip_on=False, zorder=5)
        for yy in (y_lo, y_hi):
            ax.plot([xb - 0.030, xb], [yy, yy], color=col, lw=LW_REF,
                    transform=tr, clip_on=False, zorder=5)
        ax.annotate(f"{delta['delta']:+.4f}\n"
                    f"[{delta['ci95'][0]:+.3f}, {delta['ci95'][1]:+.3f}]"
                    .replace("-", "−"),
                    xy=(LABEL_X, (y_lo + y_hi) / 2), xycoords=tr,
                    ha="left", va="center", fontsize=FS_SMALL, color=col,
                    linespacing=1.4, annotation_clip=False)

    boot = screen["bootstrap"]
    bracket("prob_ens_half", "prob_final_loso", 1.040,
            boot["final_vs_ens_half"]["auc"], hot=True)
    bracket("prob_ens_half", "prob_param", 1.085,
            boot["ens_half_vs_param"]["auc"], hot=False)
