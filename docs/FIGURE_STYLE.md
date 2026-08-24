# Figure design guide — RAP-MST

The house style for every figure in this paper. Derived from the conventions of
MICCAI, CVPR/ICCV, TPAMI, *Medical Image Analysis* and *Nature Biomedical
Engineering*, and from the two publisher specifications that bind us (Springer
LNCS; Nature's figure guide, which is the strictest of the set and therefore the
one we satisfy).

`scripts/paper_style.py` is the executable form of this document. Nothing below
should ever be re-decided inside a figure script.

---

## 0. The governing constraint

The paper is `\documentclass[runningheads]{llncs}` — **single column**,
`\textwidth = 347.12 pt = 4.803 in`. That is the widest a figure can be.

Two consequences that drive every other decision:

1. **Figures are authored at final size and never rescaled.** A figure drawn at
   8 in and included at `width=\textwidth` shrinks by 0.6×, and an 8 pt label
   lands on the page at 4.8 pt. So: canvas width is exactly 4.803 in, and
   `\includegraphics[width=\textwidth]` reproduces it 1:1. No `bbox_inches`
   (a tight box crops each figure to a different width, reintroducing per-figure
   scale factors) — explicit margins instead.
2. **The caption is 9 pt roman and sits directly under the figure.** Nothing in
   the figure may be visually louder than the caption, or the reading order
   inverts. This is what rules out bold callouts and oversized titles.

---

## 1. What goes in the figure, and what goes in the caption

The single most important rule, and the one the previous set broke:

> **The figure presents evidence. The caption makes the argument.**

| in the figure | in the caption |
|---|---|
| axis labels with units | what the experiment was |
| tick labels | sample sizes, folds, protocol |
| category / condition names | which comparison matters and why |
| the measured values, and their uncertainty | the interpretation, the verdict, the caveats |
| a reference line or level, labelled nominally | what the reference line means for the claim |

Panel titles are **nouns or absent** — "Error decorrelation", not "the
mechanism, three doses". If a panel title contains a verb, it is a conclusion
and it belongs in the caption. No sentences inside the axes. No figure-level
subtitles restating the caption's first line.

Panel labels `(a)`, `(b)` are added **only at assembly**, never by the panel
itself; that way a panel can be reused, reordered, or promoted to a figure of
its own without editing.

---

## 2. Typography

| role | size | weight | colour |
|---|---|---|---|
| panel label `(a)` | 9.0 pt | bold | ink |
| panel title (nominal) | 8.0 pt | regular | ink |
| axis label | 8.0 pt | regular | ink |
| tick label | 7.0 pt | regular | muted |
| annotation / in-panel value | 7.0 pt | regular | ink or muted |
| legend | 7.0 pt | regular | ink |
| secondary / reference note | 6.5 pt | regular | muted |

- **Typeface: Arial**, with Helvetica and DejaVu Sans as fallbacks. Sans-serif
  in figures against a serif body text is universal in all six venues.
- **Mathtext must match.** matplotlib's default renders `$p_{final}$` in DejaVu
  while the surrounding label is Arial. `mathtext.fontset = "custom"` bound to
  the sans family fixes it. Mixed typefaces inside one tick label is the
  clearest signal of an undesigned figure.
- **6.5 pt is the floor.** Nothing smaller, ever — Nature's minimum is 5 pt at
  final size and we have no rescaling headroom to spend.
- **Bold appears exactly once per figure: the panel label.** Emphasis is
  achieved by removing weight from everything else, not by adding weight to the
  highlight. No bold numbers, no bold column headings.
- **Italics are reserved for mathematical variables.** Not for asides, not for
  notes, not for headers.
- **Sentence case throughout**, lowercase axis labels included. One convention.

---

## 3. Colour

Three inks, plus a data palette. Okabe–Ito derived, so every pairing survives
deuteranopia, protanopia and greyscale conversion.

```
INK      #16181D   text, data marks that carry no category
MUTED    #5C626B   ticks, secondary annotation, reference labels
RULE     #C8CDD4   hairlines, grid, panel frames
FILL     #E9EDF1   quantitative area fills (bars, bands)

ACCENT   #D55E00   THE condition under test        (vermillion)
COOL     #0072B2   the contrasting category        (blue)
```

**The accent rule.** `ACCENT` has exactly one meaning per figure and, wherever
possible, one meaning across the paper: *the thing the panel is about*. Every
other series is grey. This is the single largest difference between the
reference venues' figures and ordinary ones — they use one saturated hue against
2–3 greys and the reader learns the code once.

Where a figure needs a categorical palette (Fig. 4's eight tumour subtypes), it
must be **nested, not flat**: benign subtypes drawn from a cool ramp, malignant
from a warm one, so the binary structure is still legible in the subtype panel
without being a recolouring of the binary panel. The two most confusable
categories (MC, PC) get the largest separation in the ramp, because they are
what the panel is about.

Fills never carry more visual weight than the data they sit behind: a reference
band is a light bounded region, not a solid slab across the panel.

---

## 4. Axes, marks and rules

| element | value | note |
|---|---|---|
| spine / tick width | 0.75 pt | publisher floor for a rendered line |
| tick length | 2.6 pt, outward | |
| spines kept | left + bottom only | top/right removed everywhere |
| grid | 0.5 pt, `RULE`, below data | only where a reader must read a value across |
| data line | 1.2 pt | |
| error bar / whisker | 1.0 pt | |
| reference line | 0.8 pt, dashed `(0,(3.6,2.2))`, `RULE`/`MUTED` | never solid, never at data weight |
| marker (emphasised) | 4.2 pt, white 0.6 pt edge | the white keyline separates overlapping marks |
| marker (context) | 3.4 pt | |
| scatter point (dense cloud) | 1.6 pt², α 0.5, rasterised | vector for 6,600 points is unprintable |

- **3–5 labelled major ticks per axis.** Never place an unlabelled major tick;
  never blank alternate labels.
- **The axis range is set by the data**, and where a zoomed range is used the
  tick labels are left at full precision so the zoom is self-evident. **Break
  marks are only ever drawn where a range has actually been removed** — a
  non-zero origin is not a break.
- **Grid lines are clipped to the axis.** Annotation gutters live outside the
  axes (`clip_on=False` in figure/axes coordinates), never by extending `xlim`
  past the spine.
- **Row spacing in categorical panels follows one rule** (`ylim = -0.6,
  n - 0.4`), shared by every forest-style panel in the paper, so two such panels
  in different figures have identical rhythm.

---

## 5. Layout and whitespace

- **Data occupies ≥ 70% of each panel's area.** If half a panel is empty, the
  range is wrong or the panel is the wrong shape.
- **Panel width is allocated in proportion to the claim it carries**, not
  equally by default.
- **Margins are identical across all figures** — the same `left`/`right`/`top`/
  `bottom` fractions in every assembly, so consecutive figures align with each
  other and with the text block.
- **Legends sit outside the data region**, frameless, horizontal, single row
  where the entry count allows, `handletextpad ≈ 0.4`. A legend inside the axes
  is acceptable only in genuinely empty space and never with a frame.
- **Where two panels share a categorical key, the legend is shared** and placed
  once, at assembly.
- **Panel labels: `(a)`/`(b)` at the top-left of each panel's cell**, in figure
  coordinates at a fixed offset — so they sit at the same distance from every
  panel regardless of how wide that panel's tick labels are. (Placing them in
  axes coordinates is what produced `x=-0.02`, `x=-0.19`, `x=-0.30` in the
  previous set.)

---

## 6. Idioms to use, by chart type

**Forest / interval plot** (Fig. 5b, Fig. 3b). Rows sorted by effect. A dashed
reference line at the comparator. Thin whiskers with short end caps, small
points, values in a right-aligned column that reads as a table. Contrasts, if
shown, are thin brackets in an outside gutter with one short label — never
multi-line bold text in the margin.

**Zoom inset** (Fig. 3a). The magnified region is *marked on the parent axes*
and joined to the inset by two hairline leaders. Without the leaders it is not
an inset; it is a second plot sitting on top of the first. The inset carries no
explanatory text of its own — only its ticks.

**Dose–response** (Fig. 5a). All series that share a range share the axis. If a
quantity is in the panel's units and monotone across the doses, it is **plotted**
— never typeset as a row of text under the tick labels.

**Embedding scatter** (Fig. 4). No frame, no ticks, no axis: a projection has no
meaningful scale, so a box with no ticks is a box that says nothing. Orientation
is conveyed by one small axis glyph in the first panel only. Equal aspect,
per-panel autoscale with a common pad. Small, translucent, rasterised points so
density reads. Per-panel metrics are set *outside* the cloud, at tick weight, in
muted grey — they are annotations, not results.

---

## 7. Production

- **Vector out**: PDF (for LaTeX) and SVG (for editing). `pdf.fonttype = 42`
  (TrueType, not Type-3 — Springer requires it), `ps.fonttype = 42`.
- **Rasterise only the dense scatter layers**, at 600 dpi, via
  `set_rasterization_zorder`. Text, axes and rules stay vector at any zoom.
- **No `bbox_inches="tight"`.** Explicit margins; the PDF is exactly
  `\textwidth` wide.
- **Raw panels are rendered separately from the assembled figure, from the same
  drawing functions.** One code path, so a panel and its assembled copy cannot
  drift; the assembler adds only the panel labels and any shared legend.
- **Stochastic content is cached** (Fig. 4's UMAP), so the standalone panel and
  the assembled panel are the same picture rather than two draws.
