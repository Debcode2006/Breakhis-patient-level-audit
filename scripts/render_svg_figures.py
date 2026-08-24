"""Rasterise / vectorise the hand-authored SVG schematics (Figures 1 and 2).

Why this exists
---------------
Figures 3-5 are data figures and are drawn by ``scripts/assemble_figures.py``
straight to PDF. Figures 1 and 2 are *schematics*: they are authored by hand as
SVG (``assets/schematics/fig1_architecture.svg``,
``assets/schematics/fig2_retrieval_memory.svg``) because a box-and-arrow diagram
is easier to lay out in markup than in matplotlib. Those two SVGs are the only
non-regenerable figure sources in the repository. This script is the missing half
of that workflow -- it turns each SVG into the two artefacts LaTeX and reviewers
want, writing them into ``images/`` alongside the data figures:

* ``<name>.pdf``  -- vector, what ``\\includegraphics`` should use;
* ``<name>.png``  -- 5x raster fallback for slides / previews / quick review.

It shells out to headless Chrome, which is the only SVG renderer present on this
machine (no Inkscape, no cairosvg). Chrome prints at 96 CSS px per inch, so an
``N``-px-wide SVG becomes an ``N * 0.75`` pt-wide PDF page with no margin; the
figure is included at ``width=\\textwidth`` anyway, so the intrinsic size only
has to keep the aspect ratio, which it does.

Fonts are embedded as subsets and no rasterisation happens on the PDF path
(verify with ``pdffonts``); the only bitmap in either figure is the histology
patch already embedded in Figure 1 as a data URI.

Usage
-----
    python scripts/render_svg_figures.py                 # both figures, pdf+png
    python scripts/render_svg_figures.py --only fig1_architecture
    python scripts/render_svg_figures.py --formats pdf
    python scripts/render_svg_figures.py --png-scale 3
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# Chrome/Chromium locations to try, in order, before falling back to $PATH.
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
]

_SIZE_RE = re.compile(r'\bwidth="(?P<w>[\d.]+)"\s+height="(?P<h>[\d.]+)"')
_VIEWBOX_RE = re.compile(r'viewBox="[\d.\s]*?([\d.]+)\s+([\d.]+)"')


def find_chrome() -> str:
    for cand in CHROME_CANDIDATES:
        if Path(cand).exists():
            return cand
        found = shutil.which(cand)
        if found:
            return found
    raise SystemExit(
        "No Chrome/Chromium/Edge binary found. Install one, or add its path to "
        "CHROME_CANDIDATES at the top of this script."
    )


def svg_size(svg_path: Path) -> Tuple[float, float]:
    """Intrinsic size in CSS px, from width/height, falling back to viewBox."""
    head = svg_path.read_text(encoding="utf-8")[:2000]
    m = _SIZE_RE.search(head)
    if m:
        return float(m.group("w")), float(m.group("h"))
    m = _VIEWBOX_RE.search(head)
    if m:
        return float(m.group(1)), float(m.group(2))
    raise SystemExit(f"Cannot determine the intrinsic size of {svg_path}")


def _wrapper_html(svg_name: str, w: float, h: float, page: bool) -> str:
    """A margin-free page holding exactly one <img> at the SVG's own size.

    ``@page`` must match the image box exactly or Chrome paginates and the
    figure is silently cropped at the page break.
    """
    page_rule = f"@page{{size:{w}px {h}px;margin:0}}" if page else ""
    return (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        f"{page_rule}"
        "html,body{margin:0;padding:0;background:#fff}"
        f"img{{display:block;width:{w}px;height:{h}px}}"
        f"</style></head><body><img src='{svg_name}'></body></html>"
    )


def render(svg_path: Path, out_dir: Path, formats: List[str], png_scale: int,
           chrome: str) -> None:
    w, h = svg_size(svg_path)
    stem = svg_path.stem
    # Chrome resolves --print-to-pdf / --screenshot against its own working
    # directory, not ours: a relative path here silently writes elsewhere and
    # still exits 0. Always hand it an absolute path.
    out_dir = out_dir.resolve()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        shutil.copy(svg_path, tmp / svg_path.name)
        common = ["--headless", "--disable-gpu", "--no-sandbox",
                  "--virtual-time-budget=4000"]

        if "pdf" in formats:
            (tmp / "page.html").write_text(
                _wrapper_html(svg_path.name, w, h, page=True), encoding="utf-8")
            out_pdf = out_dir / f"{stem}.pdf"
            subprocess.run(
                [chrome, *common, "--no-pdf-header-footer",
                 f"--print-to-pdf={out_pdf}", str(tmp / 'page.html')],
                check=True, capture_output=True)
            if not out_pdf.exists():
                raise SystemExit(f"Chrome reported success but {out_pdf} is missing")
            print(f"  wrote {out_pdf}  ({w:.0f}x{h:.0f} px -> "
                  f"{w * 0.75:.1f}x{h * 0.75:.1f} pt)")

        if "png" in formats:
            (tmp / "shot.html").write_text(
                _wrapper_html(svg_path.name, w, h, page=False), encoding="utf-8")
            out_png = out_dir / f"{stem}.png"
            subprocess.run(
                [chrome, *common,
                 f"--force-device-scale-factor={png_scale}",
                 f"--window-size={int(w)},{int(h)}",
                 f"--screenshot={out_png}", str(tmp / 'shot.html')],
                check=True, capture_output=True)
            if not out_png.exists():
                raise SystemExit(f"Chrome reported success but {out_png} is missing")
            print(f"  wrote {out_png}  ({png_scale}x = "
                  f"{int(w) * png_scale}x{int(h) * png_scale} px)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="assets/schematics",
                    help="Directory holding the hand-authored SVG sources.")
    ap.add_argument("--out", default="images",
                    help="Directory the rendered PDF/PNG are written to.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Stem(s) to render, e.g. fig2_retrieval_memory.")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"],
                    choices=["pdf", "png"])
    ap.add_argument("--png-scale", type=int, default=5)
    args = ap.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    svgs = sorted(src_dir.glob("*.svg"))
    if args.only:
        wanted = set(args.only)
        svgs = [s for s in svgs if s.stem in wanted]
    if not svgs:
        raise SystemExit(f"No SVG to render under {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    chrome = find_chrome()
    print(f"renderer: {chrome}")
    for svg in svgs:
        print(f"{svg.name}")
        render(svg, out_dir, args.formats, args.png_scale, chrome)


if __name__ == "__main__":
    main()
