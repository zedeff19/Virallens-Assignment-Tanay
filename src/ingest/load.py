"""Load the PDFs in data/ with PyMuPDF, one record per page (page numbers kept).

Each page becomes a list of `Line`s in visual (top-to-bottom, left-to-right) order, carrying
the font information later stages need: bold -> headings, size/superscript -> footnote
markers and footnote text. PyMuPDF's block order is *not* reliably visual order, and
italic/bold fragments on one baseline arrive as separate lines, so lines are re-sorted and
same-baseline fragments merged here.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

from src.ingest.docs import DocSpec

_MARKER_TEXT = re.compile(r"^(\d{1,2}|[*†‡])$")
_MARKER_SIZE_RATIO = 0.8  # span this much smaller than body text and 1-2 digits => footnote marker
_FOOTNOTE_SIZE_RATIO = 0.95  # whole line smaller than body, low on the page => footnote text
_SAME_BASELINE = 2.5  # pt


@dataclass
class Line:
    text: str
    bold: bool
    size: float
    x0: float
    x1: float
    y0: float
    y1: float
    page: int
    label: str = ""  # printed page label of the page this line is on
    footnote: bool = False  # line belongs to footnote text
    marker_only: bool = False  # line consisted solely of footnote-marker spans


@dataclass
class Page:
    number: int  # 1-based PDF page
    width: float
    height: float
    lines: list[Line] = field(default_factory=list)
    body_size: float = 0.0
    raw_chars: int = 0  # len(page.get_text()), for the loss report


def _body_size(lines: list[Line]) -> float:
    """Largest font size carried by >= 3 non-bold lines (falls back to the modal size)."""
    counts: Counter[float] = Counter()
    for ln in lines:
        if ln.text and not ln.bold:
            counts[round(ln.size, 1)] += 1
    big = [s for s, n in counts.items() if n >= 3]
    if big:
        return max(big)
    return counts.most_common(1)[0][0] if counts else 0.0


def _modal_char_size(spans: list[dict]) -> float:
    counts: Counter[float] = Counter()
    for s in spans:
        if s["text"].strip():
            counts[round(s["size"], 1)] += len(s["text"])
    return counts.most_common(1)[0][0] if counts else 0.0


def _visual_order(lines: list[Line]) -> list[Line]:
    """Sort top-to-bottom, then left-to-right within a baseline group."""
    out: list[Line] = []
    group: list[Line] = []
    for ln in sorted(lines, key=lambda l: l.y0):
        if group and ln.y0 - group[0].y0 > _SAME_BASELINE:
            out.extend(sorted(group, key=lambda l: l.x0))
            group = []
        group.append(ln)
    out.extend(sorted(group, key=lambda l: l.x0))
    return out


def _merge_same_baseline(lines: list[Line]) -> list[Line]:
    merged: list[Line] = []
    for ln in lines:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and not ln.marker_only
            and not prev.marker_only
            and abs(ln.y0 - prev.y0) <= _SAME_BASELINE
            and ln.x0 >= prev.x0
        ):
            prev.text = f"{prev.text} {ln.text}"
            prev.bold = prev.bold and ln.bold
            prev.size = max(prev.size, ln.size)
            prev.x1 = max(prev.x1, ln.x1)
            prev.y1 = max(prev.y1, ln.y1)
        else:
            merged.append(ln)
    return merged


def load_pdf(spec: DocSpec) -> list[Page]:
    pages: list[Page] = []
    with pymupdf.open(spec.path) as pdf:
        for idx, pg in enumerate(pdf, start=1):
            data = pg.get_text("dict")
            spans_all = [s for b in data["blocks"] for l in b.get("lines", []) for s in l["spans"]]
            modal = _modal_char_size(spans_all)
            page = Page(number=idx, width=pg.rect.width, height=pg.rect.height, raw_chars=len(pg.get_text()))
            raw: list[Line] = []
            for b in data["blocks"]:
                for l in b.get("lines", []):
                    if not any(s["text"].strip() for s in l["spans"]):
                        continue
                    # whitespace-only spans (a space in the other font) must stay in the text
                    markers, kept = [], []
                    for s in l["spans"]:
                        small = modal and s["size"] <= modal * _MARKER_SIZE_RATIO
                        raised = bool(s["flags"] & 1)
                        if s["text"].strip() and _MARKER_TEXT.match(s["text"].strip()) and (small or raised):
                            markers.append(s)
                        else:
                            kept.append(s)
                    x0, y0, x1, y1 = l["bbox"]
                    words = [s for s in kept if s["text"].strip()]
                    if not words:
                        raw.append(Line("", False, modal, x0, x1, y0, y1, idx, marker_only=True))
                        continue
                    text = re.sub(r"\s+", " ", "".join(s["text"] for s in kept)).strip()
                    bold = all(bool(s["flags"] & 16) or "Bold" in s["font"] for s in words)
                    raw.append(Line(text, bold, max(s["size"] for s in words), x0, x1, y0, y1, idx))
            page.lines = _merge_same_baseline(_visual_order(raw))
            page.body_size = _body_size(page.lines)
            _mark_footnotes(page)
            pages.append(page)
    return pages


def _mark_footnotes(page: Page) -> None:
    """Flag footnote text lines.

    * Body larger than footnote text (Macquarie, both briefs): smaller non-bold lines in the
      lower half of the page are footnote text.
    * Same size (Last Atlantis): the lowest raised marker line (in the lower 60% of the page)
      opens the footnote region; every line from there down is footnote text.
    """
    if not page.body_size:
        return
    for ln in page.lines:
        if ln.marker_only or ln.bold or not ln.text:
            continue
        if ln.size <= page.body_size * _FOOTNOTE_SIZE_RATIO and ln.y0 > page.height * 0.5:
            ln.footnote = True
    if any(ln.footnote for ln in page.lines):
        return
    # Same-size footnotes (Last Atlantis, Chiueh): body is double-spaced, footnotes are single-
    # spaced. The footnote region starts at the first raised-marker line after which every
    # remaining line is single-spaced (a single-spaced block quote is not preceded by a marker
    # *and* followed only by single-spaced lines).
    def is_footer(ln: Line) -> bool:
        return ln.y0 > page.height * 0.85 and bool(re.fullmatch(r"\(?\d{1,3}\)?", ln.text))

    for m in (ln for ln in page.lines if ln.marker_only):
        after = [ln for ln in page.lines if not ln.marker_only and ln.y0 >= m.y0 - 2 and not is_footer(ln)]
        if len(after) < 2:
            continue
        marker_ys = [mk.y0 for mk in page.lines if mk.marker_only]

        def tight(a: Line, b: Line) -> bool:
            if (b.y0 - a.y0) <= 1.5 * max(a.y1 - a.y0, 1.0):
                return True
            # the gap between two consecutive footnotes carries the next footnote's marker
            return any(a.y0 < y <= b.y0 + 2 for y in marker_ys)

        if all(tight(a, b) for a, b in zip(after, after[1:])):
            for ln in after:
                ln.footnote = True
            return
