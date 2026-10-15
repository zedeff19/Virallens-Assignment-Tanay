"""Turn cleaned lines into paragraphs grouped under a heading path (sections).

Responsibilities: paragraph reassembly (indent / vertical gap / cross-page joins), heading
detection (bold font; centred bare numerals for Macquarie), footnote paragraphs, numbered
paragraphs in the complaint, and section kinds (body / appendix / toc / toa).
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from src.ingest.clean import clean_page, join_lines
from src.ingest.docs import DocSpec
from src.ingest.load import Line, Page

_ROMAN_LABEL = re.compile(r"^(?P<label>[IVX]+)(?:\.\s*(?P<rest>\S.*)?)?$")
_LETTER_LABEL = re.compile(r"^(?P<label>[A-Z])\.(?:\s*(?P<rest>\S.*)?)?$")
_DIGIT_LABEL = re.compile(r"^(?P<label>\d{1,2})\.(?:\s*(?P<rest>\S.*)?)?$")
_PARA_NO = re.compile(r"^(\d{1,3})\.\s+(\S.*)$|^(\d{1,3})\.$")
_KIND_LABEL = {"toc": "Table of Contents", "toa": "Table of Authorities", "appendix": "Appendix"}
# Top-level section names that are followed by their own (caps) title; never merge the two.
_TOP_LEVEL = re.compile(r"^(ARGUMENT|STATEMENT|SUMMARY OF ARGUMENT|CONCLUSION)$")
_INDENT = 6.0  # pt
_GAP_FACTOR = 1.35


@dataclass
class Para:
    text: str
    page_start: int
    page_end: int
    label_start: str
    kind: str = "body"  # body | footnote | toc | toa
    para_no: int | None = None


@dataclass
class Section:
    path: tuple[str, ...]
    kind: str  # body | appendix | toc | toa
    paras: list[Para] = field(default_factory=list)


@dataclass
class _Heading:
    text: str
    rank: int  # 1 = roman/ALL-CAPS, 2 = letter/unlabelled, 3 = digit


def _heading_rank(text: str) -> int:
    """0 = unlabelled ALL-CAPS (ARGUMENT, FIRST CLAIM FOR RELIEF), 1 = roman numeral,
    2 = letter / unlabelled mixed case, 3 = digit."""
    t = text.strip()
    if re.fullmatch(r"[IVX]+\.?", t) or re.match(r"^[IVX]+\.(?=\s|[A-Z])", t):
        return 1
    if re.fullmatch(r"[A-Z]\.?", t) or _LETTER_LABEL.match(t):
        return 2
    if _DIGIT_LABEL.match(t):
        return 3
    letters = [c for c in t if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return 0
    return 2


_BARE_LABEL = re.compile(r"([IVX]+|[A-Z]|\d{1,2})\.")


def _starts_new_heading(text: str) -> bool:
    return bool(_ROMAN_LABEL.match(text) or _LETTER_LABEL.match(text) or _DIGIT_LABEL.match(text))


def _is_centered_label(ln: Line, page_width: float) -> bool:
    return bool(re.fullmatch(r"[IVX]+|[A-Z]", ln.text)) and abs((ln.x0 + ln.x1) / 2 - page_width / 2) < 25


def _page_kind(spec: DocSpec, page: int) -> str:
    if page in spec.toc_pages:
        return "toc"
    if page in spec.toa_pages:
        return "toa"
    if spec.appendix_from is not None and page >= spec.appendix_from:
        return "appendix"
    return "body"


def _part_for(spec: DocSpec, page: int) -> str | None:
    label = None
    for first, name in spec.parts:
        if page >= first:
            label = name
    return label


class _Builder:
    def __init__(self, spec: DocSpec, vocab: set[str]):
        self.spec = spec
        self.vocab = vocab
        self.sections: list[Section] = []
        self.stack: list[_Heading] = []
        self.part: str | None = None
        self.kind = "body"
        self.cur: Section | None = None
        self.last_para_no = 0
        self.paras_since_heading = 0
        # paragraph under construction
        self.lines: list[str] = []
        self.pages: list[int] = []
        self.label0 = ""
        self.para_no: int | None = None
        self.label_prefix = ""  # bare list label to prepend to the next paragraph
        # footnotes buffered until the current body paragraph closes
        self.pending_foot: list[Para] = []
        self.foot_target: Para | None = None  # footnote paragraph currently being extended
        self.last_foot: Para | None = None
        self.foot_left = 0.0

    # -- sections --------------------------------------------------------------------------
    def _path(self) -> tuple[str, ...]:
        head = [self.part] if self.part else []
        return tuple(head + [h.text for h in self.stack])

    def _section(self) -> Section:
        path = self._path()
        if self.cur is None or self.cur.path != path or self.cur.kind != self.kind:
            self.cur = Section(path=path, kind=self.kind)
            self.sections.append(self.cur)
        return self.cur

    # -- paragraphs ------------------------------------------------------------------------
    def close_paragraph(self) -> None:
        if self.lines:
            text = join_lines(self.lines, self.vocab)
            drop = any(re.match(p, text) for p in self.spec.drop_paragraph_res)
            if text and not drop:
                self._section().paras.append(
                    Para(text, self.pages[0], self.pages[-1], self.label0, "body", self.para_no)
                )
                self.paras_since_heading += 1
        self.lines, self.pages, self.para_no = [], [], None
        for fp in self.pending_foot:
            self._section().paras.append(fp)
        self.pending_foot = []

    def add_line(self, ln: Line, para_no: int | None = None) -> None:
        if not self.lines:
            self.label0 = ln.label
            self.para_no = para_no
            if self.label_prefix:
                self.lines.append(self.label_prefix)
                self.pages.append(ln.page)
                self.label_prefix = ""
        self.lines.append(ln.text)
        self.pages.append(ln.page)

    def set_heading(self, text: str, rank_text: str | None = None) -> None:
        self.close_paragraph()
        rank = _heading_rank(rank_text or text)
        # Consecutive top-level caps headings with no text between them nest (ARGUMENT > TITLE).
        top_is_fresh_rank1 = (
            rank == 0 and self.stack and self.stack[-1].rank == 0 and self.paras_since_heading == 0
        )
        if not top_is_fresh_rank1:
            while self.stack and self.stack[-1].rank >= rank:
                self.stack.pop()
        self.stack.append(_Heading(text, rank))
        self.paras_since_heading = 0
        self.cur = None


def _page_stats(lines: list[Line]) -> tuple[float, float]:
    body = [l for l in lines if not l.bold and not l.footnote]
    if len(body) < 6:  # too few lines for a reliable mode; caller falls back to the document margin
        return 0.0, 0.0
    xs = [round(l.x0) for l in body]
    left = statistics.mode(xs)
    pitches = [b.y0 - a.y0 for a, b in zip(body, body[1:]) if 0 < b.y0 - a.y0 < 60]
    pitch = statistics.median(pitches) if pitches else 0.0
    return float(left), float(pitch)


_TERMINAL = re.compile(r"[.!?\"”’)\]:;]$")


def _add_footnote_line(b: _Builder, ln: Line, foot_prev: Line | None, vocab: set[str]) -> None:
    """Append a footnote line to the current footnote paragraph, start a new one, or continue a
    footnote that began at the bottom of the previous page."""
    if (
        foot_prev is not None
        and foot_prev.page == ln.page
        and ln.x0 <= foot_prev.x0 + _INDENT
        and b.foot_target is not None
    ):
        b.foot_target.text = join_lines([b.foot_target.text, ln.text], vocab)
        return
    last = b.last_foot
    if (
        foot_prev is None
        and last is not None
        and last.page_end == ln.page - 1
        and not _TERMINAL.search(last.text)
        and ln.x0 <= b.foot_left + _INDENT
    ):
        last.text = join_lines([last.text, ln.text], vocab)
        last.page_end = ln.page
        b.foot_target = last
        return
    para = Para(ln.text, ln.page, ln.page, ln.label, "footnote")
    b.pending_foot.append(para)
    b.foot_target = b.last_foot = para
    b.foot_left = ln.x0


def build_sections(spec: DocSpec, pages: list[Page], vocab: set[str]) -> list[Section]:
    b = _Builder(spec, vocab)
    heading_lines: list[str] = []  # bold run being accumulated

    def flush_heading() -> None:
        nonlocal heading_lines
        if not heading_lines:
            return
        text = join_lines(heading_lines, vocab)
        if _BARE_LABEL.fullmatch(text):
            # a bare list label ("I.") in front of body text, e.g. the prayer for relief items
            b.close_paragraph()
            b.label_prefix = text
        else:
            # rank comes from the first line: "FIRST CLAIM FOR RELIEF" is a top-level caps heading
            # even though the statute subtitle that follows it is mixed case
            b.set_heading(text, rank_text=heading_lines[0])
        heading_lines = []

    cleaned = [(page, clean_page(spec, page)[1]) for page in pages]
    # Left margin: per page, but pages dominated by block quotes (mode shifted by >= 20pt) fall
    # back to the document-wide margin so quoted text is not mistaken for the body margin.
    page_lefts = {pg.number: _page_stats(ls)[0] for pg, ls in cleaned if ls}
    known = [round(v) for v in page_lefts.values() if v]
    doc_left = float(statistics.mode(known)) if known else 0.0

    prev: Line | None = None  # last non-footnote line (body or heading)
    prev_is_head = False
    foot_prev: Line | None = None  # previous footnote line in the current run
    for page, lines in cleaned:
        if not lines:
            continue
        kind = _page_kind(spec, page.number)
        part = _KIND_LABEL.get(kind) or _part_for(spec, page.number)
        if kind != b.kind or part != b.part:
            flush_heading()
            b.close_paragraph()
            if part != b.part:
                b.stack = []
            b.kind, b.part, b.cur = kind, part, None
            prev, prev_is_head, foot_prev = None, False, None

        if kind in ("toc", "toa"):
            sec = b._section()
            for ln in lines:
                sec.paras.append(Para(ln.text, page.number, page.number, ln.label, kind))
            continue

        left = page_lefts[page.number]
        if not left or left - doc_left >= 20:
            left = doc_left
        first_on_page = True
        last_pitch = 0.0
        for ln in lines:
            if spec.heading_mode == "centered":
                is_head = _is_centered_label(ln, page.width) and not ln.footnote
            else:
                is_head = (
                    ln.bold
                    and not ln.footnote
                    and (spec.heading_max_size is None or ln.size < spec.heading_max_size)
                )
            if is_head:
                if heading_lines and (
                    spec.heading_mode == "centered"
                    or _TOP_LEVEL.match(heading_lines[-1].strip())
                    or (_starts_new_heading(ln.text) and not _BARE_LABEL.fullmatch(heading_lines[-1].strip()))
                ):
                    flush_heading()
                heading_lines.append(ln.text)
                prev, prev_is_head = ln, True
                first_on_page = False
                continue
            flush_heading()

            if ln.footnote:
                _add_footnote_line(b, ln, foot_prev, vocab)
                foot_prev = ln
                continue
            foot_prev = None

            para_no = None
            new_para = False
            m = _PARA_NO.match(ln.text) if spec.numbered_paragraphs else None
            if m:
                num = int(m.group(1) or m.group(3))
                if num == b.last_para_no + 1:
                    para_no = num
                    b.last_para_no = num
                    new_para = True
            if prev is None or prev_is_head:
                new_para = True
            elif not new_para:
                if ln.x0 > left + _INDENT and ln.x0 > prev.x0 + _INDENT:
                    new_para = True  # first-line indent
                elif last_pitch and ln.page == prev.page and (ln.y0 - prev.y0) > last_pitch * _GAP_FACTOR:
                    new_para = True  # extra vertical space
                elif ln.x0 < left - _INDENT and prev.x0 >= left - _INDENT / 2:
                    new_para = True  # hanging first line (Macquarie syllabus)
            if new_para:
                b.close_paragraph()
                last_pitch = 0.0
            elif prev is not None and ln.page == prev.page:
                last_pitch = ln.y0 - prev.y0
            b.add_line(ln, para_no)
            prev, prev_is_head = ln, False
            first_on_page = False
    flush_heading()
    b.close_paragraph()
    return [s for s in b.sections if s.paras]
