"""Section/paragraph-aware chunking (~400 tokens, ~50 overlap) with chunk metadata; writes chunks.jsonl.

Rules (planv3 §4):
* chunks never cross a heading (they are built per `Section`);
* paragraphs are atomic unless a single paragraph exceeds the hard cap, in which case it is
  split at sentence boundaries;
* ~50 tokens of overlap (whole trailing sentences of the previous chunk) only inside one section;
* complaint paragraphs are grouped, never overlapped, and keep their ¶ range;
* toc/toa chunks are kept but flagged `searchable=False`.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.ingest.docs import DocSpec, cite_text
from src.ingest.structure import Para, Section

TARGET_TOKENS = 400
HARD_TOKENS = 480  # body cap; also bounded by 500 - header tokens (bge-small max_seq_length = 512)
OVERLAP_TOKENS = 50
MIN_FILL_TOKENS = 150  # do not close a chunk below this if the next unit still fits under the hard cap
MIN_TAIL_TOKENS = 60  # a trailing chunk smaller than this is merged into its predecessor
MODEL_BUDGET = 500  # 512 minus [CLS]/[SEP] and slack

_ABBREVIATIONS = {
    "u.s", "u. s", "s", "ct", "cir", "inc", "corp", "co", "ltd", "llc", "l.p", "v", "id", "no", "nos", "fed",
    "r", "civ", "p", "c.f.r", "u.s.c", "stat", "supp", "app", "dkt", "mr", "mrs", "dr", "sec", "reg", "rep",
    "ill", "n.d", "d", "e.g", "i.e", "et al", "al", "cf", "ed", "ch", "pet", "br", "op", "pp", "approx",
    "div", "comm'n", "mgmt", "res", "bd", "exch", "cong", "pub", "l", "f", "j", "a", "b", "g", "st", "mass",
    "wl", "cr", "cv", "n.y", "cal", "pa", "n.j", "fla", "tex", "wash", "ind", "mich", "minn", "mo", "ohio",
}
_RULE_REF = [
    (re.compile(r"10b-5\(([abc])\)"), lambda m: f"10b-5({m.group(1)})"),
    (re.compile(r"10b-5(?!\()"), lambda m: "10b-5"),
    (re.compile(r"Item (103|105|106|303|503)\b"), lambda m: f"Item {m.group(1)}"),
    (re.compile(r"(?:Section|§)\s?10\(b\)"), lambda m: "Section 10(b)"),
    (re.compile(r"(?:Section|§)\s?17\(a\)(?:\((\d)\))?"), lambda m: f"Section 17(a){'(' + m.group(1) + ')' if m.group(1) else ''}"),
    (re.compile(r"Rule 206\(4\)-8"), lambda m: "Rule 206(4)-8"),
]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    cite_label: str
    section_path: list[str]
    section_kind: str  # body | appendix | toc | toa
    searchable: bool
    page_start: int
    page_end: int
    page_label_start: str
    para_start: int | None
    para_end: int | None
    prev_id: str | None
    next_id: str | None
    is_footnote_only: bool
    header: str
    text: str
    n_tokens: int  # body text tokens (excludes header)
    overlap_tokens: int
    overlap_chars: int
    rule_refs: list[str] = field(default_factory=list)
    citation: str = ""  # e.g. "[Macquarie, p.5]" / "[Chiueh ¶¶45–52]"

    @property
    def contextual_text(self) -> str:
        return f"{self.header}\n{self.text}"


# --------------------------------------------------------------------------------------- tokens
def get_token_counter() -> tuple[Callable[[str], int], str]:
    """bge-small tokenizer (matches the embedder's 512 limit); regex approximation as fallback."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        return (lambda s: len(tok(s, add_special_tokens=False)["input_ids"])), "BAAI/bge-small-en-v1.5"
    except Exception:  # offline / not installed
        return (lambda s: int(len(re.findall(r"\w+|[^\w\s]", s)) * 1.2)), "regex-approx"


# ------------------------------------------------------------------------------------ sentences
_SENT_BOUNDARY = re.compile(r"(?<=[.?!])[\"”’')\]]*\s+(?=[\"“‘(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter that re-joins splits after legal abbreviations / initials."""
    pieces: list[str] = []
    last = 0
    for m in _SENT_BOUNDARY.finditer(text):
        cand = text[last : m.start()]
        tail = re.split(r"\s", cand.rstrip(".?!\"”’')]"))[-1].lower().rstrip(".")
        tail_full = tail.strip("(\"“‘[")
        if tail_full in _ABBREVIATIONS or re.fullmatch(r"[a-z]", tail_full):
            continue
        pieces.append(text[last : m.end()].strip())
        last = m.end()
    pieces.append(text[last:].strip())
    return [p for p in pieces if p]


def _split_to_budget(text: str, budget: int, count: Callable[[str], int]) -> list[str]:
    """Split an over-long paragraph into pieces of <= budget tokens at sentence boundaries."""
    out: list[str] = []
    cur: list[str] = []
    cur_t = 0
    for sent in split_sentences(text):
        t = count(sent)
        if t > budget:  # no usable boundary: fall back to word windows
            if cur:
                out.append(" ".join(cur))
                cur, cur_t = [], 0
            words, buf = sent.split(), []
            for w in words:
                buf.append(w)
                if count(" ".join(buf)) >= budget:
                    out.append(" ".join(buf[:-1]) or w)
                    buf = buf[-1:] if len(buf) > 1 else []
            if buf:
                cur, cur_t = [" ".join(buf)], count(" ".join(buf))
            continue
        if cur and cur_t + t > budget:
            out.append(" ".join(cur))
            cur, cur_t = [], 0
        cur.append(sent)
        cur_t += t
    if cur:
        out.append(" ".join(cur))
    return out


# -------------------------------------------------------------------------------------- helpers
@dataclass
class _Unit:
    text: str
    tokens: int
    para: Para


def _short(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def make_header(spec: DocSpec, section: Section) -> str:
    path = list(section.path)
    if len(path) > 2:
        path = path[-2:]
    parts = [_short(p, 100) for p in path[:-1]] + ([_short(path[-1], 230)] if path else [])
    return " · ".join([spec.title, spec.doc_type] + parts)


def rule_refs(text: str) -> list[str]:
    found: list[str] = []
    for rx, fmt in _RULE_REF:
        for m in rx.finditer(text):
            ref = fmt(m)
            if ref not in found:
                found.append(ref)
    return found


def _unit_text(p: Para) -> str:
    return f"[Footnote] {p.text}" if p.kind == "footnote" else p.text


def _tail_overlap(prev_text: str, count: Callable[[str], int]) -> str:
    """Trailing whole sentences of the previous chunk's last paragraph, <= OVERLAP_TOKENS."""
    last_para = prev_text.rsplit("\n\n", 1)[-1]
    if last_para.startswith("[Footnote]"):
        return ""
    picked: list[str] = []
    total = 0
    for sent in reversed(split_sentences(last_para)):
        t = count(sent)
        if total + t > OVERLAP_TOKENS:
            break
        picked.insert(0, sent)
        total += t
    # overlap must start at a real sentence start (not "32) that the Ninth Circuit...")
    while picked and not re.match(r"[\"“‘(\[]?[A-Z]", picked[0]):
        picked.pop(0)
    # never repeat a whole one-sentence paragraph as "overlap"
    if not picked or len(picked) == len(split_sentences(last_para)):
        return ""
    return " ".join(picked)


def chunk_section(
    spec: DocSpec, section: Section, count: Callable[[str], int]
) -> list[tuple[str, list[Para], int, int]]:
    """Return [(text, paras, overlap_tokens, overlap_chars)] for one section."""
    header = make_header(spec, section)
    cap = min(HARD_TOKENS, MODEL_BUDGET - count(header))
    target = min(TARGET_TOKENS, cap)
    joiner = "\n" if section.kind in ("toc", "toa") else "\n\n"

    units: list[_Unit] = []
    for p in section.paras:
        text = _unit_text(p)
        t = count(text)
        if t <= cap:
            units.append(_Unit(text, t, p))
        else:
            for piece in _split_to_budget(text, target, count):
                units.append(_Unit(piece, count(piece), p))

    groups: list[list[_Unit]] = []
    cur: list[_Unit] = []
    cur_t = 0
    for u in units:
        if cur and cur_t + u.tokens > target and not (cur_t < MIN_FILL_TOKENS and cur_t + u.tokens <= cap):
            groups.append(cur)
            cur, cur_t = [], 0
        cur.append(u)
        cur_t += u.tokens + (1 if cur_t else 0)
    if cur:
        groups.append(cur)
    if len(groups) > 1:
        tail_t = sum(u.tokens for u in groups[-1])
        prev_t = sum(u.tokens for u in groups[-2])
        if tail_t < MIN_TAIL_TOKENS and prev_t + tail_t <= cap:
            groups[-2].extend(groups.pop())

    use_overlap = section.kind == "body" and not spec.numbered_paragraphs
    out: list[tuple[str, list[Para], int, int]] = []
    prev_text = ""
    for i, g in enumerate(groups):
        body = joiner.join(u.text for u in g)
        ov_t = ov_c = 0
        if use_overlap and i > 0 and not all(u.para.kind == "footnote" for u in g):
            ov = _tail_overlap(prev_text, count)
            if ov and count(ov + joiner + body) <= cap:
                body = f"{ov}{joiner}{body}"
                ov_t, ov_c = count(ov), len(ov) + len(joiner)
        prev_text = joiner.join(u.text for u in g)
        out.append((body, [u.para for u in g], ov_t, ov_c))
    return out


def build_chunks(
    spec: DocSpec, sections: list[Section], count: Callable[[str], int]
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in sections:
        header = make_header(spec, sec)
        for text, paras, ov_t, ov_c in chunk_section(spec, sec, count):
            nos = [p.para_no for p in paras if p.para_no is not None]
            p0, p1 = (min(nos), max(nos)) if nos else (None, None)
            ps = min(p.page_start for p in paras)
            pe = max(p.page_end for p in paras)
            first = min(paras, key=lambda p: p.page_start)
            searchable = sec.kind in ("body", "appendix")
            # planv3 format: [Macquarie, p.5] for pages, [Chiueh ¶45] for complaint paragraphs
            sep = " " if p0 is not None else ", "
            cite = f"[{spec.cite_label}{sep}{cite_text(ps, pe, p0, p1)}]"
            chunks.append(
                Chunk(
                    chunk_id=f"{spec.doc_id}-{len(chunks):04d}",
                    doc_id=spec.doc_id,
                    doc_title=spec.title,
                    doc_type=spec.doc_type,
                    cite_label=spec.cite_label,
                    section_path=list(sec.path),
                    section_kind=sec.kind,
                    searchable=searchable,
                    page_start=ps,
                    page_end=pe,
                    page_label_start=first.label_start,
                    para_start=p0,
                    para_end=p1,
                    prev_id=None,
                    next_id=None,
                    is_footnote_only=all(p.kind == "footnote" for p in paras),
                    header=header,
                    text=text,
                    n_tokens=count(text),
                    overlap_tokens=ov_t,
                    overlap_chars=ov_c,
                    rule_refs=rule_refs(text),
                    citation=cite,
                )
            )
    # prev/next chain over searchable chunks only (neighbours never land in a TOC/TOA)
    chain = [c for c in chunks if c.searchable]
    for a, b in zip(chain, chain[1:]):
        a.next_id, b.prev_id = b.chunk_id, a.chunk_id
    return chunks


def write_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
