"""Strip running headers, soft hyphens and line-break hyphenation; drop cover/counsel blocks.

All functions here are pure so they can be unit-tested on short literals.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from src.ingest.docs import DocSpec
from src.ingest.load import Line, Page

_PAGE_NO = re.compile(r"^\(?(?:[ivxIVX]{1,6}|\d{1,3}a?)\)?$")
_NO_ALNUM = re.compile(r"^[\W_]+$")
_SPACE_CHARS = dict.fromkeys(map(ord, "       "), " ")
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
# Rule 10b-5 is typeset with an EN DASH in Macquarie ("10b–5"); BM25 needs one canonical form.
_RULE_DASH = re.compile(r"\b(10b)\s?[‐-―−]\s?(5)\b")
_HYPHEN_COMPOUND = re.compile(r"[A-Za-z0-9()]+(?:-[A-Za-z0-9()]+)+")
# Left fragments that are almost always the first half of a real compound.
_KEEP_PREFIXES = {"half", "risk", "self", "non", "anti", "quasi"}
_DASHES_NO_SPACE = ("–", "—")  # en/em dash at end of line: join without a space


def normalize_text(text: str) -> str:
    """Unicode/whitespace normalisation. Keeps curly quotes, § and ¶ (quotes are checked verbatim)."""
    text = text.replace("\xad", "").translate(_SPACE_CHARS)
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = _RULE_DASH.sub(r"\1-\2", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def build_hyphen_vocab(texts: Iterable[str]) -> set[str]:
    """Hyphenated compounds seen *inside* lines anywhere in the corpus (lower-cased).

    Used to decide whether an end-of-line hyphen is typesetting ("implau-/sible") or part of
    the word ("half-/truth").
    """
    vocab: set[str] = set()
    for t in texts:
        for m in _HYPHEN_COMPOUND.finditer(t.replace("\xad", "")):
            vocab.add(m.group(0).lower())
    return vocab


def _keep_hyphen(left: str, right: str, vocab: set[str]) -> bool:
    if not left or not right:
        return False
    if left[-1].isdigit() or left[-1] == ")" or right[0].isdigit():
        return True  # 10b-/5(b), 206(4)-/8(a)(1), 80a-/8
    if left.lower() in _KEEP_PREFIXES:
        return True
    cand = f"{left}-{right}".lower().rstrip(".,;:)”\"'")
    return cand in vocab or cand.rstrip("s") in vocab


def join_lines(lines: list[str], vocab: set[str]) -> str:
    """Join hard-wrapped lines into one paragraph, repairing end-of-line hyphenation."""
    out = ""
    for raw in lines:
        t = raw.strip()
        if not t:
            continue
        if not out:
            out = t
        elif out.endswith("\xad"):
            out = out[:-1] + t
        elif out.endswith(_DASHES_NO_SPACE) and len(out) > 1 and out[-2].isalnum() and t[0].isalnum():
            out = out + t
        elif out.endswith("-") and len(out) > 1 and (out[-2].isalnum() or out[-2] in ")]") and t[0].isalnum():
            left = re.split(r"[\s(\[“\"‘']", out[:-1])[-1]
            right = re.split(r"\s", t, maxsplit=1)[0]
            # "cut-and-/dry": the token already carries a hyphen, so the break hyphen is a real one
            keep = "-" in left or _keep_hyphen(left, right, vocab)
            out = out + t if keep else out[:-1] + t
        else:
            out = f"{out} {t}"
    return normalize_text(out)


def _split_page_number(text: str) -> tuple[str, str]:
    m = re.fullmatch(r"(\d{1,3}a?)\s+(\S.*)", text) or re.fullmatch(r"(\S.*?)\s+(\d{1,3})", text)
    if not m:
        return text, ""
    a, b = m.group(1), m.group(2)
    return (b, a) if a[0].isdigit() and re.fullmatch(r"\d{1,3}a?", a) else (a, b)


def clean_page(spec: DocSpec, page: Page) -> tuple[str, list[Line]]:
    """Return (printed page label, body lines) for one page, dropping running headers etc."""
    if page.number in spec.drop_pages:
        return "", []
    height = page.height
    label = ""
    lines: list[Line] = []
    header_res = [re.compile(p) for p in spec.header_res]

    for ln in page.lines:
        text = ln.text.translate(_SPACE_CHARS).strip()
        if ln.marker_only or not text:
            continue
        in_zone = ln.y0 < 0.17 * height or ln.y0 > 0.85 * height
        if spec.page_number_pos != "none" and in_zone and _PAGE_NO.match(text):
            label = label or text.strip("()")
            continue
        if in_zone:
            # A bare page number on the same baseline as a running header is merged into it.
            core, number = _split_page_number(text)
            if number and any(r.fullmatch(core) for r in header_res):
                label = label or number
                continue
        if _NO_ALNUM.match(text) or any(r.fullmatch(text) for r in header_res):
            continue
        ln.text = text
        lines.append(ln)

    start = spec.start_at.get(page.number)
    if start:
        rx = re.compile(start)
        for i, ln in enumerate(lines):
            if rx.search(ln.text):
                lines = lines[i:]
                break
    stop = spec.stop_at.get(page.number)
    if stop:
        rx = re.compile(stop)
        for i, ln in enumerate(lines):
            if rx.search(ln.text):
                lines = lines[:i]
                break

    label = spec.label_prefix.get(page.number, "") + (label or str(page.number))
    for ln in lines:
        ln.label = label
    return label, lines
