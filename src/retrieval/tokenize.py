"""Lexical tokenizer shared by the BM25 index (built at ingest time) and the retriever (query time).

Legal text is full of exact strings ("10b-5(b)", "Item 303", "17(a)(2)"). Each such compound is
emitted both whole and split into its parts, so a query for "Rule 10b-5(b)" and a chunk that
only says "10b-5" (or "10b5") still overlap, and a stray end-of-line hyphen error cannot hide a word.
"""
from __future__ import annotations

import re

_COMPOUND = re.compile(r"[a-z0-9]+(?:[-'’][a-z0-9]+)*(?:\([a-z0-9]+\))*(?:-[a-z0-9]+)?")
_PARTS = re.compile(r"[a-z0-9]+")
_STOP = {"the", "of", "a", "an", "and", "or", "to", "in", "is", "are", "was", "were", "that", "for", "on", "by", "as", "at"}


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("–", "-").replace("—", " ")
    out: list[str] = []
    for m in _COMPOUND.finditer(text):
        tok = m.group(0)
        parts = _PARTS.findall(tok)
        if len(parts) > 1:
            out.append(tok)  # whole compound, e.g. "10b-5(b)"
        out.extend(p for p in parts if p not in _STOP)
    return out
