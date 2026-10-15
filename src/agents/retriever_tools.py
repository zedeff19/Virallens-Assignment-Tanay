"""Retriever-local search tool: same ranking as `src.tools.search_corpus`, compact output.

Why a wrapper: on Groq's free tier a request over ~6K tokens is rejected (413) and the retriever's
context grows by every search result. Full 300-token passages x 4 hits x 3 searches blew through that
on one task, which pushed it onto the fallback models. Here each hit carries a ~450-character preview
centred on the query terms; `save_evidence` still copies the *full* passage by chunk_id, so nothing
is lost from the evidence file.
"""
from __future__ import annotations

import json
import re

from langchain_core.tools import tool

from src.retrieval.tokenize import tokenize
from src.tools import search_corpus as _search_corpus

K = 3
PREVIEW_CHARS = 450
_STEP = 80


def preview(text: str, query: str, width: int = PREVIEW_CHARS) -> str:
    """`text` if short, else the `width`-char window (word-aligned) with the most query-term overlap.

    Ties go to the earliest window, so a chunk with no lexical overlap shows its opening.
    """
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    terms = set(tokenize(query))
    best_start, best_score = 0, -1
    for start in range(0, len(flat) - width + _STEP, _STEP):
        start = min(start, len(flat) - width)
        score = len(terms & set(tokenize(flat[start : start + width])))
        if score > best_score:
            best_start, best_score = start, score
        if start == len(flat) - width:
            break
    start, end = best_start, best_start + width
    if start > 0:
        m = re.search(r"\s", flat[start:])
        start = start + m.end() if m else start
    if end < len(flat):
        cut = flat.rfind(" ", start, end)
        end = cut if cut > start else end
    return ("… " if start > 0 else "") + flat[start:end] + (" …" if end < len(flat) else "")


@tool("search_corpus")
def retriever_search(query: str, doc_id: str | None = None, section: str | None = None) -> str:
    """Hybrid semantic + keyword search over the legal corpus; returns the 3 best passages as previews.

    Args:
        query: What to look for, in the documents' own words. Exact strings such as "Item 303" or
            "10b-5(b)" are matched literally.
        doc_id: Optional. One of macquarie, fb_petitioners, fb_amicus, last_atlantis, sec_chiueh.
        section: Optional. Heading text to restrict to, e.g. "ARGUMENT", "Rule 10b-5(a)", "FACTS".

    Returns:
        JSON {"hits": [{chunk_id, citation, doc_id, section, preview}]}, best first. `preview` is an
        excerpt: save_evidence stores the full passage by chunk_id. Search always returns something;
        judge relevance from the text.
    """
    raw = json.loads(_search_corpus(query, doc_id=doc_id, section=section, k=K))
    if "hits" not in raw:
        return json.dumps(raw, ensure_ascii=False)  # unknown doc_id etc.: pass the error through
    hits = [
        {
            "chunk_id": h["chunk_id"],
            "citation": h["citation"],
            "doc_id": h["doc_id"],
            "section": h["section"],
            "preview": preview(h["text"], query),
        }
        for h in raw["hits"]
    ]
    out: dict = {"hits": hits}
    if "note" in raw:
        out["note"] = raw["note"]
    return json.dumps(out, ensure_ascii=False)
