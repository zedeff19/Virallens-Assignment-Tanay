"""Agent tools: search_corpus(query, doc_id?, section?, k=6), get_neighbors(chunk_id), find_citations(case_name)."""
from __future__ import annotations

import json
from functools import lru_cache

from src.ingest.chunk import read_jsonl
from src.ingest.docs import CHUNKS_PATH, CITATIONS_PATH, DOCS
from src.retrieval import hybrid
from src.retrieval.citations import find_citations as _lookup

_MAX_MATCHES = 5
_MAX_CHUNKS = 12
_MAX_K = 10  # tool-level cap: each chunk is ~400 tokens and the free tiers have small TPM limits
_MAX_WINDOW = 3


@lru_cache(maxsize=1)
def _citation_index() -> list[dict]:
    return json.loads(CITATIONS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _chunks_by_id() -> dict[str, dict]:
    return {c["chunk_id"]: c for c in read_jsonl(CHUNKS_PATH)}


def _chunk_labels() -> dict[str, str]:
    """chunk_id -> "[Macquarie, p.5]"-style citation label."""
    return {cid: c.get("citation", "") for cid, c in _chunks_by_id().items()}


def _passage(chunk: dict) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "citation": chunk["citation"],
        "doc_id": chunk["doc_id"],
        "section": " > ".join(chunk["section_path"]),
        "text": chunk["text"],
    }


def _none_if_blank(value: str | None) -> str | None:
    """Small models send "", "null" or "none" for an omitted optional argument."""
    if value is None or value.strip().lower() in ("", "null", "none"):
        return None
    return value.strip()


def search_corpus(query: str, doc_id: str | None = None, section: str | None = None, k: int = 6) -> str:
    """Search the legal corpus (hybrid semantic + keyword) and return the best-matching passages.

    Args:
        query: What to look for. Natural language works; exact strings such as "Item 303" or
            "10b-5(b)" are matched literally as well.
        doc_id: Optional. Restrict to one document: macquarie, fb_petitioners, fb_amicus,
            last_atlantis or sec_chiueh. Omit to search all documents.
        section: Optional. Restrict to sections whose heading contains this text (case-insensitive),
            e.g. "ARGUMENT", "Rule 10b-5(a)", "FACTS". Use "table_of_authorities" to search the tables of authorities, which are excluded by default.
        k: Number of passages to return (default 6, max 10).

    Returns:
        JSON: {"query", "doc_id", "section", "hits": [...]}. Each hit has chunk_id, citation (use it verbatim when citing, e.g. "[Macquarie, p.5]"), doc_id, section and the passage text. Hits are best-first. A result is always returned even when nothing in the corpus answers the question, so judge relevance from the text; an empty list means the filters matched nothing.
    """
    doc_id, section = _none_if_blank(doc_id), _none_if_blank(section)
    if doc_id is not None and doc_id not in DOCS:
        return json.dumps({"error": f"unknown doc_id {doc_id!r}", "valid_doc_ids": sorted(DOCS)})
    hits = hybrid.search(query, doc_id=doc_id, section=section, k=min(int(k), _MAX_K))
    out = {"query": query, "doc_id": doc_id, "section": section, "hits": [_passage(_chunks_by_id()[h.chunk_id]) for h in hits]}
    if not hits:
        out["note"] = "no passages matched; check the section filter or retry without doc_id/section"
    return json.dumps(out, ensure_ascii=False)


def get_neighbors(chunk_id: str, window: int = 1) -> str:
    """Get the passages immediately before and after a chunk, to read it in context.

    Args:
        chunk_id: A chunk_id returned by search_corpus or find_citations, e.g. "macquarie-0012".
        window: How many chunks to fetch on each side (default 1, max 3).

    Returns:
        JSON: {"chunk_id", "before": [...], "after": [...]}. `before` is in reading order ending at the
        chunk; `after` starts right after it. Each passage has chunk_id, citation, doc_id, section, text.
        Neighbours stay within the same document and never include table-of-contents/authorities chunks.
    """
    chunks = _chunks_by_id()
    if chunk_id not in chunks:
        return json.dumps({"error": f"unknown chunk_id {chunk_id!r}", "hint": "use a chunk_id from search_corpus"})
    window = max(1, min(int(window), _MAX_WINDOW))

    def walk(start: str, link: str) -> list[dict]:
        found: list[dict] = []
        cur = chunks[start].get(link)
        while cur and cur in chunks and len(found) < window:
            found.append(_passage(chunks[cur]))
            cur = chunks[cur].get(link)
        return found

    return json.dumps(
        {"chunk_id": chunk_id, "before": walk(chunk_id, "prev_id")[::-1], "after": walk(chunk_id, "next_id")},
        ensure_ascii=False,
    )


def find_citations(case_name: str) -> str:
    """Find where a case, statute, or rule is cited in the corpus.

    Args:
        case_name: A case name, short name or reporter cite (e.g. "Macquarie", "Basic v. Levinson",
            "485 U.S. 224"), or a rule/item/section reference (e.g. "Item 303"). Case-insensitive,
            substring match.

    Returns:
        JSON list of matches, most-cited first. Each has citation_id, kind, case_name, reporter_cite,
        year, aliases, count (number of citing mentions), docs citing it, and `chunks`: the chunks
        (chunk_id + citation label) where it appears. `in_corpus_doc_id` is set when the authority
        is itself one of the corpus documents. Use search_corpus / get_neighbors to read the chunks.
    """
    matches = _lookup(_citation_index(), case_name)
    if not matches:
        return json.dumps({"query": case_name, "matches": [], "note": "no citation found; try a shorter name or the reporter cite"})

    labels = _chunk_labels()
    matches = sorted(matches, key=lambda e: (-e["count"], e["citation_id"]))
    out = []
    for e in matches[:_MAX_MATCHES]:
        ids = e["chunk_ids"]
        out.append(
            {
                "citation_id": e["citation_id"],
                "kind": e["kind"],
                "case_name": e["case_name"],
                "reporter_cite": e["reporter_cite"],
                "year": e["year"],
                "aliases": e["aliases"],
                "count": e["count"],
                "in_corpus_doc_id": e["in_corpus_doc_id"],
                "doc_ids": e["doc_ids"],
                "chunks": [{"chunk_id": i, "citation": labels.get(i, "")} for i in ids[:_MAX_CHUNKS]],
                "chunks_omitted": max(0, len(ids) - _MAX_CHUNKS),
            }
        )
    return json.dumps(
        {"query": case_name, "matches": out, "matches_omitted": max(0, len(matches) - _MAX_MATCHES)},
        ensure_ascii=False,
    )
