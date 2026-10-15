"""Hard invariants and fact probes for the ingest output (used by __main__ and tests)."""
from __future__ import annotations

import re
from collections.abc import Callable

from src.ingest.chunk import Chunk
from src.ingest.docs import DOCS

MAX_EMBED_TOKENS = 512

# Running-header strings that must not survive cleaning in any chunk text.
_HEADER_LEAKS = [
    r"Cite as: 601 U\. S\.",
    r"Opinion of the Court",
    r"Case: 1:04-cv-00397",
    r"Case 2:25-cv-01920",
    r"PageID",
]

# (doc_id, substring): must appear in a *searchable* chunk of that doc.
FACT_PROBES = [
    ("macquarie", "Pure omissions are not actionable under Rule 10b-5(b)"),
    ("macquarie", "half-truth"),
    ("fb_petitioners", "warned-of risk has materialized in"),
    ("fb_amicus", "quintessential half-truth"),
    ("last_atlantis", "shingle theory"),
    ("sec_chiueh", "25% of its total assets in one industry"),
    ("sec_chiueh", "FIRST CLAIM FOR RELIEF"),
    ("sec_chiueh", "ELEVENTH CLAIM FOR RELIEF"),
]


def validate(chunks: list[Chunk], count_tokens: Callable[[str], int] | None = None) -> list[str]:
    problems: list[str] = []
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        problems.append("duplicate chunk ids")
    by_id = {c.chunk_id: c for c in chunks}

    for c in chunks:
        blob = c.text
        if "\xad" in blob or " " in blob:
            problems.append(f"{c.chunk_id}: soft hyphen / hair space left in text")
        if "10b–5" in blob or "10b‑5" in blob:
            problems.append(f"{c.chunk_id}: un-normalised 10b-5 dash")
        for rx in _HEADER_LEAKS:
            if re.search(rx, blob, flags=re.M):
                problems.append(f"{c.chunk_id}: running header leaked ({rx})")
        if c.page_start < 1 or c.page_end < c.page_start:
            problems.append(f"{c.chunk_id}: bad page range")
        if not c.searchable and (c.prev_id or c.next_id):
            problems.append(f"{c.chunk_id}: non-searchable chunk is linked into the chain")
        if c.section_kind in ("toc", "toa") and c.searchable:
            problems.append(f"{c.chunk_id}: toc/toa marked searchable")
        if c.prev_id and by_id[c.prev_id].next_id != c.chunk_id:
            problems.append(f"{c.chunk_id}: prev/next not symmetric")
        if c.next_id and by_id[c.next_id].prev_id != c.chunk_id:
            problems.append(f"{c.chunk_id}: next/prev not symmetric")
        if not c.text.strip():
            problems.append(f"{c.chunk_id}: empty text")

    if count_tokens is not None:
        for c in chunks:
            total = count_tokens(c.header + "\n" + c.text) + 2
            if total > MAX_EMBED_TOKENS:
                problems.append(f"{c.chunk_id}: {total} tokens incl. header > {MAX_EMBED_TOKENS}")

    mac = [c for c in chunks if c.doc_id == "macquarie"]
    if not any("10b-5" in c.text for c in mac):
        problems.append("macquarie: no '10b-5' found (dash normalisation failed?)")

    # complaint paragraph numbers 1..N contiguous across chunks
    chi = [c for c in chunks if c.doc_id == "sec_chiueh" and c.para_start is not None]
    seen = sorted({n for c in chi for n in range(c.para_start, c.para_end + 1)})
    if seen:
        missing = sorted(set(range(1, seen[-1] + 1)) - set(seen))
        if missing:
            problems.append(f"sec_chiueh: missing paragraph numbers {missing[:10]}")

    for doc_id, needle in FACT_PROBES:
        hits = [
            c for c in chunks if c.doc_id == doc_id and c.searchable and needle in (c.header + "\n" + c.text)
        ]
        if not hits:
            problems.append(f"fact probe failed: [{doc_id}] {needle!r}")
    for doc_id in DOCS:
        if not any(c.doc_id == doc_id for c in chunks):
            problems.append(f"no chunks for {doc_id}")
    return problems
