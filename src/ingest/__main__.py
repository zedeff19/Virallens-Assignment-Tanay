"""`python -m src.ingest`: PDFs -> data/processed/{chunks.jsonl, citations.json, ingest_report.md}."""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

from src.ingest.chunk import Chunk, build_chunks, get_token_counter, write_jsonl
from src.ingest.clean import build_hyphen_vocab
from src.ingest.docs import (
    CHUNKS_PATH,
    CITATIONS_PATH,
    DOCS,
    PREVIEW_PATH,
    PROCESSED_DIR,
    REPORT_PATH,
)
from src.ingest.load import Page, load_pdf
from src.ingest.structure import Section, build_sections
from src.ingest.validate import validate


def run_pipeline():
    """Return (chunks, sections_by_doc, pages_by_doc, token_counter, counter_name)."""
    pages = {doc_id: load_pdf(spec) for doc_id, spec in DOCS.items()}
    vocab = build_hyphen_vocab(ln.text for pgs in pages.values() for pg in pgs for ln in pg.lines)
    count, counter_name = get_token_counter()
    chunks: list[Chunk] = []
    sections: dict[str, list[Section]] = {}
    for doc_id, spec in DOCS.items():
        sections[doc_id] = build_sections(spec, pages[doc_id], vocab)
        chunks.extend(build_chunks(spec, sections[doc_id], count))
    return chunks, sections, pages, count, counter_name


def build_report(chunks: list[Chunk], pages: dict[str, list[Page]], counter_name: str) -> str:
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)
    lines = [
        "# Ingest report",
        "",
        f"Token counter: `{counter_name}`. Chunks: **{len(chunks)}** "
        f"({sum(c.searchable for c in chunks)} searchable).",
        "",
        "| doc | pdf pages | raw chars | kept chars | kept % | chunks | searchable | tokens min/med/max | overlap chunks |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for doc_id, cs in by_doc.items():
        raw = sum(p.raw_chars for p in pages[doc_id])
        kept = sum(len(c.text) - c.overlap_chars for c in cs)
        toks = [c.n_tokens for c in cs]
        lines.append(
            f"| {doc_id} | {len(pages[doc_id])} | {raw:,} | {kept:,} | {100 * kept / raw:.0f}% | {len(cs)} | "
            f"{sum(c.searchable for c in cs)} | {min(toks)}/{int(statistics.median(toks))}/{max(toks)} | "
            f"{sum(1 for c in cs if c.overlap_tokens)} |"
        )
    return "\n".join(lines) + "\n"


def build_preview(chunks: list[Chunk], n_per_doc: int = 8) -> str:
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if c.searchable:
            by_doc[c.doc_id].append(c)
    out = ["# Chunk preview (sampled)\n"]
    for doc_id, cs in by_doc.items():
        step = max(1, len(cs) // n_per_doc)
        picks = cs[::step][:n_per_doc] + [cs[-1]]
        out.append(f"\n## {doc_id} ({len(cs)} searchable chunks)\n")
        for c in picks:
            out.append(
                f"### {c.chunk_id} | {c.citation} | {c.n_tokens} tok | overlap {c.overlap_tokens}\n"
                f"_{c.header}_\n\n{c.text}\n"
            )
    return "\n".join(out)


def main() -> int:
    chunks, _sections, pages, count, counter_name = run_pipeline()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(chunks, CHUNKS_PATH)

    citations = None
    try:
        from src.retrieval.citations import build_citation_index
    except ImportError:
        build_citation_index = None
    if build_citation_index is not None:
        citations = build_citation_index(chunks)
        CITATIONS_PATH.write_text(json.dumps(citations, ensure_ascii=False, indent=1), encoding="utf-8")

    REPORT_PATH.write_text(build_report(chunks, pages, counter_name), encoding="utf-8")
    PREVIEW_PATH.write_text(build_preview(chunks), encoding="utf-8")
    print(REPORT_PATH.read_text(encoding="utf-8"))
    if citations is not None:
        print(f"citation entries: {len(citations)}")

    problems = validate(chunks, count)
    if problems:
        print("VALIDATION PROBLEMS:")
        for p in problems:
            print(" -", p)
        return 1
    print("validation: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
