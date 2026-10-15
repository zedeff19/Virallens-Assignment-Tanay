"""Hybrid retriever: dense + BM25 merged by reciprocal-rank fusion, with doc_id/section filters.

    python -m src.retrieval.hybrid "pure omission half-truth"
    python -m src.retrieval.hybrid "risk factors" --doc fb_amicus --section ARGUMENT -k 4

The corpus is a few hundred chunks, so both rankers score every filtered candidate exactly (no ANN
approximation): dense = cosine over the embeddings persisted in Chroma, lexical = BM25 over the same
`embed_text` the index was built from. Filters are applied *before* ranking, so a filtered search
always returns the best k chunks inside the filter. Non-searchable chunks (TOC / table of
authorities) are excluded unless `section` explicitly names them.

`Hit.dense_score` is the raw cosine similarity. RRF scores are rank-based, so they always look "fine";
cosine is not a reliable "nothing relevant" signal either (an unanswerable question about the final
Facebook decision still scored 0.75 against the briefs' conclusions). Decide answerability from the
passages' content, not from a score threshold.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from src.ingest.chunk import read_jsonl
from src.ingest.docs import CHUNKS_PATH, DOCS
from src.ingest.index import BM25_PATH, CHROMA_DIR, COLLECTION, EMBED_MODEL, QUERY_PREFIX
from src.retrieval.tokenize import tokenize

RRF_K = 60  # standard RRF constant: score = sum(1 / (RRF_K + rank))
POOL = 30  # per-ranker depth that goes into fusion
DEFAULT_K = 6
MAX_K = 20

# `section` values that name a section *kind* rather than a heading.
_SECTION_KINDS = {
    "table_of_authorities": "toa",
    "table of authorities": "toa",
    "toa": "toa",
    "table_of_contents": "toc",
    "table of contents": "toc",
    "toc": "toc",
    "appendix": "appendix",
}


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    doc_title: str
    citation: str  # "[Macquarie, p.5]" / "[Chiueh ¶45]"
    section: str  # full section path, "Opinion > II"
    page_start: int
    page_end: int
    para_start: int | None
    para_end: int | None
    prev_id: str | None
    next_id: str | None
    text: str
    score: float  # RRF score (rank-based)
    dense_score: float  # cosine similarity of the query to the chunk
    dense_rank: int | None  # 1-based rank in the dense list, None if outside the top POOL
    bm25_score: float
    bm25_rank: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def _section_matcher(section: str):
    """Case-insensitive match against the section path; short values ("II", "A") must be whole tokens."""
    s = section.strip().lower()
    kind = _SECTION_KINDS.get(s)
    pattern = re.compile(r"(?<![a-z0-9])" + re.escape(s) + (r"(?![a-z0-9])" if len(s) <= 4 else ""))

    def match(row: dict) -> bool:
        if kind is not None and row["section_kind"] == kind:
            return True
        return bool(pattern.search(" > ".join(row["section_path"]).lower()))

    return match, kind in ("toa", "toc")


class HybridRetriever:
    def __init__(self) -> None:
        with BM25_PATH.open("rb") as f:
            store = pickle.load(f)
        if store.get("model") != EMBED_MODEL:
            raise RuntimeError(f"index built with {store.get('model')!r}, expected {EMBED_MODEL!r}: run `python -m src.ingest.index`")
        by_id = {r["chunk_id"]: r for r in read_jsonl(CHUNKS_PATH)}
        self._ids: list[str] = store["ids"]
        if set(self._ids) != set(by_id):
            raise RuntimeError("index is out of date with chunks.jsonl: run `python -m src.ingest.index`")
        self._rows = [by_id[i] for i in self._ids]
        self._bm25 = store["bm25"]
        self._emb = self._load_embeddings()
        self._model = None  # sentence-transformers is slow to import/load: do it on the first query

    def _load_embeddings(self) -> np.ndarray:
        import chromadb

        col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
        got = col.get(ids=self._ids, include=["embeddings"])
        pos = {cid: i for i, cid in enumerate(got["ids"])}
        return np.asarray(got["embeddings"], dtype=np.float32)[[pos[i] for i in self._ids]]

    def _embed_query(self, query: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMBED_MODEL, device="cpu")
        vec = self._model.encode([QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vec, dtype=np.float32)

    def _candidates(self, doc_id: str | None, section: str | None) -> np.ndarray:
        include_unsearchable = False
        match = None
        if section and section.strip():
            match, include_unsearchable = _section_matcher(section)
        keep = []
        for i, row in enumerate(self._rows):
            if doc_id is not None and row["doc_id"] != doc_id:
                continue
            if not row["searchable"] and not include_unsearchable:
                continue
            if match is not None and not match(row):
                continue
            keep.append(i)
        return np.asarray(keep, dtype=np.int64)

    def search(self, query: str, doc_id: str | None = None, section: str | None = None, k: int = DEFAULT_K) -> list[Hit]:
        """Top-k chunks for `query`, best first. Empty list if the query is blank or the filter matches nothing."""
        query = query.strip()
        if doc_id is not None and doc_id not in DOCS:
            raise ValueError(f"unknown doc_id {doc_id!r}; expected one of {sorted(DOCS)}")
        k = max(1, min(int(k), MAX_K))
        cand = self._candidates(doc_id, section)
        if not query or len(cand) == 0:
            return []

        dense = self._emb[cand] @ self._embed_query(query)
        bm25 = np.asarray(self._bm25.get_scores(tokenize(query)))[cand]

        dense_rank = {int(j): r for r, j in enumerate(np.argsort(-dense, kind="stable")[:POOL], 1)}
        # chunks sharing no term with the query score 0: they carry no lexical signal, keep them out of fusion
        bm25_order = [int(j) for j in np.argsort(-bm25, kind="stable")[:POOL] if bm25[j] > 0]
        bm25_rank = {j: r for r, j in enumerate(bm25_order, 1)}

        fused: dict[int, float] = defaultdict(float)
        for ranks in (dense_rank, bm25_rank):
            for j, r in ranks.items():
                fused[j] += 1.0 / (RRF_K + r)

        top = sorted(fused, key=lambda j: (-fused[j], -float(dense[j])))[:k]
        hits = []
        for j in top:
            row = self._rows[int(cand[j])]
            hits.append(
                Hit(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    doc_title=row["doc_title"],
                    citation=row["citation"],
                    section=" > ".join(row["section_path"]),
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    para_start=row.get("para_start"),
                    para_end=row.get("para_end"),
                    prev_id=row.get("prev_id"),
                    next_id=row.get("next_id"),
                    text=row["text"],
                    score=fused[j],
                    dense_score=float(dense[j]),
                    dense_rank=dense_rank.get(j),
                    bm25_score=float(bm25[j]),
                    bm25_rank=bm25_rank.get(j),
                )
            )
        return hits


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


def search(query: str, doc_id: str | None = None, section: str | None = None, k: int = DEFAULT_K) -> list[Hit]:
    return get_retriever().search(query, doc_id=doc_id, section=section, k=k)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid search over the indexed corpus")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--doc", dest="doc_id", choices=sorted(DOCS))
    ap.add_argument("--section")
    ap.add_argument("-k", type=int, default=DEFAULT_K)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    query = " ".join(args.query)
    hits = search(query, doc_id=args.doc_id, section=args.section, k=args.k)
    print(f"{len(hits)} hits for {query!r} (doc={args.doc_id}, section={args.section})")
    for n, h in enumerate(hits, 1):
        print(
            f"\n{n}. {h.chunk_id}  {h.citation}  rrf={h.score:.4f}  cos={h.dense_score:.3f}"
            f"  dense#{h.dense_rank}  bm25#{h.bm25_rank}\n   {h.section}\n   {' '.join(h.text.split())[:200]}"
        )


if __name__ == "__main__":
    _main()
