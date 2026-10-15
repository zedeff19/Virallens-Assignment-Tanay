"""Build the persisted vector index (bge-small-en-v1.5) and BM25 index from chunks.jsonl.

    python -m src.ingest.index            # build index/chroma and index/bm25.pkl
    python -m src.ingest.index "query"    # build if missing, then print the top hits (smoke test)

All chunks are indexed (including TOC/TOA); `searchable` is stored as metadata and the retriever
filters on it, so the table of authorities never competes with real passages by default.
"""
from __future__ import annotations

import pickle
import sys

from src.ingest.chunk import read_jsonl
from src.ingest.docs import CHUNKS_PATH, INDEX_DIR
from src.retrieval.tokenize import tokenize

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # query side only
COLLECTION = "corpus"
BM25_PATH = INDEX_DIR / "bm25.pkl"
CHROMA_DIR = INDEX_DIR / "chroma"


def embed_text(row: dict) -> str:
    """What gets embedded / BM25-indexed: the one-line header plus the chunk body."""
    return f"{row['header']}\n{row['text']}"


def _metadata(row: dict) -> dict:
    """Chroma accepts only scalar metadata: flatten lists, drop Nones."""
    meta = {
        "doc_id": row["doc_id"],
        "doc_title": row["doc_title"],
        "doc_type": row["doc_type"],
        "cite_label": row["cite_label"],
        "section_path": " > ".join(row["section_path"]),
        "section": row["section_path"][-1] if row["section_path"] else "",
        "section_kind": row["section_kind"],
        "searchable": bool(row["searchable"]),
        "page_start": int(row["page_start"]),
        "page_end": int(row["page_end"]),
        "page_label_start": str(row["page_label_start"]),
        "is_footnote_only": bool(row["is_footnote_only"]),
        "rule_refs": ", ".join(row["rule_refs"]),
        "citation": row["citation"],
    }
    for key in ("para_start", "para_end", "prev_id", "next_id"):
        if row.get(key) is not None:
            meta[key] = row[key]
    return meta


def build_index() -> None:
    import chromadb
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    rows = read_jsonl(CHUNKS_PATH)
    texts = [embed_text(r) for r in rows]
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # collection did not exist yet
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[r["chunk_id"] for r in rows],
        embeddings=emb.tolist(),
        documents=texts,
        metadatas=[_metadata(r) for r in rows],
    )

    bm25 = BM25Okapi([tokenize(t) for t in texts])
    with BM25_PATH.open("wb") as f:
           pickle.dump({"ids": [r["chunk_id"] for r in rows], "bm25": bm25, "model": EMBED_MODEL}, f)
    print(f"indexed {len(rows)} chunks -> {CHROMA_DIR} and {BM25_PATH}")


def smoke_query(query: str, k: int = 5) -> None:
    """Dense + BM25 top-k side by side (the real hybrid/RRF retriever lives in retrieval/hybrid.py)."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    with BM25_PATH.open("rb") as f:
        store = pickle.load(f)
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0].tolist()
    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
    dense = col.query(query_embeddings=[q], n_results=k, where={"searchable": True})
    print(f"\nDENSE top {k} for: {query!r}")
    for cid, meta in zip(dense["ids"][0], dense["metadatas"][0]):
        print(f"  {cid:<22} {meta['citation']:<32} {meta['section'][:60]}")
    scores = store["bm25"].get_scores(tokenize(query))
    searchable = {r["chunk_id"] for r in read_jsonl(CHUNKS_PATH) if r["searchable"]}
    ranked = [i for i in sorted(range(len(scores)), key=lambda i: -scores[i]) if store["ids"][i] in searchable]
    print(f"BM25 top {k}")
    for i in ranked[:k]:
        print(f"  {store['ids'][i]:<22} score={scores[i]:.2f}")


if __name__ == "__main__":
    if not BM25_PATH.exists() or len(sys.argv) == 1:
        build_index()
    if len(sys.argv) > 1:
        smoke_query(" ".join(sys.argv[1:]))
