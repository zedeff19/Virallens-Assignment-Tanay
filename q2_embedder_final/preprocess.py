"""Preprocessing: load the Q1 chunks, build the evaluation labels / split, and make training pairs.

Nothing here re-parses PDFs: chunking, cleaning and header stripping come from `src.ingest`.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

import numpy as np

from q2_embedder_final import config
from src.ingest.chunk import read_jsonl, split_sentences
from src.retrieval.tokenize import tokenize

_FOOTNOTE = re.compile(r"\[Footnote\]\s*")
_REPORTER = re.compile(
    r"\b\d{1,3}\s+(?:U\.\s?S\.|S\.\s?Ct\.|F\.\s?(?:2d|3d|4th)|F\.\s?Supp\.(?:\s?\d\w+)?|L\.\s?Ed\.\s?2d)\s+\d{1,4}(?:,\s*\d{1,4})*"
)
# Citation / quotation boilerplate that carries no topical signal in court filings.
LEGAL_BOILERPLATE = {
    "see", "id", "cf", "supra", "infra", "cir", "quoting", "internal", "quotation", "marks", "omitted",
    "citation", "citations", "alteration", "emphasis", "added", "accord", "ibid", "cert", "denied",
}


@dataclass
class Corpus:
    chunk_ids: list[str]
    texts: list[str]  # body text only (no header: the header names the document and would leak the label)
    doc_ids: np.ndarray
    case_ids: np.ndarray
    sections: list[str]
    citations: list[str]

    def __len__(self) -> int:
        return len(self.texts)

    def subset(self, idx: np.ndarray) -> "Corpus":
        pick = lambda xs: [xs[i] for i in idx]  # noqa: E731
        return Corpus(pick(self.chunk_ids), pick(self.texts), self.doc_ids[idx], self.case_ids[idx],
                      pick(self.sections), pick(self.citations))


def clean_text(text: str) -> str:
    """Light normalisation for dense models: drop footnote markers, collapse whitespace."""
    return re.sub(r"\s+", " ", _FOOTNOTE.sub("", text)).strip()


def lexical_tokens(text: str) -> list[str]:
    """Legal-aware tokens for the from-scratch model.

    Re-uses the Q1 tokenizer (keeps compounds such as `10b-5(b)` whole *and* split), then drops
    reporter citations, bare numbers, one-letter fragments and citation boilerplate.
    """
    toks = tokenize(_REPORTER.sub(" ", text))
    return [t for t in toks if len(t) > 1 and not t.isdigit() and t not in LEGAL_BOILERPLATE]


def load_corpus(min_tokens: int = config.MIN_CHUNK_TOKENS) -> Corpus:
    rows = [
        r for r in read_jsonl(config.CHUNKS_PATH)
        if r["searchable"] and not r["is_footnote_only"] and r["n_tokens"] >= min_tokens
    ]
    return Corpus(
        chunk_ids=[r["chunk_id"] for r in rows],
        texts=[clean_text(r["text"]) for r in rows],
        doc_ids=np.array([r["doc_id"] for r in rows]),
        case_ids=np.array([config.CASE_OF[r["doc_id"]] for r in rows]),
        sections=[" > ".join(r["section_path"]) for r in rows],
        citations=[r["citation"] for r in rows],
    )


def block_split(corpus: Corpus, block: int = config.BLOCK_SIZE, test_frac: float = config.TEST_FRAC,
                seed: int = config.SEED) -> tuple[np.ndarray, np.ndarray]:
    """Hold out contiguous blocks of chunks per document (>= 1 block per doc)."""
    rng = random.Random(seed)
    train, test = [], []
    for doc in dict.fromkeys(corpus.doc_ids):  # preserves document order
        idx = np.flatnonzero(corpus.doc_ids == doc)
        blocks = [idx[i : i + block] for i in range(0, len(idx), block)]
        n_test = max(1, round(len(blocks) * test_frac))
        test_blocks = set(rng.sample(range(len(blocks)), n_test))
        for b, ids in enumerate(blocks):
            (test if b in test_blocks else train).extend(ids.tolist())
    return np.array(sorted(train)), np.array(sorted(test))


def make_pairs(texts: list[str], idx: np.ndarray, n_aug: int = config.FT_AUGMENTATIONS,
               min_sentences: int = config.FT_MIN_SENTENCES, seed: int = config.SEED):
    """Self-supervised positives: two disjoint random halves of the *same chunk's* sentences.

    No labels are used. Returned as `n_aug` rounds that hold at most one pair per chunk, so
    in-batch negatives never include the pair's own chunk (no false negatives).
    """
    rng = random.Random(seed)
    rounds: list[list[tuple[str, str, int]]] = [[] for _ in range(n_aug)]
    for i in idx:
        sents = split_sentences(texts[i])
        if len(sents) < min_sentences:
            continue
        for r in range(n_aug):
            order = list(range(len(sents)))
            rng.shuffle(order)
            half = len(order) // 2
            a, b = sorted(order[:half]), sorted(order[half:])
            rounds[r].append((" ".join(sents[j] for j in a), " ".join(sents[j] for j in b), int(i)))
    return rounds
