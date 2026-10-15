"""Embedders behind one interface: `.name` and `.encode(texts) -> L2-normalised float32 array`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from q2_embedder_final import config
from q2_embedder_final.preprocess import lexical_tokens


class DenseEmbedder:
    """Sentence-transformers model (the off-the-shelf baseline, or the fine-tuned checkpoint)."""

    def __init__(self, name: str, model_or_path="", max_seq_len: int = config.MAX_SEQ_LEN):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.model = model_or_path if not isinstance(model_or_path, (str, Path)) else SentenceTransformer(
            str(model_or_path), device="cpu")
        self.model.max_seq_length = max_seq_len

    def encode(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False).astype(np.float32)


class LSAEmbedder:
    """Trained from scratch on the corpus: legal tokens -> TF-IDF (1-2 grams) -> truncated SVD."""

    def __init__(self, name: str = "custom-lsa", dim: int = config.LSA_DIM, seed: int = config.SEED):
        self.name, self.dim, self.seed = name, dim, seed
        self.tfidf = TfidfVectorizer(tokenizer=lexical_tokens, lowercase=False, token_pattern=None,
                                     ngram_range=(1, 2), min_df=2, max_df=0.6, sublinear_tf=True)
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        X = self.tfidf.fit_transform(texts)
        dim = min(self.dim, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=dim, random_state=self.seed).fit(X)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return normalize(self.svd.transform(self.tfidf.transform(texts))).astype(np.float32)

    @property
    def vocab_size(self) -> int:
        return len(self.tfidf.vocabulary_)


def embed_all(embedders: dict[str, object], texts: list[str], cache_dir: Path | None = None) -> dict[str, np.ndarray]:
    out = {}
    for key, emb in embedders.items():
        out[key] = emb.encode(texts)
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_dir / f"embeddings_{key}.npy", out[key])
    return out
