"""Traditional clustering on L2-normalised embeddings (k-means, agglomerative) + 2-D projection."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from q2_embedder_final import config


def kmeans(X: np.ndarray, k: int, seed: int = config.SEED) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)


def agglomerative(X: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)


def sweep_k(X: np.ndarray, ks=config.K_SWEEP, seed: int = config.SEED) -> dict[int, float]:
    """Silhouette (cosine) of k-means labels for each k; the elbow-free way to pick k."""
    ks = [k for k in ks if 2 <= k < len(X)]
    return {k: float(silhouette_score(X, kmeans(X, k, seed), metric="cosine")) for k in ks}


def project_2d(X: np.ndarray, seed: int = config.SEED) -> np.ndarray:
    perplexity = max(2, min(15, (len(X) - 1) // 3))
    return TSNE(n_components=2, metric="cosine", init="random", perplexity=perplexity,
                random_state=seed).fit_transform(X)
