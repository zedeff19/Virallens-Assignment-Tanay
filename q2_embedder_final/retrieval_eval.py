"""Retrieval metrics for (question -> source chunk) evaluation.

All embeddings are L2-normalised, so a dot product is the cosine similarity. A question's *gold* chunk is the
chunk it was written from; `rank` is the 1-based position of that chunk when the pool is sorted by similarity.
"""
from __future__ import annotations

import numpy as np

from q2_embedder_final import config


def gold_ranks(Q: np.ndarray, C: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """1-based rank of each question's gold chunk in pool `C` (`gold` indexes rows of `C`)."""
    sims = Q @ C.T
    g = sims[np.arange(len(Q)), gold]
    return (sims > g[:, None]).sum(axis=1) + 1


def summarize(ranks: np.ndarray, ks=config.QA_RECALL_KS) -> dict:
    out = {"n": int(len(ranks))}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(ranks <= k))
    out["mrr"] = float(np.mean(1.0 / ranks))
    out["median_rank"] = float(np.median(ranks))
    return out


def by_group(ranks: np.ndarray, groups: np.ndarray, ks=config.QA_RECALL_KS) -> dict[str, dict]:
    return {str(g): summarize(ranks[groups == g], ks) for g in dict.fromkeys(groups.tolist())}


def _metric(ranks: np.ndarray, name: str) -> np.ndarray:
    if name == "mrr":
        return 1.0 / ranks
    k = int(name.split("@")[1])
    return (ranks <= k).astype(float)


def cluster_bootstrap_diff(ranks_a: np.ndarray, ranks_b: np.ndarray, chunk_ids: np.ndarray, metric: str = "mrr",
                           n_boot: int = 2000, seed: int = config.SEED) -> dict:
    """Mean of metric(a) - metric(b) with a 95% interval from resampling *chunks* (questions from one chunk
    are correlated, so resampling single questions would understate the uncertainty)."""
    diff = _metric(ranks_a, metric) - _metric(ranks_b, metric)
    chunks = list(dict.fromkeys(chunk_ids.tolist()))
    per_chunk = [diff[chunk_ids == c] for c in chunks]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(chunks), len(chunks))
        means[i] = np.concatenate([per_chunk[j] for j in pick]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)], "n_chunks": len(chunks)}
