"""Metrics: clustering separation, pairwise-similarity separation, kNN purity, neighbour checks.

All inputs are L2-normalised, so a dot product is the cosine similarity.
`doc_ids` / `case_ids` are *evaluation* labels only; no model is trained with them.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (adjusted_rand_score, homogeneity_score, normalized_mutual_info_score,
                             roc_auc_score, silhouette_score)

from q2_embedder_final import config
from q2_embedder_final.cluster import agglomerative, kmeans, sweep_k


def label_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    return float(silhouette_score(X, labels, metric="cosine"))


def pair_similarities(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of every chunk pair, split into same-label and different-label pairs."""
    S = X @ X.T
    i, j = np.triu_indices(len(X), k=1)
    same = labels[i] == labels[j]
    return S[i, j][same], S[i, j][~same]


def pair_auc(same: np.ndarray, diff: np.ndarray) -> float:
    """P(sim(same-label pair) > sim(different-label pair)); 0.5 = no separation."""
    y = np.r_[np.ones(len(same)), np.zeros(len(diff))]
    return float(roc_auc_score(y, np.r_[same, diff]))


def knn_purity(X: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Mean fraction of a chunk's k nearest neighbours that share its label."""
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    k = min(k, len(X) - 1)
    nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
    return float((labels[nn] == labels[:, None]).mean())


def evaluate_embeddings(X: np.ndarray, doc_ids: np.ndarray, case_ids: np.ndarray) -> dict:
    n_doc, n_case = len(set(doc_ids)), len(set(case_ids))
    same, diff = pair_similarities(X, doc_ids)
    res = {
        "n": int(len(X)), "dim": int(X.shape[1]),
        "silhouette_doc": label_silhouette(X, doc_ids),
        "silhouette_case": label_silhouette(X, case_ids),
        "intra_doc_sim": float(same.mean()), "inter_doc_sim": float(diff.mean()),
        "sim_gap": float(same.mean() - diff.mean()),
        "pair_auc_doc": pair_auc(same, diff),
        "knn_purity_doc": {int(k): knn_purity(X, doc_ids, k) for k in config.KNN_KS},
    }
    km, ag = kmeans(X, n_doc), agglomerative(X, n_doc)
    res["kmeans_k_docs"] = {
        "k": n_doc,
        "silhouette": label_silhouette(X, km),
        "ARI": float(adjusted_rand_score(doc_ids, km)),
        "NMI": float(normalized_mutual_info_score(doc_ids, km)),
        "homogeneity": float(homogeneity_score(doc_ids, km)),
    }
    res["agglomerative_k_docs"] = {
        "k": n_doc,
        "silhouette": label_silhouette(X, ag),
        "ARI": float(adjusted_rand_score(doc_ids, ag)),
        "NMI": float(normalized_mutual_info_score(doc_ids, ag)),
    }
    km_case = kmeans(X, n_case)
    res["kmeans_k_cases"] = {"k": n_case, "ARI": float(adjusted_rand_score(case_ids, km_case)),
                             "NMI": float(normalized_mutual_info_score(case_ids, km_case))}
    sweep = sweep_k(X)
    res["k_sweep"] = {int(k): v for k, v in sweep.items()}
    res["best_k"] = int(max(sweep, key=sweep.get))
    return res


def neighbour_table(queries: list[str], embedders: dict, X_by_model: dict, corpus, top: int = 3) -> str:
    """Markdown table: top-`top` chunks (with citation + snippet) per probe query and model."""
    lines = ["| query | model | rank | chunk | snippet |", "|---|---|---|---|---|"]
    for q in queries:
        for key, emb in embedders.items():
            qv = emb.encode([q])[0]
            sims = X_by_model[key] @ qv
            for r, j in enumerate(np.argsort(-sims)[:top], 1):
                snippet = corpus.texts[j][:110].replace("|", "/") + "…"
                lines.append(f"| {q if r == 1 and key == next(iter(embedders)) else ''} | {config.NAMES[key]} | {r} "
                             f"| {corpus.citations[j]} (sim {sims[j]:.2f}) | {snippet} |")
    return "\n".join(lines)


def results_table(results: dict[str, dict]) -> str:
    cols = [("silhouette_doc", "sil(doc)"), ("silhouette_case", "sil(case)"), ("sim_gap", "intra-inter sim"),
            ("pair_auc_doc", "pair AUC"), ("ari", "ARI@k=docs"), ("nmi", "NMI@k=docs"),
            ("knn5", "kNN-purity@5"), ("best_k", "best k")]
    head = "| model | " + " | ".join(c[1] for c in cols) + " |\n|---|" + "---|" * len(cols)
    rows = []
    for key, r in results.items():
        flat = {**r, "ari": r["kmeans_k_docs"]["ARI"], "nmi": r["kmeans_k_docs"]["NMI"],
                "knn5": r["knn_purity_doc"][5]}
        cells = [f"{flat[c]:.3f}" if isinstance(flat[c], float) else str(flat[c]) for c, _ in cols]
        rows.append(f"| {config.NAMES[key]} | " + " | ".join(cells) + " |")
    return head + "\n" + "\n".join(rows)
