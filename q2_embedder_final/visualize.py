"""Static figures (PNG). Colours: validated categorical slots 1-3 (blue, orange, aqua) identify the
three models in every chart; documents on the cluster map use marker shape as a second channel."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from q2_embedder_final import config  # noqa: E402

MODEL_COLORS = {"baseline": "#2a78d6", "lsa": "#eb6834", "finetuned": "#1baf7a", "finetuned_qa": "#6250d6"}
DOC_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"]
DOC_MARKERS = ["o", "s", "^", "D", "P"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 10, "legend.frameon": False,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_separation_bars(results: dict[str, dict], path: Path, title: str) -> Path:
    """Fig 1 - clustering separation: silhouette vs doc labels, vs case labels, and k-means ARI."""
    metrics = [("silhouette_doc", "Silhouette\n(true doc labels)"), ("silhouette_case", "Silhouette\n(true case labels)"),
               ("ari", "ARI of k-means\n(k = #docs)")]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.26
    for m, key in enumerate(results):
        vals = [results[key]["silhouette_doc"], results[key]["silhouette_case"], results[key]["kmeans_k_docs"]["ARI"]]
        xs = np.arange(len(metrics)) + (m - (len(results) - 1) / 2) * (w + 0.02)  # centred for any #models
        bars = ax.bar(xs, vals, w, color=MODEL_COLORS[key], label=config.NAMES[key], edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + (0.01 if v >= 0 else -0.04), f"{v:.2f}", ha="center", fontsize=8, color=MUTED)
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_xticks(range(len(metrics)), [m[1] for m in metrics])
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_maps(coords: dict[str, np.ndarray], doc_ids: np.ndarray, path: Path, title: str) -> Path:
    """Fig 2 - t-SNE cluster maps, one panel per model, coloured + shaped by source document."""
    docs = list(dict.fromkeys(doc_ids))
    fig, axes = plt.subplots(1, len(coords), figsize=(4.6 * len(coords), 4.4), sharex=False, sharey=False)
    for ax, (key, xy) in zip(np.atleast_1d(axes), coords.items()):
        for d, doc in enumerate(docs):
            m = doc_ids == doc
            ax.scatter(xy[m, 0], xy[m, 1], s=28, color=DOC_COLORS[d % 5], marker=DOC_MARKERS[d % 5],
                       edgecolor=SURFACE, linewidth=0.8, label=doc)
        ax.set_title(config.NAMES[key], loc="left", fontsize=10, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(docs), fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    return _save(fig, path)


def plot_similarity_hist(sims: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str) -> Path:
    """Fig 3 - cosine similarity of same-document vs different-document chunk pairs."""
    fig, axes = plt.subplots(1, len(sims), figsize=(4.4 * len(sims), 3.6), sharey=True)
    for ax, (key, (same, diff)) in zip(np.atleast_1d(axes), sims.items()):
        lo, hi = min(same.min(), diff.min()), max(same.max(), diff.max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(diff, bins, color="#898781", alpha=0.75, density=True, label="different document")
        ax.hist(same, bins, color=MODEL_COLORS[key], alpha=0.75, density=True, label="same document")
        ax.set_title(f"{config.NAMES[key]}\ngap {same.mean() - diff.mean():.2f}", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("cosine similarity")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(title, x=0.01, y=1.06, ha="left", fontsize=11, color=INK)  # clear of the 2-line panel titles
    return _save(fig, path)


def plot_k_sweep(results: dict[str, dict], path: Path, title: str) -> Path:
    """Fig 4 - silhouette of k-means labels as k grows (is there a natural number of clusters?)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for key, r in results.items():
        ks = sorted(r["k_sweep"])
        ax.plot(ks, [r["k_sweep"][k] for k in ks], color=MODEL_COLORS[key], linewidth=2, marker="o",
                markersize=5, markeredgecolor=SURFACE, label=config.NAMES[key])
    ax.set_xlabel("number of clusters k"); ax.set_ylabel("silhouette (cosine)")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8)
    return _save(fig, path)


# ------------------------------------------------------------------ retrieval figures (synthetic-question approach)
def plot_recall_bars(summaries: dict[str, dict], path: Path, title: str) -> Path:
    """Recall@k of (question -> source chunk) retrieval, one bar group per k, one bar per model."""
    ks = list(config.QA_RECALL_KS)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.26
    for m, (key, s) in enumerate(summaries.items()):
        vals = [s[f"recall@{k}"] for k in ks]
        xs = np.arange(len(ks)) + (m - (len(summaries) - 1) / 2) * (w + 0.02)
        ax.bar(xs, vals, w, color=MODEL_COLORS[key], label=config.NAMES[key], edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(ks)), [f"Recall@{k}" for k in ks])
    ax.set_ylim(0, 1.08)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_metric_by_type(by_type: dict[str, dict[str, dict]], metric: str, path: Path, title: str) -> Path:
    """`metric` (e.g. 'mrr' or 'recall@1') per question type; by_type[model][qtype] is a `summarize` dict."""
    types = list(next(iter(by_type.values())))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.26
    for m, (key, d) in enumerate(by_type.items()):
        vals = [d[t][metric] for t in types]
        xs = np.arange(len(types)) + (m - (len(by_type) - 1) / 2) * (w + 0.02)
        ax.bar(xs, vals, w, color=MODEL_COLORS[key], label=config.NAMES[key], edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(types)), types)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(metric)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_loss(losses: list[float], path: Path, title: str, window: int = 10) -> Path:
    """Contrastive loss per optimiser step with a moving average (batches are noisy)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    steps = np.arange(1, len(losses) + 1)
    ax.plot(steps, losses, color="#c3c2b7", linewidth=1, label="batch loss")
    if len(losses) >= window:
        ma = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(losses) + 1), ma, color=MODEL_COLORS["finetuned_qa"], linewidth=2,
                label=f"{window}-step average")
    ax.set_xlabel("optimiser step"); ax.set_ylabel("symmetric InfoNCE loss")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8)
    return _save(fig, path)
