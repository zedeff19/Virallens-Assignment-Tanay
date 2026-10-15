"""Q2, approach 3: fine-tune bge-small on synthetic (question, chunk) pairs *mixed with* the self-supervised pairs.

    python -m q2_embedder_final.build_training_data build     # once: validate + filter the authored questions
    python -m q2_embedder_final.main_qa                       # fine-tune, embed, evaluate, write outputs3/heldout/
    python -m q2_embedder_final.main_qa --reuse               # skip training, re-evaluate the saved checkpoint

Training uses the 147 train chunks only: their self-supervised sentence-split pairs (as in `main.py`) plus their
kept questions. Evaluation uses the 49 held-out chunks: retrieval of each held-out question's source chunk
(headline pool = the 49 held-out chunks; secondary pool = all 196 chunks) and the usual clustering metrics.
Compared models: baseline bge-small, the self-supervised fine-tune from `outputs2/heldout`, and the new mixed
fine-tune. `training_data/` must already hold `train.jsonl` / `test.jsonl` / `split.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from q2_embedder_final import config
from q2_embedder_final.cluster import project_2d
from q2_embedder_final.embed import DenseEmbedder
from q2_embedder_final.evaluate import evaluate_embeddings, neighbour_table, pair_similarities, results_table
from q2_embedder_final.preprocess import Corpus, block_split, load_corpus, make_pairs
from q2_embedder_final.retrieval_eval import by_group, cluster_bootstrap_diff, gold_ranks, summarize
from q2_embedder_final.train import finetune_dense
from q2_embedder_final.visualize import (plot_loss, plot_maps, plot_metric_by_type, plot_recall_bars,
                                   plot_separation_bars, plot_similarity_hist)

MAIN = ("baseline", "finetuned", "finetuned_qa")  # models drawn in the figures / clustering table
SHORT = "bge-small"
config.NAMES.update({
    "baseline": f"Baseline ({SHORT})",
    "finetuned": f"Self-supervised FT ({SHORT})",
    "finetuned_qa": f"Questions + self-supervised FT ({SHORT})",
    "baseline_noprefix": f"Baseline ({SHORT}), no query prefix",
    "finetuned_noprefix": f"Self-supervised FT ({SHORT}), no query prefix",
    "finetuned_qa_noprefix": f"Questions + self-supervised FT ({SHORT}), no query prefix",
})


class PrefixedQueries:
    """Wraps an embedder so `encode` prepends the bge query instruction (used for free-text probe queries)."""

    def __init__(self, emb: DenseEmbedder, prefix: str = config.BGE_QUERY_PREFIX):
        self.emb, self.prefix = emb, prefix

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.emb.encode([self.prefix + t for t in texts])


# ------------------------------------------------------------------------------------------------ data
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_qa(corpus: Corpus, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[list[dict], list[dict]]:
    """train/test question rows from `training_data/`, checked against the corpus and the block split."""
    d = config.TRAINING_DATA_DIR
    train_rows, test_rows = read_jsonl(d / "train.jsonl"), read_jsonl(d / "test.jsonl")
    split = json.loads((d / "split.json").read_text(encoding="utf-8"))
    ids = corpus.chunk_ids
    assert split["train_chunk_ids"] == [ids[i] for i in train_idx], "training_data split != block_split(train)"
    assert split["test_chunk_ids"] == [ids[i] for i in test_idx], "training_data split != block_split(test)"
    train_chunks = set(split["train_chunk_ids"])
    assert all(r["chunk_id"] in train_chunks for r in train_rows), "a train question belongs to a held-out chunk"
    assert all(r["chunk_id"] in set(split["test_chunk_ids"]) for r in test_rows), "a test question is not held out"
    return train_rows, test_rows


def question_rounds(rows: list[dict], corpus: Corpus) -> list[list[tuple[str, str, int]]]:
    """Round r holds the r-th kept question of every chunk that has one: <= 1 pair per chunk per round, so a
    chunk is never its own in-batch negative. Questions carry the bge query prefix; chunks do not."""
    idx_of = {c: i for i, c in enumerate(corpus.chunk_ids)}
    per_chunk: dict[str, list[str]] = {}
    for r in rows:
        per_chunk.setdefault(r["chunk_id"], []).append(r["question"])
    n_rounds = max(len(v) for v in per_chunk.values())
    rounds: list[list[tuple[str, str, int]]] = [[] for _ in range(n_rounds)]
    for cid, qs in per_chunk.items():
        i = idx_of[cid]
        for r, q in enumerate(qs):
            rounds[r].append((config.BGE_QUERY_PREFIX + q, corpus.texts[i], i))
    return [r for r in rounds if len(r) >= 2]


# --------------------------------------------------------------------------------------------- evaluation
def evaluate_retrieval(embedders: dict[str, DenseEmbedder], corpus: Corpus, test_idx: np.ndarray,
                       test_rows: list[dict]) -> dict:
    """Retrieval of each held-out question's source chunk. Variants: bge query prefix on / off."""
    idx_of = {c: i for i, c in enumerate(corpus.chunk_ids)}
    gold_corpus = np.array([idx_of[r["chunk_id"]] for r in test_rows])
    pos_in_test = {int(c): p for p, c in enumerate(test_idx)}
    gold_heldout = np.array([pos_in_test[int(g)] for g in gold_corpus])
    qtypes = np.array([r["qtype"] for r in test_rows])
    chunk_ids = np.array([r["chunk_id"] for r in test_rows])
    rt = np.array([r["rt_rank"] for r in test_rows])
    texts = [r["question"] for r in test_rows]

    ranks: dict[str, dict[str, np.ndarray]] = {"heldout": {}, "full": {}}
    X_all: dict[str, np.ndarray] = {}
    for key, emb in embedders.items():
        X_all[key] = emb.encode(corpus.texts)
        for variant, prefix in ((key, config.BGE_QUERY_PREFIX), (f"{key}_noprefix", "")):
            Q = emb.encode([prefix + t for t in texts])
            ranks["full"][variant] = gold_ranks(Q, X_all[key], gold_corpus)
            ranks["heldout"][variant] = gold_ranks(Q, X_all[key][test_idx], gold_heldout)

    out: dict = {"n_questions": len(test_rows), "n_chunks_heldout": int(len(test_idx)), "pools": {}}
    for pool, rk in ranks.items():
        res = {}
        for variant, r in rk.items():
            res[variant] = {
                "all": summarize(r),
                "by_type": by_group(r, qtypes),
                "baseline_rank_le_10_subset": summarize(r[rt <= config.QA_ROUNDTRIP_TOP]),
            }
        out["pools"][pool] = res
    # paired, chunk-clustered bootstrap: does the mixed model beat the baseline / the self-supervised one?
    boots = {}
    for pool in ("heldout", "full"):
        rk = ranks[pool]
        for a, b in (("finetuned_qa", "baseline"), ("finetuned_qa", "finetuned"), ("finetuned", "baseline")):
            for metric in ("mrr", "recall@1", "recall@5"):
                boots[f"{pool}: {a} - {b} [{metric}]"] = cluster_bootstrap_diff(rk[a], rk[b], chunk_ids, metric)
    out["bootstrap_paired_diff"] = boots
    out["_ranks"] = {pool: {k: v.tolist() for k, v in rk.items()} for pool, rk in ranks.items()}
    out["_X_all"] = X_all
    return out


def retrieval_markdown(ret: dict) -> str:
    L = [f"# Retrieval of held-out questions ({ret['n_questions']} questions from {ret['n_chunks_heldout']} "
         "held-out chunks)", "",
         "Gold chunk = the chunk a question was written from. Models never saw these chunks or questions in "
         "training. `no query prefix` rows omit bge's query instruction.", ""]
    ks = config.QA_RECALL_KS
    head = "| model | " + " | ".join(f"R@{k}" for k in ks) + " | MRR | median rank |\n|---|" + "---|" * (len(ks) + 2)

    def table(res: dict, section: str) -> list[str]:
        rows = [head]
        for key, v in res.items():
            s = v[section]
            rows.append(f"| {config.NAMES[key]} | " + " | ".join(f"{s[f'recall@{k}']:.3f}" for k in ks)
                        + f" | {s['mrr']:.3f} | {s['median_rank']:.0f} |")
        return rows

    for pool, title in (("heldout", "Pool = the 49 held-out chunks (headline)"),
                        ("full", "Pool = all 196 chunks (harder; the fine-tuned models saw the 147 train chunks)")):
        res = ret["pools"][pool]
        L += [f"## {title}", ""] + table(res, "all") + [""]
        sub = next(iter(res.values()))["baseline_rank_le_10_subset"]["n"]
        L += [f"### Subset: questions whose gold chunk is in the baseline's top {config.QA_ROUNDTRIP_TOP} "
              f"(n = {sub})", ""] + table(res, "baseline_rank_le_10_subset") + [""]
    res = ret["pools"]["heldout"]
    types = list(res["baseline"]["by_type"])
    L += ["## By question type (held-out pool): MRR / Recall@1", "",
          "| model | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for key in MAIN:
        cells = [f"{res[key]['by_type'][t]['mrr']:.3f} / {res[key]['by_type'][t]['recall@1']:.3f}" for t in types]
        L.append(f"| {config.NAMES[key]} | " + " | ".join(cells) + " |")
    L += ["", "## Paired difference with 95% chunk-bootstrap interval", "",
          "| comparison | mean diff | 95% interval | interval excludes 0 |", "|---|---|---|---|"]
    for name, b in ret["bootstrap_paired_diff"].items():
        lo, hi = b["ci95"]
        L.append(f"| {name} | {b['mean_diff']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {'yes' if lo > 0 or hi < 0 else 'no'} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=config.OUT_DIR3.name, help="relative paths resolve under q2_embedder_final/")
    ap.add_argument("--baseline", default=config.QA_BASELINE_MODEL)
    ap.add_argument("--selfsup-model", default="outputs2/heldout/finetuned_model",
                    help="self-supervised checkpoint to compare against (relative to q2_embedder_final/)")
    ap.add_argument("--epochs", type=int, default=config.FT_EPOCHS)
    ap.add_argument("--reuse", action="store_true", help="evaluate the saved checkpoint instead of retraining")
    args = ap.parse_args(argv)
    if not (config.PKG_DIR / args.selfsup_model).exists():
        ap.error(f"self-supervised checkpoint not found: {config.PKG_DIR / args.selfsup_model} "
                 "(checkpoints are not committed; train it first or pass --selfsup-model)")

    out = config.PKG_DIR / args.out_dir / "heldout"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "finetuned_model"

    corpus = load_corpus()
    train_idx, test_idx = block_split(corpus)
    train_rows, test_rows = load_qa(corpus, train_idx, test_idx)
    print(f"corpus {len(corpus)} chunks | train {len(train_idx)} chunks / {len(train_rows)} questions | "
          f"held-out {len(test_idx)} chunks / {len(test_rows)} questions")

    # --- fine-tune on mixed pairs ------------------------------------------------------------------
    log_path = out / "train_log.json"
    if args.reuse and ckpt.exists():
        model_qa = DenseEmbedder("custom-finetuned-qa", ckpt)
        log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"losses": []}
        print(f"reusing checkpoint {ckpt}")
    else:
        rounds_ss = make_pairs(corpus.texts, train_idx)
        rounds_q = question_rounds(train_rows, corpus)
        n_ss, n_q = sum(map(len, rounds_ss)), sum(map(len, rounds_q))
        print(f"fine-tuning {args.baseline}: {n_ss} self-supervised + {n_q} question pairs, {args.epochs} epochs")
        t0 = time.time()
        model_qa, losses = finetune_dense(rounds_ss + rounds_q, base_model=args.baseline, epochs=args.epochs,
                                          save_dir=ckpt)
        log = {"losses": losses, "n_selfsup_pairs": n_ss, "n_question_pairs": n_q, "epochs": args.epochs,
               "seconds": round(time.time() - t0, 1)}
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(f"trained in {log['seconds']}s, {len(losses)} steps")
        model_qa = DenseEmbedder("custom-finetuned-qa", ckpt)  # reload from disk = exactly what is evaluated

    embedders = {
        "baseline": DenseEmbedder("baseline", args.baseline),
        "finetuned": DenseEmbedder("finetuned", config.PKG_DIR / args.selfsup_model),
        "finetuned_qa": model_qa,
    }

    # --- retrieval ---------------------------------------------------------------------------------
    ret = evaluate_retrieval(embedders, corpus, test_idx, test_rows)
    X_all, ranks = ret.pop("_X_all"), ret.pop("_ranks")
    (out / "retrieval_results.json").write_text(json.dumps(ret, indent=2), encoding="utf-8")
    (out / "retrieval_results.md").write_text(retrieval_markdown(ret), encoding="utf-8")
    (out / "question_ranks.json").write_text(json.dumps({
        "questions": [{"chunk_id": r["chunk_id"], "qtype": r["qtype"], "question": r["question"],
                       "rt_rank": r["rt_rank"]} for r in test_rows], "ranks": ranks}), encoding="utf-8")
    print(retrieval_markdown(ret).split("## By question type")[0])

    # --- clustering of the 49 held-out chunks (same protocol as main.py / outputs2) -------------------
    ev = corpus.subset(test_idx)
    X = {k: X_all[k][test_idx] for k in MAIN}
    for k, v in X.items():
        np.save(out / f"embeddings_{k}.npy", v)
    clus = {k: evaluate_embeddings(v, ev.doc_ids, ev.case_ids) for k, v in X.items()}
    (out / "results.md").write_text(results_table(clus), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"results": clus, "ft_losses": log.get("losses", [])}, indent=2),
                                      encoding="utf-8")
    (out / "neighbors.md").write_text(
        neighbour_table(config.PROBE_QUERIES, {k: PrefixedQueries(embedders[k]) for k in MAIN}, X, ev),
        encoding="utf-8")
    print(results_table(clus))

    # --- figures -----------------------------------------------------------------------------------
    sub = f"(held-out: {len(test_idx)} chunks, {len(test_rows)} questions)"
    heldout = ret["pools"]["heldout"]
    plot_recall_bars({k: heldout[k]["all"] for k in MAIN}, out / "fig1_recall.png", f"Question retrieval {sub}")
    plot_metric_by_type({k: heldout[k]["by_type"] for k in MAIN}, "mrr", out / "fig2_mrr_by_type.png",
                        f"MRR by question type {sub}")
    plot_separation_bars(clus, out / "fig3_clustering.png", f"Cluster separation ({len(test_idx)} held-out chunks)")
    plot_similarity_hist({k: pair_similarities(v, ev.doc_ids) for k, v in X.items()}, out / "fig4_similarity.png",
                         f"Same- vs different-document similarity ({len(test_idx)} held-out chunks)")
    if log.get("losses"):
        plot_loss(log["losses"], out / "fig5_loss.png", "Fine-tuning loss (mixed pairs)")
    plot_maps({k: project_2d(v) for k, v in X.items()}, ev.doc_ids, out / "fig6_cluster_maps.png",
              f"t-SNE of held-out chunk embeddings ({len(test_idx)} chunks)")
    print(f"wrote results to {out}")


if __name__ == "__main__":
    main()
