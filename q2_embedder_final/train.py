"""Training of the two custom embedders.

* Custom A  - `train_lsa`: TF-IDF + SVD fitted on the corpus (no neural network, no labels).
* Custom B  - `finetune_dense`: baseline MiniLM adapted with a contrastive objective on
              self-supervised sentence-split pairs (in-batch negatives, symmetric InfoNCE).
              A plain torch loop is used on purpose: it is ~25 lines and keeps the loss visible.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F

from q2_embedder_final import config
from q2_embedder_final.embed import DenseEmbedder, LSAEmbedder


def train_lsa(train_texts: list[str], dim: int = config.LSA_DIM) -> LSAEmbedder:
    return LSAEmbedder(dim=dim).fit(train_texts)


def finetune_dense(rounds, base_model: str = config.BASELINE_MODEL, epochs: int = config.FT_EPOCHS,
                   batch_size: int = config.FT_BATCH_SIZE, lr: float = config.FT_LR,
                   scale: float = config.FT_SCALE, seed: int = config.SEED,
                   save_dir: Path | None = None, verbose: bool = True) -> tuple[DenseEmbedder, list[float]]:
    from sentence_transformers import SentenceTransformer

    random.seed(seed)
    torch.manual_seed(seed)
    model = SentenceTransformer(base_model, device="cpu")
    model.max_seq_length = config.MAX_SEQ_LEN

    # batches: never mix rounds (one pair per chunk per round) -> no same-chunk false negatives
    def make_batches():
        batches = []
        for rnd in rounds:
            pairs = rnd[:]
            random.shuffle(pairs)
            batches += [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]
            batches = [b for b in batches if len(b) >= 2]
        random.shuffle(batches)
        return batches

    n_steps = epochs * len(make_batches())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, int(0.1 * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (n_steps - s) / max(1, n_steps - warm))))

    def embed(texts):
        feats = {k: v for k, v in model.tokenize(texts).items()}
        return F.normalize(model(feats)["sentence_embedding"], dim=-1)

    losses: list[float] = []
    model.train()
    step = 0
    for ep in range(epochs):
        for batch in make_batches():
            a = embed([p[0] for p in batch])
            b = embed([p[1] for p in batch])
            logits = scale * a @ b.T
            target = torch.arange(len(batch))
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
            step += 1
            if verbose and step % 10 == 0:
                print(f"  finetune step {step}/{n_steps}  loss {loss.item():.4f}")
    model.eval()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(save_dir))
    return DenseEmbedder("custom-finetuned", model), losses
