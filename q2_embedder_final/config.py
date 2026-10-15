"""Central configuration: paths, model names, hyper-parameters, labels and probe queries."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = Path(__file__).resolve().parent
OUT_DIR = PKG_DIR / "outputs"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"  # written by the Q1 ingest (src.ingest)

SEED = 42

# ----------------------------------------------------------------------------------- data
MIN_CHUNK_TOKENS = 40  # drop tiny tail / footnote stubs
TEST_FRAC = 0.25  # held-out share of contiguous chunk blocks
BLOCK_SIZE = 6  # blocks (not single chunks) are held out: chunks overlap by ~50 tokens, so
#                 random chunk-level splits would leak text between train and test

# 5 PDFs, but the two Facebook briefs are the same case -> 4 "case" labels
CASE_OF = {
    "macquarie": "Macquarie v. Moab",
    "fb_petitioners": "Facebook v. Amalgamated",
    "fb_amicus": "Facebook v. Amalgamated",
    "last_atlantis": "Last Atlantis v. AGS",
    "sec_chiueh": "SEC v. Chiueh",
}

# ----------------------------------------------------------------------------------- models
BASELINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # cached locally; swap for BAAI/bge-small-en-v1.5
MAX_SEQ_LEN = 384  # MiniLM default is 256 and would truncate ~400-token chunks; same for all dense models

LSA_DIM = 64  # "no need for long embedding lengths"

FT_EPOCHS = 2
FT_BATCH_SIZE = 16
FT_LR = 2e-5
FT_SCALE = 20.0  # cosine-sim temperature for the in-batch-negatives loss
FT_AUGMENTATIONS = 4  # random sentence-split pairs generated per chunk
FT_MIN_SENTENCES = 2

# ----------------------------------------------------------------------------------- evaluation
K_SWEEP = list(range(2, 11))
KNN_KS = (1, 3, 5, 10)
N_PAIR_SAMPLES = None  # None = use all chunk pairs (n is small)

NAMES = {
    "baseline": "Baseline (MiniLM)",
    "lsa": "Custom A: legal TF-IDF + LSA",
    "finetuned": "Custom B: fine-tuned MiniLM",
    "baseline_noprefix": "Baseline (no query prefix)",
    "finetuned_qa": "Custom C: fine-tuned on questions + self-supervised",
}

# ----------------------------------------------------------------------------------- synthetic questions (Q2, approach 3)
TRAINING_DATA_DIR = PKG_DIR / "training_data"
OUT_DIR3 = PKG_DIR / "outputs3"
QA_BASELINE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge-small's documented query instruction
QUESTION_TYPES = ("factual", "conceptual", "argument", "paraphrase", "keyword")
QA_MIN_WORDS, QA_MAX_WORDS = 3, 40  # keyword-style queries are short on purpose
QA_DUP_COSINE = 0.95  # near-duplicate threshold between sibling questions
QA_COPY_NGRAM = 8  # a question containing an 8-token span copied from its chunk is dropped
QA_ROUNDTRIP_TOP = 10  # train filter: baseline must rank the source chunk within the top-10 of the corpus
QA_RECALL_KS = (1, 3, 5, 10)

# neighbour check: free-text probes embedded by each model, top-3 chunks reported
PROBE_QUERIES = [
    "omission of required disclosure that makes a statement misleading",
    "pure omissions versus half-truths under Rule 10b-5(b)",
    "Item 303 known trends and uncertainties in management's discussion",
    "SEC enforcement action for fraud by an investment adviser",
    "private securities fraud class action scienter and loss causation",
]
