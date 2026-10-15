# Resources referred to (raw dump)

## Data
- `data/*.pdf` – the five source documents (Macquarie v. Moab; Facebook v. Amalgamated petitioners' brief and U.S. amicus brief; Last Atlantis v. AGS; SEC v. Chiueh).
- `data/processed/chunks.jsonl`, `ingest_report.md` – chunks produced by the Q1 ingest (`src/ingest`), reused unchanged here.

## Models
- Baseline: `sentence-transformers/all-MiniLM-L6-v2` – https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Alternative baseline (used by Q1): `BAAI/bge-small-en-v1.5` – https://huggingface.co/BAAI/bge-small-en-v1.5

## Papers / references
- Muennighoff et al. (2023), MTEB: Massive Text Embedding Benchmark (source of the base-model rankings) – https://arxiv.org/abs/2210.07316
- scikit-learn: `silhouette_score` – https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

> Paper PDFs / model cards are not vendored here; drop copies into this folder if the "raw dump" needs to be self-contained.
