# Resources referred to (raw dump)

## Data
- `data/*.pdf` – the five source documents (Macquarie v. Moab; Facebook v. Amalgamated petitioners' brief and U.S. amicus brief; Last Atlantis v. AGS; SEC v. Chiueh).
- `data/processed/chunks.jsonl`, `ingest_report.md` – chunks produced by the Q1 ingest (`src/ingest`), reused unchanged here.

## Models
- Baseline: `sentence-transformers/all-MiniLM-L6-v2` – https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Alternative baseline (used by Q1): `BAAI/bge-small-en-v1.5` – https://huggingface.co/BAAI/bge-small-en-v1.5

## Papers / references
- Reimers & Gurevych (2019), Sentence-BERT – https://arxiv.org/abs/1908.10084
- Henderson et al. (2017), in-batch negatives ranking loss (basis of MultipleNegativesRankingLoss) – https://arxiv.org/abs/1705.00652
- Gao et al. (2021), SimCSE (unsupervised contrastive sentence embeddings) – https://arxiv.org/abs/2104.08821
- Wang et al. (2021), TSDAE (unsupervised domain adaptation of sentence embeddings) – https://arxiv.org/abs/2104.06979
- Deerwester et al. (1990), Latent Semantic Analysis, JASIS 41(6)
- Rousseeuw (1987), Silhouettes, J. Comput. Appl. Math. 20 – the silhouette coefficient
- van der Maaten & Hinton (2008), t-SNE, JMLR 9
- scikit-learn: `silhouette_score`, `adjusted_rand_score`, `TruncatedSVD`, `TfidfVectorizer`, `TSNE` – https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

> Paper PDFs / model cards are not vendored here; drop copies into this folder if the "raw dump" needs to be self-contained.
