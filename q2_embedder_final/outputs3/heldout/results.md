| model | sil(doc) | sil(case) | intra-inter sim | pair AUC | ARI@k=docs | NMI@k=docs | kNN-purity@5 | best k |
|---|---|---|---|---|---|---|---|---|
| Baseline (bge-small) | 0.083 | 0.113 | 0.046 | 0.641 | 0.189 | 0.406 | 0.678 | 10 |
| Self-supervised FT (bge-small) | 0.126 | 0.131 | 0.114 | 0.714 | 0.415 | 0.574 | 0.682 | 9 |
| Questions + self-supervised FT (bge-small) | 0.117 | 0.124 | 0.128 | 0.707 | 0.439 | 0.585 | 0.682 | 8 |