# Synthetic question data - build report

**Authorship:** every question was written by Claude (the coding assistant) after reading the chunk; no LLM API or RAG-system model was used.
**Round-trip model:** `BAAI/bge-small-en-v1.5` with the bge query prefix; rank = position of the source chunk among all 196 chunks.

## Counts

| split | chunks | questions authored | kept | dropped |
|---|---|---|---|---|
| train | 147 | 735 | 573 | 162 |
| test | 49 | 245 | 224 | 21 |

## Kept questions by type

| type | train kept / authored | test kept / authored |
|---|---|---|
| factual | 118 / 147 | 40 / 49 |
| conceptual | 114 / 147 | 43 / 49 |
| argument | 102 / 147 | 45 / 49 |
| paraphrase | 102 / 147 | 49 / 49 |
| keyword | 137 / 147 | 47 / 49 |

## Drop reasons

| reason | train | test |
|---|---|---|
| copies_chunk_span | 69 | 20 |
| document_reference | 1 | 1 |
| roundtrip_rank_gt_10 | 92 | 0 |

## Baseline rank of the source chunk (all authored questions)

| split | rank 1 | rank <= 3 | rank <= 10 | median rank |
|---|---|---|---|---|
| train | 0.53 | 0.72 | 0.86 | 1 |
| test | 0.53 | 0.70 | 0.86 | 1 |

## Kept questions per train chunk

mean 3.90; min 0; chunks with < 3 kept: 13; chunks with 0 kept: 1

## Notes

- Train questions must pass the round-trip filter (source chunk in the baseline's top 10). Test questions are **not** round-trip filtered, so the test set is not biased toward questions the baseline already answers; `rt_rank` is stored for subset analysis.
- Structural filters (both splits): word count, references to the document itself ("this brief", "page 5"), copied 8-token spans, near-duplicate siblings.
- See `dropped.jsonl` for every dropped question with its reason.
