# Retrieval of held-out questions (224 questions from 49 held-out chunks)

Gold chunk = the chunk a question was written from. Models never saw these chunks or questions in training. `no query prefix` rows omit bge's query instruction.

## Pool = the 49 held-out chunks (headline)

| model | R@1 | R@3 | R@5 | R@10 | MRR | median rank |
|---|---|---|---|---|---|---|
| Baseline (bge-small) | 0.661 | 0.848 | 0.906 | 0.969 | 0.768 | 1 |
| Baseline (bge-small), no query prefix | 0.638 | 0.839 | 0.911 | 0.964 | 0.754 | 1 |
| Self-supervised FT (bge-small) | 0.683 | 0.884 | 0.938 | 0.964 | 0.792 | 1 |
| Self-supervised FT (bge-small), no query prefix | 0.674 | 0.875 | 0.938 | 0.960 | 0.785 | 1 |
| Questions + self-supervised FT (bge-small) | 0.781 | 0.915 | 0.960 | 0.982 | 0.860 | 1 |
| Questions + self-supervised FT (bge-small), no query prefix | 0.768 | 0.929 | 0.951 | 0.978 | 0.853 | 1 |

### Subset: questions whose gold chunk is in the baseline's top 10 (n = 193)

| model | R@1 | R@3 | R@5 | R@10 | MRR | median rank |
|---|---|---|---|---|---|---|
| Baseline (bge-small) | 0.767 | 0.974 | 1.000 | 1.000 | 0.865 | 1 |
| Baseline (bge-small), no query prefix | 0.741 | 0.959 | 1.000 | 1.000 | 0.848 | 1 |
| Self-supervised FT (bge-small) | 0.782 | 0.969 | 0.995 | 1.000 | 0.876 | 1 |
| Self-supervised FT (bge-small), no query prefix | 0.782 | 0.964 | 1.000 | 1.000 | 0.874 | 1 |
| Questions + self-supervised FT (bge-small) | 0.886 | 0.979 | 1.000 | 1.000 | 0.936 | 1 |
| Questions + self-supervised FT (bge-small), no query prefix | 0.876 | 0.984 | 0.995 | 1.000 | 0.931 | 1 |

## Pool = all 196 chunks (harder; the fine-tuned models saw the 147 train chunks)

| model | R@1 | R@3 | R@5 | R@10 | MRR | median rank |
|---|---|---|---|---|---|---|
| Baseline (bge-small) | 0.536 | 0.705 | 0.768 | 0.862 | 0.646 | 1 |
| Baseline (bge-small), no query prefix | 0.496 | 0.696 | 0.750 | 0.835 | 0.616 | 2 |
| Self-supervised FT (bge-small) | 0.558 | 0.737 | 0.808 | 0.906 | 0.669 | 1 |
| Self-supervised FT (bge-small), no query prefix | 0.518 | 0.714 | 0.804 | 0.888 | 0.641 | 1 |
| Questions + self-supervised FT (bge-small) | 0.612 | 0.830 | 0.879 | 0.938 | 0.734 | 1 |
| Questions + self-supervised FT (bge-small), no query prefix | 0.598 | 0.817 | 0.875 | 0.938 | 0.725 | 1 |

### Subset: questions whose gold chunk is in the baseline's top 10 (n = 193)

| model | R@1 | R@3 | R@5 | R@10 | MRR | median rank |
|---|---|---|---|---|---|---|
| Baseline (bge-small) | 0.622 | 0.819 | 0.891 | 1.000 | 0.741 | 1 |
| Baseline (bge-small), no query prefix | 0.575 | 0.808 | 0.870 | 0.964 | 0.707 | 1 |
| Self-supervised FT (bge-small) | 0.642 | 0.845 | 0.912 | 0.984 | 0.757 | 1 |
| Self-supervised FT (bge-small), no query prefix | 0.601 | 0.824 | 0.902 | 0.969 | 0.728 | 1 |
| Questions + self-supervised FT (bge-small) | 0.705 | 0.917 | 0.964 | 0.995 | 0.819 | 1 |
| Questions + self-supervised FT (bge-small), no query prefix | 0.689 | 0.907 | 0.964 | 0.995 | 0.810 | 1 |

## By question type (held-out pool): MRR / Recall@1

| model | factual | conceptual | argument | paraphrase | keyword |
|---|---|---|---|---|---|
| Baseline (bge-small) | 0.792 / 0.675 | 0.742 / 0.628 | 0.690 / 0.578 | 0.686 / 0.531 | 0.932 / 0.894 |
| Self-supervised FT (bge-small) | 0.814 / 0.725 | 0.777 / 0.651 | 0.748 / 0.622 | 0.707 / 0.551 | 0.920 / 0.872 |
| Questions + self-supervised FT (bge-small) | 0.872 / 0.800 | 0.879 / 0.814 | 0.821 / 0.711 | 0.765 / 0.653 | 0.968 / 0.936 |

## Paired difference with 95% chunk-bootstrap interval

| comparison | mean diff | 95% interval | interval excludes 0 |
|---|---|---|---|
| heldout: finetuned_qa - baseline [mrr] | +0.092 | [+0.063, +0.124] | yes |
| heldout: finetuned_qa - baseline [recall@1] | +0.121 | [+0.075, +0.172] | yes |
| heldout: finetuned_qa - baseline [recall@5] | +0.054 | [+0.027, +0.086] | yes |
| heldout: finetuned_qa - finetuned [mrr] | +0.067 | [+0.044, +0.093] | yes |
| heldout: finetuned_qa - finetuned [recall@1] | +0.098 | [+0.058, +0.141] | yes |
| heldout: finetuned_qa - finetuned [recall@5] | +0.022 | [+0.005, +0.041] | yes |
| heldout: finetuned - baseline [mrr] | +0.024 | [-0.001, +0.051] | no |
| heldout: finetuned - baseline [recall@1] | +0.022 | [-0.018, +0.063] | no |
| heldout: finetuned - baseline [recall@5] | +0.031 | [+0.004, +0.063] | yes |
| full: finetuned_qa - baseline [mrr] | +0.089 | [+0.056, +0.122] | yes |
| full: finetuned_qa - baseline [recall@1] | +0.076 | [+0.023, +0.130] | yes |
| full: finetuned_qa - baseline [recall@5] | +0.112 | [+0.071, +0.153] | yes |
| full: finetuned_qa - finetuned [mrr] | +0.065 | [+0.035, +0.096] | yes |
| full: finetuned_qa - finetuned [recall@1] | +0.054 | [+0.009, +0.100] | yes |
| full: finetuned_qa - finetuned [recall@5] | +0.071 | [+0.040, +0.107] | yes |
| full: finetuned - baseline [mrr] | +0.024 | [-0.009, +0.056] | no |
| full: finetuned - baseline [recall@1] | +0.022 | [-0.027, +0.071] | no |
| full: finetuned - baseline [recall@5] | +0.040 | [+0.000, +0.081] | no |
