# Manual output audit

**Purpose:** diagnostic check of the automatic CUAD answer score, not a replacement metric or an independent annotation study.

**Sampling:** 20 paired question examples, sampled with seed `261126` within answerable and no-answer strata: 10 answerable and 10 no-answer. Both methods' outputs were reviewed for each sampled question.

## Findings

| Question group and method | Clearly correct / correct abstention | Partial or relevant | Missed |
|---|---:|---:|---:|
| Answerable, full contract prompt (n=10) | 2 | 6 | 2 |
| Answerable, dense RAG (n=10) | 2 | 2 | 6 |
| No-answer, full contract prompt (n=10) | 10 | — | 0 |
| No-answer, dense RAG (n=10) | 10 | — | 0 |

One answerable example involved a heavily redacted clause; both systems abstained. That case may reflect ambiguity in the source annotation as well as model behavior. The sample is too small to estimate system accuracy, and a few cases included judgment calls between a partial answer and a miss.

## Interpretation limit

The audit is consistent with the main metric's direction on answerable questions, but it is only a small sanity check. The primary quantitative result remains SQuAD-style best-span token F1. This is not the official full multi-span CUAD extraction score and does not establish legal correctness.

Raw generated answers were kept outside the project during review and are not part of this artifact.
