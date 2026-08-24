# Dataset statistics - measured, not estimated

Source: `ade-benchmark-corpus/ade_corpus_v2` (Hugging Face Hub, parquet-native).
Produced by `scripts/sanity_check.py` (PLAN step 0.3).
**Quote these numbers in the report, not the PRD's approximations.**

## Row counts

| Config | Rows | Unique sentences | Duplicate rows |
|---|---|---|---|
| `Ade_corpus_v2_classification` | 23,516 | 20,896 | 2,620 |
| `Ade_corpus_v2_drug_ade_relation` | 6,821 | 4,271 | 2,550 |

## Class balance (classification config)

| Label | Rows | Share |
|---|---|---|
| 0 (not ADE) | 16,695 | 70.99% |
| 1 (ADE) | 6,821 | 29.01% |

### After deduplication - the balance Stage 1 actually trains on

| Label | Unique sentences | Share |
|---|---|---|
| 0 (not ADE) | 16,625 | 79.56% |
| 1 (ADE) | 4,271 | 20.44% |

Raw positive share is 29.01%, but after dedup it is **20.44%**. The duplication is concentrated in the positive class, so dedup is not label-neutral - it shifts the balance materially. The PRD's "roughly 1 in 5" describes the *post-dedup* corpus, which is the correct one to design against.

Either way, **macro-F1 and per-class F1 are the primary metrics**; raw accuracy is misleading at this imbalance (PRD 6.1 fact 3).

Sentences carrying conflicting labels across duplicate rows: **0**. Dedup is well-defined - any duplicate row can be kept.

## Cross-config overlap (PLAN F4)

- Sentences in both configs: **4,271**
- Unique sentences across the union of both configs: **20,896**

Because the configs share sentences, splitting them independently would put a sentence in Stage-1 *train* and Stage-2 *test*. The split must be built once over the union and projected onto each config, or the run-12 error-propagation number is invalid.

## Relations per sentence (PLAN F3)

| Relations in one sentence | Number of sentences |
|---|---|
| 1 | 2,865 |
| 2 | 943 |
| 3 | 207 |
| 4 | 123 |
| 5 | 33 |
| 6 | 52 |
| 7 | 4 |
| 8 | 25 |
| 9 | 2 |
| 10 | 5 |
| 12 | 5 |
| 14 | 1 |
| 15 | 1 |
| 16 | 2 |
| 18 | 1 |
| 21 | 1 |
| 24 | 1 |

Naive `drop_duplicates` on the relation config would discard **2,550 drug-effect pairs** (37.4% of all relations), permanently capping Stage 2 recall. Dedup for Stage 2 must GROUP BY sentence and UNION the spans.

## First row of each config

```python
cls[0] = {'text': 'Intravenous azithromycin-induced ototoxicity.', 'label': 1}

rel[0] = {'text': 'Intravenous azithromycin-induced ototoxicity.', 'drug': 'azithromycin', 'effect': 'ototoxicity', 'indexes': {'drug': {'start_char': [12], 'end_char': [24]}, 'effect': {'start_char': [33], 'end_char': [44]}}}
```

## Reference check

Published NER benchmark statistics for the positive portion: 4,272 sentences, 86,865 tokens, 12,264 ADE tags, 5,544 Drug tags. Use these to sanity-check Stage 2 numbers against the literature (PRD 6.1 fact 4).
