# Vocabulary coverage (PRD 7.2)

Measured over the deduplicated ADE corpus: 20,896 sentences, 373,201 tokens, 20,298 distinct types, using the project's own tokenizer.

| Embedding | Type coverage | Token coverage | Types known | Types OOV | Type coverage, exact case |
|---|---|---|---|---|---|
| E1 GloVe (general) | 66.18% | 95.67% | 13,434 | 6,864 | 65.77% |
| E2 Word2Vec (ours) | 79.86% | 98.31% | 16,210 | 4,088 | 79.86% |

## Reading this table

**Token coverage is high for every embedding; type coverage is not.** That gap is the finding. General-purpose vectors cover the common English scaffolding - `the`, `patient`, `was`, `after` - which is most of the running text, so token coverage flatters them. Type coverage counts each distinct word once, and it is the distinct words that carry the domain: a drug name appearing three times is as informative as one appearing three hundred.

The `exact case` column exists for fairness, not decoration. Our tokenizer protects medical abbreviations from lowercasing (`ALL`, `HIV`, `CT`), while `glove.6B` is an uncased release holding `hiv` and `ct`. Scoring those as GloVe misses would penalise the baseline for **our** preprocessing choice, so every lookup retries case-folded. The headline columns use that retry; the last column shows what a naive exact-match comparison would have reported instead.

For GloVe that retry is worth +0.41 percentage points (65.77% -> 66.18%). Small, but now measured rather than assumed - and the domain advantage survives it, which is the point.

## Most frequent task terms GloVe has never seen

Case-folded, so these are genuinely absent rather than artefacts of casing.

| Term | Frequency in the ADE corpus |
|---|---|
| `patient's` | 193 |
| `intravitreal` | 75 |
| `crohn's` | 68 |
| `endophthalmitis` | 55 |
| `rechallenge` | 46 |
| `ifn-alpha` | 36 |
| `siadh` | 36 |
| `microg` | 32 |
| `hodgkin's` | 31 |
| `parkinson's` | 28 |
| `mofetil` | 28 |
| `ptld` | 28 |
| `neutropenic` | 28 |
| `l-asparaginase` | 28 |
| `capecitabine` | 27 |
| `leflunomide` | 27 |
| `interferon-alpha` | 26 |
| `post-transplant` | 26 |
| `fludarabine` | 25 |
| `chlorambucil` | 25 |
| `non-hodgkin's` | 24 |
| `therapy-related` | 24 |
| `arabinoside` | 24 |
| `cystoid` | 24 |
| `anaphylactoid` | 24 |
| `propafenone` | 24 |
| `trimethoprim-sulfamethoxazole` | 24 |
| `cholestatic` | 23 |
| `tobramycin` | 23 |
| `heparin-induced` | 22 |

Two kinds of miss appear here. Most are **domain vocabulary** - `intravitreal`, `endophthalmitis`, `rechallenge`, `capecitabine`, `l-asparaginase` - words a general-purpose corpus simply never contains. A few are **possessives** (`patient's`, `crohn's`, `parkinson's`), where GloVe's tokenisation split the clitic and ours does not. The second group is a tokenisation difference rather than a vocabulary gap, and is called out here so the table is not over-read.
