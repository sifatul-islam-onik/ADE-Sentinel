# Domain tokenizer - before/after

Measured on the deduplicated ADE corpus (20,896 sentences) by
`scripts/tokenizer_report.py` (PLAN step 2.2).

**Baseline** is `re.findall(r'[A-Za-z0-9]+')` plus unconditional lowercasing -
the default pipeline in most tutorials.

## Aggregate

| Measure | Naive baseline | Domain tokenizer | Delta |
|---|---|---|---|
| Running tokens (all) | 385,127 | 421,474 | +36,347 |
| Running tokens (words only) | 385,127 | 373,201 | -11,926 |
| Vocabulary (types) | 17,198 | 20,320 | +3,122 |
| Hapax legomena | 5,843 | 8,081 | +2,238 |
| Type/token ratio | 0.0447 | 0.0482 | |

Read the *words only* row, not the first one: the naive regex discards punctuation while the domain tokenizer emits it, so the raw totals differ mostly by punctuation and say nothing about tokenisation quality.

On words alone the domain tokenizer emits **11,926 fewer tokens** (373,201 vs 385,127), because it refuses to shatter multi-part terms.

Vocabulary moves the other way: **20,320 types vs 17,198**, because `long-term` and `drug-induced` are single types here and recycled fragments there. That is a real cost - a larger vocabulary means more embedding rows and more rare types - and it is the right trade: **3,954 of those types cannot be represented by the naive pipeline at all**, and they are the domain-bearing ones. It also sharpens the Section 7 experiment rather than blunting it, since these are exactly the terms general-purpose GloVe vectors will fail to cover.

## What the baseline destroys

The 25 most frequent multi-part tokens the domain tokenizer keeps whole, each
of which the naive baseline splits into meaningless fragments:

| Token kept whole | Freq | Naive baseline produces |
|---|---|---|
| `long-term` | 179 | `long` + `term` |
| `high-dose` | 145 | `high` + `dose` |
| `follow-up` | 139 | `follow` + `up` |
| `drug-induced` | 133 | `drug` + `induced` |
| `life-threatening` | 109 | `life` + `threatening` |
| `5-fluorouracil` | 59 | `5` + `fluorouracil` |
| `5-fu` | 51 | `5` + `fu` |
| `low-dose` | 49 | `low` + `dose` |
| `g-csf` | 43 | `g` + `csf` |
| `b-cell` | 41 | `b` + `cell` |
| `side-effects` | 38 | `side` + `effects` |
| `ifn-alpha` | 36 | `ifn` + `alpha` |
| `anti-inflammatory` | 35 | `anti` + `inflammatory` |
| `tnf-alpha` | 34 | `tnf` + `alpha` |
| `x-ray` | 31 | `x` + `ray` |
| `colony-stimulating` | 28 | `colony` + `stimulating` |
| `l-asparaginase` | 28 | `l` + `asparaginase` |
| `epstein-barr` | 27 | `epstein` + `barr` |
| `interferon-alpha` | 26 | `interferon` + `alpha` |
| `t-cell` | 26 | `t` + `cell` |
| `post-transplant` | 26 | `post` + `transplant` |
| `pre-existing` | 25 | `pre` + `existing` |
| `non-hodgkin's` | 24 | `non` + `hodgkin` + `s` |
| `therapy-related` | 24 | `therapy` + `related` |
| `trimethoprim-sulfamethoxazole` | 24 | `trimethoprim` + `sulfamethoxazole` |

## Case protection

All-caps medical abbreviations that unconditional lowercasing would collapse
into ordinary English words. `ALL` is the canonical example: lowercased it
becomes a determiner and acute lymphoblastic leukemia disappears from the
corpus entirely.

Protected forms observed in this corpus: **84**

| Abbreviation | Occurrences | Collapses to |
|---|---|---|
| `HIV` | 89 | `hiv` |
| `CT` | 66 | `ct` |
| `CMV` | 65 | `cmv` |
| `AML` | 57 | `aml` |
| `MTX` | 56 | `mtx` |
| `RA` | 52 | `ra` |
| `MRI` | 47 | `mri` |
| `CNS` | 46 | `cns` |
| `ACE` | 42 | `ace` |
| `IV` | 39 | `iv` |
| `AIDS` | 38 | `aids` |
| `HBV` | 34 | `hbv` |
| `ALL` | 30 | `all` |
| `SLE` | 29 | `sle` |
| `INR` | 28 | `inr` |
| `EEG` | 27 | `eeg` |
| `ECG` | 25 | `ecg` |
| `MDS` | 24 | `mds` |
| `MS` | 24 | `ms` |
| `CML` | 23 | `cml` |

## Why this matters downstream

Every fragmented token is a vocabulary entry the embedding layer must learn
separately, and a token whose meaning has been destroyed. `5` and
`fluorouracil` carry neither the identity of the drug nor its relationship to
the effect it causes. Since the project's headline claim is about the quality
of the embedding space (PRD section 7), the tokenizer sets the ceiling on what
any embedding can achieve - which is why this table precedes the embedding
experiment rather than following it.
