# BIO conversion - corpus-wide diagnostics

Produced by `scripts/bio_spotcheck.py` (PLAN step 5.3). These numbers exist
because the PRD's reference converter fails silently in two ways, and
"probably zero" is not a measurement (PLAN F5).

| Measure | Value |
|---|---|
| Sentences converted | 4,271 |
| Spans in | 11,014 |
| Entities tagged | 10,857 |
| Spans needing boundary snapping | 744 |
| Spans matching no token | 0 |
| Nested spans dropped (flat BIO cannot hold them) | 157 |
| Illegal transitions in gold tags | 0 |

## Entity counts

| Label | Entities |
|---|---|
| DRUG | 5,044 |
| EFFECT | 5,799 |

## Tag distribution

| Tag | Tokens | Share |
|---|---|---|
| `B-DRUG` | 5,044 | 5.86% |
| `B-EFFECT` | 5,799 | 6.74% |
| `I-DRUG` | 593 | 0.69% |
| `I-EFFECT` | 6,402 | 7.44% |
| `O` | 68,242 | 79.28% |

`O` accounts for 79.3% of tokens, which is exactly why token accuracy is not reported for Stage 2: a model predicting `O` everywhere would score that high while finding nothing. Entity-level precision/recall/F1 via `seqeval` is the metric.

## Interpretation

No span failed to align - every annotation reached the tagger.

**157 spans were dropped as nested.** The ADE corpus annotates drug names inside effect phrases - `theophylline` inside `theophylline intoxication`, `lead` inside `high blood lead level`. Flat BIO gives each token exactly one tag, so the two cannot coexist; letting the inner span overwrite the outer one orphans the remainder into an illegal `I-EFFECT`. The longer span wins, since it is the more complete annotation of the adverse event. Dropped by label: DRUG 140, EFFECT 17.

The general fix is one BIO plane per entity type - all observed nesting is DRUG-inside-EFFECT - which needs a two-head tagger and is outside the PRD's model ladder. The loss is therefore measured and reported rather than hidden, and it caps Stage 2 recall by this amount.

- `DRUG 'theophylline' nested in 'A case is reported of theophylline intoxication due to a dra'`
- `DRUG 'theophylline' nested in 'A case is reported of theophylline intoxication due to a dra'`
- `DRUG 'theophylline' nested in 'A diagnosis of masked theophylline poisoning should be consi'`
- `DRUG 'methotrexate' nested in 'A macrophage activation syndrome, possibly related to methot'`
- `DRUG 'phenytoin' nested in 'A patient is described with the characteristic features of p'`
- `DRUG 'loperamide' nested in 'A retrospective study was conducted of 40 loperamide poisoni'`
- `DRUG 'alum' nested in 'Acute aluminum toxicity after continuous intravesical alum i'`
- `DRUG 'esmolol' nested in 'Acute esmolol toxicity may be self-limiting because of its e'`
- `DRUG 'acetazolamide' nested in 'Acute hemorrhagic gastritis associated with acetazolamide in'`
- `DRUG 'nitrite' nested in 'Acute nitrite toxicity results from industrial exposure, acc'`

**744 spans did not fall on token boundaries** and were widened to the enclosing tokens. The PRD's containment test would have dropped these entirely. Examples (annotation -> tagged tokens):

- `'phenytoin' -> 'phenytoin-induced'`
- `'dapsone' -> 'dapsone-induced'`
- `'barbiturate' -> 'barbiturate-induced'`
- `'colchicine' -> 'colchicine-induced'`
- `'metoclopramide' -> 'metoclopramide-induced'`
- `'phenytoin' -> 'phenytoin-induced'`
- `'timolol' -> 'timolol-associated'`
- `'triazolam' -> 'triazolam-induced'`
- `'infliximab' -> 'infliximab-induced'`
- `'rifampicin' -> 'rifampicin-induced'`

Gold tag sequences contain no illegal transitions, as they must by construction. That zero is the baseline the CRF ablation in step 5.8 measures model output against.
