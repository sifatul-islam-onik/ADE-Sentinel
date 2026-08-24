# Split protocol - measured

Produced by `scripts/build_splits.py`, seed **42**, 70/15/15 stratified on the Stage 1 label.

## Deduplication

| Config | Rows | Unique sentences | Note |
|---|---|---|---|
| classification | 23,516 | 20,896 | true duplicates, dropped |
| drug-ade relation | 6,821 | 4,271 | one row per *relation*: grouped, spans unioned |

Naive `drop_duplicates` on the relation config would have discarded **2,550** drug-effect pairs (37.4% of all Stage 2 supervision). Grouping preserves them: 11,014 unique spans across 4,271 sentences.

9 drug mentions and 28 effect mentions name an entity but carry no character offsets, so they contribute no BIO supervision. They are reported here rather than silently producing all-`O` sequences (PLAN F5). 0 rows lack offsets for *both* fields.

## One split, shared by both stages

Cross-config overlap is total: every relation sentence is also a classification row. Independent splits would leak Stage-1 training sentences into the Stage-2 test set and invalidate the run-12 error-propagation measurement, so the assignment is computed once over the union and projected onto each config.

| Split | Stage 1 rows | % positive | Stage 2 sentences | Stage 2 spans |
|---|---|---|---|---|
| train | 14,628 | 20.44% | 2,990 | 7,739 |
| dev | 3,135 | 20.45% | 641 | 1,639 |
| test | 3,133 | 20.43% | 640 | 1,636 |

## Assertions

- No sentence appears in more than one split, checked across **both** configs.
- Per-split positive ratio stays within 1pp of the corpus ratio.
- Every Stage 2 sentence is a Stage 1 positive.

These files are committed to git deliberately: they are small, and committing them is what makes "never re-split at runtime" enforceable.
