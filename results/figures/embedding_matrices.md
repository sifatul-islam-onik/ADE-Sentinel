# Embedding matrices (step 3.6)

Built by `scripts/build_embedding_matrices.py`, seed 42. These are the files runs 3-6 load; the BiLSTM code and every hyperparameter stay fixed, and only the matrix changes.

Shared vocabulary: **12,220 rows** (300d), built from tokens appearing at least 2 times in the ADE corpus. Row 0 is `<pad>` (zero), row 1 is `<unk>`. 8,080 singleton types were dropped - no model here can learn a useful vector from one occurrence.

| Matrix | Representation | Rows filled | Coverage |
|---|---|---|---|
| `E0_random.npy` | Randomly initialised (ablation floor) | 0 | 0.0% |
| `E1.npy` | GloVe 300d (general purpose) | 9,413 | 77.0% |
| `E2.npy` | Word2Vec skip-gram 300d (ours) | 11,160 | 91.3% |
| `E3.npy` | FastText skip-gram 300d (ours) | 12,218 | 100.0% |

## Why one shared random base

Every matrix starts from the *same* seeded `Uniform(-0.25, 0.25)` draw, and each representation overwrites only the rows it covers. Uncovered rows are therefore byte-identical across E0-E3.

This matters for the validity of runs 3-6. Drawing fresh noise per matrix would mean E1's uncovered third and E2's uncovered fifth held *different* random values, so part of any measured F1 difference would be attributable to that noise rather than to the embeddings. With a shared base, the only thing that differs between the four runs is the vectors themselves.

## Note on E3

FastText fills 100% of rows because it synthesises a vector for any string from character n-grams. That is not evidence of superiority - see the caveat in `coverage.md`. It does mean E3 is the only representation with no random rows at all, which is itself a difference between the runs and should be mentioned when interpreting them.

## Consumption

```python
import json, numpy as np
vocab = json.load(open('models/emb_matrices/vocab.json'))
index = {t: i for i, t in enumerate(vocab)}
weights = np.load('models/emb_matrices/E2.npy')   # swap E0/E1/E2/E3 here
```

Row order is identical across all four files. Loading a different file must be the *only* change between runs 3-6.
