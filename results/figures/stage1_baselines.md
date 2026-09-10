# Stage 1 - sparse baselines (step 4.1, runs 1-2)

Produced by `scripts/train_baselines.py`. Trained on the frozen split (`data/splits/stage1_*.parquet`): 14,628 train / 3,135 dev / 3,133 test, 20.4% positive.

Hyperparameters were selected on **dev** macro-F1 and the winner is reported on **test**, fitted on train only - the neural runs need dev for early stopping, so refitting these on train+dev would hand them a data advantage that has nothing to do with the model.

## Test-set results

| Run | Model | Features | Macro-F1 | ADE P | ADE R | ADE F1 | PR-AUC | Acc. |
|---|---|---|---|---|---|---|---|---|
| 1 | naive_bayes | count-1-2gram | **0.7678** | 0.602 | 0.675 | 0.636 | 0.683 | 0.842 |
| 2 | logreg | tfidf-1-2gram | **0.7934** | 0.671 | 0.672 | 0.671 | 0.743 | 0.866 |
| 2b | linear_svm | tfidf-1-2gram | **0.7977** | 0.660 | 0.703 | 0.681 | 0.749 | 0.865 |

Majority-class accuracy on this test split is **0.796**. Every model above beats it on accuracy while differing sharply on macro-F1, which is the concrete reason PRD section 5 deprecates accuracy for this task.

## Selected hyperparameters

| Run | Chosen | Grid | Dev macro-F1 | Features | Fit time |
|---|---|---|---|---|---|
| 1 | `alpha=0.3` | 4 configs | 0.7697 | 37,992 | 0.1s |
| 2 | `C=16.0`, `class_weight=balanced` | 8 configs | 0.7895 | 37,992 | 3.5s |
| 2b | `C=0.5`, `class_weight=balanced` | 8 configs | 0.7921 | 37,992 | 1.5s |

## Confusion matrices (test)

**Run 1 - naive_bayes**

| | pred not-ADE | pred ADE |
|---|---|---|
| **true not-ADE** | 2,207 | 286 |
| **true ADE** | 208 | 432 |

**Run 2 - logreg**

| | pred not-ADE | pred ADE |
|---|---|---|
| **true not-ADE** | 2,282 | 211 |
| **true ADE** | 210 | 430 |

**Run 2b - linear_svm**

| | pred not-ADE | pred ADE |
|---|---|---|
| **true not-ADE** | 2,261 | 232 |
| **true ADE** | 190 | 450 |

## Most informative features

Top-weighted 1-2grams from the linear models. This is a sanity check as much as a figure: if the ADE side were dominated by function words, the tokenizer or the split would be wrong.

**Run 2 - logreg**

| Direction | Features |
|---|---|
| toward **ADE** | `neurotoxicity`, `after`, `induced`, `hypersensitivity`, `developed`, `toxicity`, `following`, `taking`, `receiving`, `hepatotoxicity`, `pulmonary fibrosis`, `MTX` |
| toward **not-ADE** | `was`, `the`, `were`, `chemotherapy`, `agents`, `transplantation`, `improvement`, `corticosteroids`, `steroid`, `showed`, `surgery`, `this` |

**Run 2b - linear_svm**

| Direction | Features |
|---|---|
| toward **ADE** | `neurotoxicity`, `after`, `induced`, `hypersensitivity`, `developed`, `toxicity`, `receiving`, `following`, `taking`, `hepatotoxicity`, `methotrexate`, `during` |
| toward **not-ADE** | `was`, `were`, `the`, `chemotherapy`, `agents`, `transplantation`, `corticosteroids`, `improvement`, `steroid`, `this`, `surgery`, `on a` |

## Reading these numbers

These are the floor the embedding ablation has to clear. A BiLSTM that fails to beat TF-IDF + LogReg is not evidence that embeddings do not help - it is evidence that the BiLSTM is undertrained, and should be diagnosed as such before any conclusion is drawn from runs 3-6.

Vectorisation cost 3.7s and all three models fit in seconds on CPU. That asymmetry belongs in the report: the transformer runs cost roughly four orders of magnitude more compute for the margin recorded in `stage1_results.md`.
