"""Stage 1 metrics, computed once and imported everywhere.

Runs 1-8 appear in one table, so they have to be scored by identical code. Two
implementations of "macro-F1" that differ in how they treat a class with no
predicted examples produce a table whose rows are not comparable — and the
discrepancy is invisible, because it looks like a modelling result.

Macro-F1 is primary. Accuracy is recorded but never used to choose anything:
at this class balance, always answering "not ADE" scores 79.6%.

(Stage 2 is scored by entity, not by sentence — that lives in `src/bio.py`.)
"""

from __future__ import annotations

CLASS_NAMES = ("not_ade", "ade")


def classification_metrics(y_true, y_pred, scores=None) -> dict:
    """Every Stage 1 metric in one flat dict, ready for `log_run(metrics=...)`.

    `scores` is the positive-class probability, used for PR-AUC — the metric
    that survives the class imbalance without having to pick a threshold. Leave
    it out and PR-AUC is simply absent rather than wrong.
    """
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix,
        precision_recall_fscore_support,
    )

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "macro_f1": float(f1.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": [[int(v) for v in row] for row in cm],
    }
    for i, name in enumerate(CLASS_NAMES):
        metrics[f"precision_{name}"] = float(p[i])
        metrics[f"recall_{name}"] = float(r[i])
        metrics[f"f1_{name}"] = float(f1[i])
        metrics[f"support_{name}"] = int(support[i])

    if scores is not None:
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))

    return metrics


def macro_f1(y_true, y_pred) -> float:
    """Just the selection criterion — used by early stopping, every epoch."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
