"""Stage 1 metrics, computed once and imported everywhere.

Runs 1-8 are compared directly in one table, so they must be scored by identical
code. Two implementations of "macro-F1" that differ in how they handle a class
with no predicted examples produce a table whose rows are not comparable, and
the discrepancy is essentially invisible - it looks like a modelling result.

The metric set is PRD section 8.1: macro-F1 primary, per-class precision/recall,
PR-AUC, confusion matrix, and accuracy reported but deprecated.
"""

from __future__ import annotations

CLASS_NAMES = ("not_ade", "ade")


def classification_metrics(y_true, y_pred, scores=None) -> dict:
    """Every Stage 1 metric in one flat dict, ready for `log_run(metrics=...)`.

    `scores` is the positive-class score (probability or margin) used for PR-AUC,
    which is the metric that survives the class imbalance without a threshold
    choice. Omit it and PR-AUC is simply absent rather than wrong.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
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
    """Just the selection criterion - used by early stopping every epoch."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def format_confusion(cm) -> list[str]:
    """The confusion matrix as markdown table lines, for the report figures."""
    return [
        "| | pred not-ADE | pred ADE |",
        "|---|---|---|",
        f"| **true not-ADE** | {cm[0][0]:,} | {cm[0][1]:,} |",
        f"| **true ADE** | {cm[1][0]:,} | {cm[1][1]:,} |",
    ]
