"""Steps 5.7-5.8 - entity-level scoring for the Stage 2 taggers.

Two things the PRD asks for, and one it gets nearly free.

**Strict and lenient, both reported (5.7).** `seqeval` in default mode is
forgiving about scheme violations: a sequence like `O I-DRUG I-DRUG` is read as
one DRUG entity, repairing what the model actually emitted. In
`mode='strict', scheme=IOB2` the same sequence is not a valid entity and scores
nothing. Reporting only the lenient number flatters a per-token softmax
specifically, because that is the model which produces malformed sequences -
which would quietly undercut the very comparison run 10 exists to make. The gap
between the two *is* the PRD's "partial vs strict" stretch goal, obtained by
scoring twice rather than by any extra modelling.

**Illegal transitions (5.8).** The count of `I-X` not preceded by `B-X` or
`I-X`. This is the concrete evidence for the CRF: entity-F1 usually moves only a
little between runs 9 and 10, so on F1 alone the CRF looks like an expensive
no-op. The illegal count is where the structural difference actually shows, and
it explains the direction of the strict-vs-lenient gap at the same time.

`seqeval` is in `requirements-local.txt`, so every number here is reproducible
on the machine that writes the report - the remote session does not have to be
trusted for the scoring, only for the training.
"""

from __future__ import annotations

from src.bio_convert import bio_to_entities, count_illegal_transitions


def entity_metrics(y_true: list[list[str]], y_pred: list[list[str]]) -> dict:
    """Entity-level P/R/F1 under both scoring modes, plus per-label detail.

    Args:
        y_true, y_pred: per-sentence lists of BIO tag strings, already trimmed
            to the real tokens. Ragged is expected; padding must not appear.

    Returns a flat dict ready for `log_run(metrics=...)`.
    """
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from seqeval.scheme import IOB2

    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} gold sequences vs {len(y_pred)} predicted")
    for i, (gold, pred) in enumerate(zip(y_true, y_pred)):
        if len(gold) != len(pred):
            raise ValueError(
                f"sentence {i}: {len(gold)} gold tags vs {len(pred)} predicted. "
                "Padding was probably not trimmed before scoring."
            )

    # zero_division=0 is explicit rather than left to the default: a tagger that
    # predicts no entities at all is a real early-epoch state, and it should
    # score 0.0 rather than emit a warning and score 0.0.
    metrics = {
        "entity_f1_lenient": float(f1_score(y_true, y_pred, zero_division=0)),
        "entity_precision_lenient": float(precision_score(y_true, y_pred, zero_division=0)),
        "entity_recall_lenient": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    # Strict mode raises on tag sets it does not recognise rather than scoring
    # zero, so a scheme mismatch is loud instead of silently halving the result.
    strict = dict(mode="strict", scheme=IOB2, zero_division=0)
    metrics.update({
        "entity_f1_strict": float(f1_score(y_true, y_pred, **strict)),
        "entity_precision_strict": float(precision_score(y_true, y_pred, **strict)),
        "entity_recall_strict": float(recall_score(y_true, y_pred, **strict)),
    })
    metrics["strict_lenient_gap"] = (
        metrics["entity_f1_lenient"] - metrics["entity_f1_strict"]
    )

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for label, scores in report.items():
        if label in ("micro avg", "macro avg", "weighted avg"):
            continue
        key = label.lower()
        metrics[f"{key}_precision"] = float(scores["precision"])
        metrics[f"{key}_recall"] = float(scores["recall"])
        metrics[f"{key}_f1"] = float(scores["f1-score"])
        metrics[f"{key}_support"] = int(scores["support"])

    metrics.update(illegal_transition_stats(y_pred))
    metrics["gold_entities"] = sum(len(bio_to_entities([""] * len(s), s)) for s in y_true)
    metrics["pred_entities"] = sum(len(bio_to_entities([""] * len(s), s)) for s in y_pred)

    return metrics


def illegal_transition_stats(sequences: list[list[str]]) -> dict:
    """Step 5.8: how often the model emitted a structurally impossible tag.

    Reported three ways because they answer different questions: the raw count
    scales with corpus size, the per-sentence rate is what a reader can picture,
    and the affected-sentence count says whether the problem is concentrated in
    a few sequences or spread across many.
    """
    counts = [count_illegal_transitions(seq) for seq in sequences]
    total = sum(counts)
    affected = sum(1 for c in counts if c)

    return {
        "illegal_transitions": total,
        "illegal_sentences": affected,
        "illegal_sentence_rate": affected / len(sequences) if sequences else 0.0,
    }


def token_accuracy(y_true: list[list[str]], y_pred: list[list[str]]) -> float:
    """Reported but deprecated, exactly as accuracy is in Stage 1.

    `O` is 79% of tokens here, so a model that predicted nothing at all would
    score about 0.79. It is in the table only to be argued against.
    """
    correct = total = 0
    for gold, pred in zip(y_true, y_pred):
        for g, p in zip(gold, pred):
            correct += g == p
            total += 1
    return correct / total if total else 0.0


def format_entity_table(metrics: dict, labels=("drug", "effect")) -> list[str]:
    """Per-label entity scores as markdown rows, for the report figures."""
    lines = [
        "| Entity | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for label in labels:
        if f"{label}_f1" not in metrics:
            continue
        lines.append(
            f"| {label.upper()} | {metrics[f'{label}_precision']:.3f} | "
            f"{metrics[f'{label}_recall']:.3f} | {metrics[f'{label}_f1']:.3f} | "
            f"{metrics[f'{label}_support']:,} |"
        )
    return lines
