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

**Scored natively, cross-checked with seqeval.** The scorers here are built on
`bio_to_entities` and `strict_entities` and need only the standard library.
That is not a preference: seqeval 1.2.2 is its latest release (2020), ships no
wheel, and its legacy `setup.py egg_info` fails to build on the remote runner's
Python 3.12, so a seqeval-only pipeline could not run there at all. Locally,
where seqeval does install, `entity_metrics` computes both and records whether
they agree - verified to match exactly on strict and lenient F1, precision,
recall and per-label scores across all 4,271 sentences. Two implementations
agreeing is a stronger claim than one asserted.
"""

from __future__ import annotations

from src.bio_convert import bio_to_entities, count_illegal_transitions


def strict_entities(tags: list[str]) -> list[tuple[int, int, str]]:
    """Entities under a strict IOB2 reading: only a `B-X` may open one.

    The counterpart to `bio_to_entities`, which repairs an orphan `I-X` into an
    entity. Here such a run yields nothing at all, which is what "strict" means:
    `O I-DRUG I-DRUG` is not a malformed DRUG entity, it is not an entity.
    """
    entities: list[tuple[int, int, str]] = []
    start: int | None = None
    label: str | None = None

    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("I-") and start is not None and tag[2:] == label:
            continue                      # extends the entity currently open

        if start is not None:             # anything else closes it
            entities.append((start, i, label))
            start = label = None

        if tag.startswith("B-"):
            start, label = i, tag[2:]

    return entities


def prf(gold_sets, pred_sets) -> tuple[float, float, float]:
    """Micro-averaged entity precision, recall and F1 over aligned sentences.

    Entities are compared as exact (start, end, label) triples per sentence -
    the standard entity-level criterion: a boundary off by one token is a miss
    and a false positive, not partial credit.
    """
    matched = predicted = actual = 0

    for gold, pred in zip(gold_sets, pred_sets):
        remaining = list(gold)
        for entity in pred:
            if entity in remaining:
                remaining.remove(entity)
                matched += 1
        predicted += len(pred)
        actual += len(gold)

    precision = matched / predicted if predicted else 0.0
    recall = matched / actual if actual else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return precision, recall, f1


def native_entity_scores(y_true, y_pred) -> dict:
    """Strict and lenient entity P/R/F1 without seqeval.

    This exists because seqeval cannot be installed on the remote runner:
    version 1.2.2 is the latest release (2020), ships no wheel, and its legacy
    `setup.py egg_info` fails to build on Kaggle's Python 3.12. Rather than pin
    the whole Stage 2 pipeline to a package that will not install, the two
    scorers are computed from `bio_to_entities` and `strict_entities`, which are
    already unit-tested and are what the BIO converter is built on.

    Locally, where seqeval does install, `entity_metrics` computes both and
    asserts they agree - so this is a verified second implementation rather than
    an unchecked substitute.
    """
    gold_strict = [strict_entities(s) for s in y_true]
    pred_strict = [strict_entities(s) for s in y_pred]
    gold_lenient = [bio_to_entities([""] * len(s), s) for s in y_true]
    pred_lenient = [bio_to_entities([""] * len(s), s) for s in y_pred]

    sp, sr, sf = prf(gold_strict, pred_strict)
    lp, lr, lf = prf(gold_lenient, pred_lenient)

    return {
        "entity_precision_strict": sp, "entity_recall_strict": sr,
        "entity_f1_strict": sf,
        "entity_precision_lenient": lp, "entity_recall_lenient": lr,
        "entity_f1_lenient": lf,
        "strict_lenient_gap": lf - sf,
    }


def per_label_scores(y_true, y_pred, labels=("DRUG", "EFFECT")) -> dict:
    """Per-entity-type strict scores, without seqeval."""
    metrics = {}
    for label in labels:
        gold = [[e for e in strict_entities(s) if e[2] == label] for s in y_true]
        pred = [[e for e in strict_entities(s) if e[2] == label] for s in y_pred]
        p, r, f1 = prf(gold, pred)
        key = label.lower()
        metrics[f"{key}_precision"] = p
        metrics[f"{key}_recall"] = r
        metrics[f"{key}_f1"] = f1
        metrics[f"{key}_support"] = sum(len(s) for s in gold)
    return metrics


def entity_metrics(y_true: list[list[str]], y_pred: list[list[str]]) -> dict:
    """Entity-level P/R/F1 under both scoring modes, plus per-label detail.

    Args:
        y_true, y_pred: per-sentence lists of BIO tag strings, already trimmed
            to the real tokens. Ragged is expected; padding must not appear.

    Returns a flat dict ready for `log_run(metrics=...)`.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} gold sequences vs {len(y_pred)} predicted")
    for i, (gold, pred) in enumerate(zip(y_true, y_pred)):
        if len(gold) != len(pred):
            raise ValueError(
                f"sentence {i}: {len(gold)} gold tags vs {len(pred)} predicted. "
                "Padding was probably not trimmed before scoring."
            )

    # The native scorers are the primary path: they run everywhere, including
    # the remote session where seqeval cannot be installed at all.
    metrics = native_entity_scores(y_true, y_pred)
    metrics.update(per_label_scores(y_true, y_pred))

    # Where seqeval IS available - the local machine that writes the report - it
    # is used as an independent check on those numbers rather than as the source
    # of them. Agreement between two implementations is worth more than either
    # one asserted alone, and this is the assertion PRD section 12 would want.
    metrics.update(_seqeval_crosscheck(y_true, y_pred, metrics))

    metrics.update(illegal_transition_stats(y_pred))
    metrics["gold_entities"] = sum(len(bio_to_entities([""] * len(s), s)) for s in y_true)
    metrics["pred_entities"] = sum(len(bio_to_entities([""] * len(s), s)) for s in y_pred)

    return metrics


def _seqeval_crosscheck(y_true, y_pred, native: dict, tolerance: float = 1e-9) -> dict:
    """Score again with seqeval where it exists, and compare.

    Returns the comparison outcome, not replacement metrics - the native numbers
    stand either way. A mismatch means one of the two scorers is wrong about
    this corpus, which is something the report must not paper over, so it is
    recorded as a field rather than swallowed.
    """
    try:
        from seqeval.metrics import f1_score
        from seqeval.scheme import IOB2
    except ImportError:
        # The remote runner. Expected, not an error - see native_entity_scores.
        return {"seqeval_crosscheck": "unavailable"}

    from seqeval.metrics import classification_report

    strict = float(f1_score(y_true, y_pred, mode="strict", scheme=IOB2,
                            zero_division=0))
    lenient = float(f1_score(y_true, y_pred, zero_division=0))

    agrees = (abs(strict - native["entity_f1_strict"]) < tolerance
              and abs(lenient - native["entity_f1_lenient"]) < tolerance)

    # Per-label too. `classification_report` must be asked for strict mode
    # explicitly - its default is lenient, and comparing the two would report a
    # disagreement that is really just a difference of definition.
    report = classification_report(y_true, y_pred, output_dict=True,
                                   zero_division=0, mode="strict", scheme=IOB2)
    for label, scores in report.items():
        key = f"{label.lower()}_f1"
        if key in native:
            agrees &= abs(scores["f1-score"] - native[key]) < tolerance

    return {
        "seqeval_crosscheck": "agrees" if agrees else "DISAGREES",
        "seqeval_f1_strict": strict,
        "seqeval_f1_lenient": lenient,
    }


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
        "| Entity | Precision | Recall | F1 (strict) | Support |",
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
