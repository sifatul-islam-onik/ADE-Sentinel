"""BIO tagging: character spans <-> per-word tags, and entity-level scoring.

The corpus says "characters 12-24 of this sentence are a DRUG". A tagger needs
one label per word:

    tokens   a  case  of  toxic    hepatitis  caused  by  methotrexate
    tags     O  O     O   B-EFFECT I-EFFECT   O       O   B-DRUG

`to_bio` does that conversion; `strict_entities` does the reverse, reading a tag
sequence back into entities. `entity_prf` scores predicted entities against gold
ones.

Two details worth knowing, because both are easy to get silently wrong:

* A token straddling a span boundary belongs to the entity. Testing for
  containment instead of overlap quietly drops such tokens, and nothing crashes.
* B-/I- is assigned per span, not per label. Two adjacent DRUG entities must
  give `B-DRUG B-DRUG`, not `B-DRUG I-DRUG`, or the scorer merges them into one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.tokenizer import tokenize

Span = tuple[int, int, str]

# The tag inventory, in a FIXED order. Every tagger maps tags to integer ids by
# this order, so a model trained under one ordering would score as noise under
# another. It lives here, next to the converter that produces the tags.
ENTITY_LABELS: tuple[str, ...] = ("DRUG", "EFFECT")
TAGS: tuple[str, ...] = ("O",) + tuple(
    f"{prefix}-{label}" for label in ENTITY_LABELS for prefix in ("B", "I")
)
TAG_TO_ID: dict[str, int] = {tag: i for i, tag in enumerate(TAGS)}


class BIOConversionError(ValueError):
    """A span could not be aligned to any token."""


def resolve_overlaps(spans: list[Span], policy: str = "longest") -> tuple[list[Span], list[Span]]:
    """Drop spans overlapping an accepted one. Returns (kept, dropped).

    The corpus annotates drug names *inside* effect phrases:

        'theophylline intoxication'   EFFECT [22:47]
        'theophylline'                DRUG   [22:34]   <- nested

    One word can only carry one tag, so flat BIO cannot represent that. Letting
    the inner span overwrite the middle of the outer one leaves an `I-EFFECT`
    with no `B-EFFECT` in front of it - an impossible sequence, in the gold
    labels. So overlaps are resolved first: the longer span wins, keeping the
    full adverse event, and the drug is usually annotated elsewhere anyway.
    """
    if policy == "longest":
        ordered = sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0]))
    elif policy == "first":
        ordered = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    else:
        raise ValueError(f"unknown overlap policy: {policy!r}")

    kept: list[Span] = []
    dropped: list[Span] = []
    for span in ordered:
        start, end, _ = span
        if any(start < k_end and end > k_start for k_start, k_end, _ in kept):
            dropped.append(span)
        else:
            kept.append(span)

    return sorted(kept), sorted(dropped)


@dataclass
class ConversionStats:
    """Per-corpus diagnostics, so "nothing was lost" is measured, not assumed."""
    sentences: int = 0
    spans: int = 0
    entities_tagged: int = 0
    spans_snapped: int = 0          # span edge did not land on a word boundary
    spans_unmatched: int = 0        # span aligned to no word at all
    spans_dropped_nested: int = 0   # nested span removed before tagging
    dropped_by_label: Counter = field(default_factory=Counter)
    examples_dropped: list[str] = field(default_factory=list)
    examples_snapped: list[str] = field(default_factory=list)
    examples_unmatched: list[str] = field(default_factory=list)


def to_bio(
    text: str,
    spans: list[Span],
    *,
    stats: ConversionStats | None = None,
    strict: bool = True,
    overlap_policy: str = "longest",
) -> tuple[list[str], list[str]]:
    """Convert character spans to BIO tags over the project tokenizer's words.

    Args:
        text: the sentence.
        spans: (start_char, end_char, LABEL) triples, end exclusive.
        stats: optional accumulator; snapping and misses are recorded into it.
        strict: raise if a span matches no word. False for corpus-wide surveys
            where the count is wanted instead of the exception.

    Returns (tokens, tags), equal length and always well-formed.
    """
    tokens, offsets = tokenize(text, lower=True)
    tags = ["O"] * len(tokens)

    if stats is not None:
        stats.sentences += 1
        stats.spans += len(spans)

    spans, dropped = resolve_overlaps(spans, overlap_policy)
    if stats is not None and dropped:
        stats.spans_dropped_nested += len(dropped)
        for d_start, d_end, d_label in dropped:
            stats.dropped_by_label[d_label] += 1
            if len(stats.examples_dropped) < 10:
                stats.examples_dropped.append(
                    f"{d_label} {text[d_start:d_end]!r} nested in {text[:60]!r}")

    for start, end, label in spans:
        # OVERLAP, not containment - see the module docstring.
        matched = [i for i, (ts, te) in enumerate(offsets) if ts < end and te > start]

        if not matched:
            snippet = f"{text[start:end]!r} at [{start}:{end}] in {text[:70]!r}"
            if stats is not None:
                stats.spans_unmatched += 1
                if len(stats.examples_unmatched) < 10:
                    stats.examples_unmatched.append(snippet)
            if strict:
                raise BIOConversionError(f"span matched no token: {snippet}")
            continue

        first, last = matched[0], matched[-1]
        if offsets[first][0] != start or offsets[last][1] != end:
            if stats is not None:
                stats.spans_snapped += 1
                if len(stats.examples_snapped) < 10:
                    stats.examples_snapped.append(
                        f"{text[start:end]!r} -> {' '.join(tokens[first:last + 1])!r}")

        if stats is not None:
            stats.entities_tagged += 1

        for n, i in enumerate(matched):
            tags[i] = ("B-" if n == 0 else "I-") + label

    return tokens, tags


def bio_to_entities(tokens: list[str], tags: list[str]) -> list[tuple[int, int, str]]:
    """Decode BIO into (start_word, end_word_exclusive, label), repairing errors.

    An `I-X` with no `B-X` in front of it opens a new entity here rather than
    being thrown away. That is the *lenient* reading. Use `strict_entities` for
    the headline metric.
    """
    entities: list[tuple[int, int, str]] = []
    start: int | None = None
    label: str | None = None

    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            if start is not None:
                entities.append((start, i, label))
            start, label = i, tag[2:]
        elif tag.startswith("I-"):
            if start is None or tag[2:] != label:
                if start is not None:
                    entities.append((start, i, label))
                start, label = i, tag[2:]
        else:
            if start is not None:
                entities.append((start, i, label))
            start = label = None

    return entities


def strict_entities(tags: list[str]) -> list[tuple[int, int, str]]:
    """Entities under a strict reading: only a `B-X` may open one.

    `O I-DRUG I-DRUG` is not a malformed DRUG entity here - it is not an entity
    at all. This is the reading the headline entity-F1 uses, and the one the
    demo highlights with, so what you see marked is what the scorer counted.
    """
    entities: list[tuple[int, int, str]] = []
    start: int | None = None
    label: str | None = None

    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("I-") and start is not None and tag[2:] == label:
            continue                      # extends the open entity

        if start is not None:             # anything else closes it
            entities.append((start, i, label))
            start = label = None

        if tag.startswith("B-"):
            start, label = i, tag[2:]

    return entities


def count_illegal_transitions(tags: list[str]) -> int:
    """`I-X` not preceded by `B-X` or `I-X` - sequences that cannot be an entity.

    This is the number the CRF ablation in notebook 05 reports. A per-word
    softmax emits these freely because it decides each position independently.
    """
    illegal = 0
    previous = "O"
    for tag in tags:
        if tag.startswith("I-"):
            label = tag[2:]
            if previous not in (f"B-{label}", f"I-{label}"):
                illegal += 1
        previous = tag
    return illegal


def entity_prf(gold_sets, pred_sets) -> tuple[float, float, float]:
    """Micro-averaged entity precision, recall and F1 over aligned sentences.

    Entities are compared as exact (start, end, label) triples: a boundary off
    by one word is a miss *and* a false positive, not partial credit.
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
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def entity_f1(y_true: list[list[str]], y_pred: list[list[str]]) -> dict:
    """Strict entity P/R/F1 over lists of tag sequences."""
    gold = [strict_entities(t) for t in y_true]
    pred = [strict_entities(p) for p in y_pred]
    precision, recall, f1 = entity_prf(gold, pred)
    return {"precision": precision, "recall": recall, "f1": f1,
            "gold_entities": sum(len(g) for g in gold),
            "predicted_entities": sum(len(p) for p in pred)}
