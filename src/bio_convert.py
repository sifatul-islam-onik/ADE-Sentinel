"""Steps 5.1 - character spans -> BIO tags.

This is the single most dangerous function in the project. Off-by-one offset
bugs do not crash and do not show up in the loss curve; they produce training
labels that are quietly wrong, and the first symptom is an entity-F1 that looks
plausible but is measuring nothing. PLAN F5.

Two defects in the PRD's reference implementation (section 8.2) are fixed here.

**1. Containment instead of overlap.** The PRD tests `ts >= s and te <= e`, which
only tags tokens lying *entirely inside* the span. Any token straddling a
boundary is dropped. The ADE corpus contains such spans - an annotation ending
mid-token, or starting mid-token - and containment silently loses them. This
module tests overlap (`ts < e and te > s`) and reports how many spans needed
boundary snapping so the number is visible rather than assumed to be zero.

**2. Zero-match spans fail silently.** If offsets and tokenisation disagree
entirely, the PRD version returns all-`O` and training proceeds happily on empty
labels. Here, a span that tags no token raises by default.

The B-/I- distinction is assigned per span, not per label: two adjacent DRUG
entities must produce `B-DRUG B-DRUG`, not `B-DRUG I-DRUG`, or the entity-level
scorer will merge them into one and the count will be wrong.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.tokenizer import tokenize

Span = tuple[int, int, str]


class BIOConversionError(ValueError):
    """A span could not be aligned to any token."""


def resolve_overlaps(
    spans: list[Span],
    policy: str = "longest",
) -> tuple[list[Span], list[Span]]:
    """Drop spans that overlap an accepted one, returning (kept, dropped).

    NESTED ENTITIES. The ADE corpus annotates drug names *inside* effect phrases:

        'theophylline intoxication'      EFFECT [22:47]
        'theophylline'                   DRUG   [22:34]   <- nested
        'high blood lead level'          EFFECT [67:88]
        'lead'                           DRUG   [78:82]   <- nested

    Flat BIO cannot represent nesting: one token carries one tag. Letting the
    inner span overwrite the middle of the outer one orphans the remainder into
    an `I-EFFECT` with no `B-EFFECT`, which is a structurally illegal sequence.
    Gold labels containing illegal sequences would also destroy the CRF ablation
    in step 5.8, whose whole premise is that gold has zero and the softmax
    tagger's output does not.

    So overlaps are resolved before tagging rather than papered over afterwards:

      "longest" - the longer span wins, so `theophylline intoxication` (the full
                  adverse event) is kept over the bare drug name. Preferred: the
                  outer phrase is the more complete annotation, and the drug is
                  usually also annotated elsewhere in the same sentence.
      "first"   - earliest start wins; kept for ablation.

    The properly general fix is one BIO plane per entity type, since all observed
    nesting is DRUG-inside-EFFECT. That needs a two-head tagger, which is out of
    scope for the PRD's model ladder - so the loss is measured and reported
    instead of being hidden.
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
    """Per-corpus diagnostics. Zero snapping is a claim, not an assumption."""
    sentences: int = 0
    spans: int = 0
    entities_tagged: int = 0
    spans_snapped: int = 0          # span boundary did not fall on a token boundary
    spans_unmatched: int = 0        # span aligned to no token at all
    spans_dropped_nested: int = 0   # nested/overlapping span removed before tagging
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
    """Convert character spans to BIO tags over the domain tokenizer's tokens.

    Args:
        text: the sentence.
        spans: (start_char, end_char, LABEL) triples. End is exclusive.
        stats: optional accumulator; boundary snapping and misses are recorded.
        strict: raise if a span matches no token. Set False only for corpus-wide
            surveys where you want the count rather than the exception.
        overlap_policy: how to resolve nested spans; see `resolve_overlaps`.

    Returns:
        (tokens, tags), equal length. The tag sequence is always well-formed
        BIO - no `I-X` without a preceding `B-X`.
    """
    tokens, offsets = tokenize(text, lower=True)
    tags = ["O"] * len(tokens)

    if stats is not None:
        stats.sentences += 1
        stats.spans += len(spans)

    # Nested annotations must be resolved BEFORE tagging. Overwriting part of an
    # already-tagged entity is what produces illegal gold sequences.
    spans, dropped = resolve_overlaps(spans, overlap_policy)
    if stats is not None and dropped:
        stats.spans_dropped_nested += len(dropped)
        for d_start, d_end, d_label in dropped:
            stats.dropped_by_label[d_label] += 1
            if len(stats.examples_dropped) < 10:
                stats.examples_dropped.append(
                    f"{d_label} {text[d_start:d_end]!r} nested in {text[:60]!r}")

    for start, end, label in spans:
        matched = [
            i for i, (ts, te) in enumerate(offsets)
            # OVERLAP, not containment. A token that straddles the boundary
            # belongs to the entity; dropping it truncates the span.
            if ts < end and te > start
        ]

        if not matched:
            snippet = f"{text[start:end]!r} at [{start}:{end}] in {text[:70]!r}"
            if stats is not None:
                stats.spans_unmatched += 1
                if len(stats.examples_unmatched) < 10:
                    stats.examples_unmatched.append(snippet)
            if strict:
                raise BIOConversionError(f"span matched no token: {snippet}")
            continue

        # Did the span's edges land on token boundaries? If not, the tagged
        # region is wider than the annotation and that is worth counting.
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
    """Decode BIO back into (start_token, end_token_exclusive, label) entities.

    Used by the CRF ablation and by the round-trip tests. `I-X` with no preceding
    `B-X` starts a new entity here rather than being discarded, which is what
    makes the illegal-sequence count in step 5.8 meaningful: the decoder must not
    quietly repair what the model got wrong.
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


def count_illegal_transitions(tags: list[str]) -> int:
    """`I-X` not preceded by `B-X` or `I-X` - structurally impossible sequences.

    This is the number the CRF ablation reports (step 5.8). A per-token softmax
    can emit these freely because it decides each tag independently; a CRF learns
    transition scores and Viterbi decoding makes them vanishingly unlikely.
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
