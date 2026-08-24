"""Tests for char-span -> BIO conversion (PLAN F5, step 5.2).

Written before the corpus-wide run, because this is the failure mode that does
not announce itself: wrong labels train quietly and the loss curve looks fine.

The two tests that matter most are `test_mid_token_span_is_not_dropped` and
`test_span_matching_no_token_raises` - they encode the exact defects in the
PRD's reference implementation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bio_convert import (  # noqa: E402
    BIOConversionError, ConversionStats, bio_to_entities, count_illegal_transitions,
    resolve_overlaps, to_bio,
)


# --------------------------------------------------------------------------
# Basic alignment
# --------------------------------------------------------------------------

def test_exact_boundary_span():
    text = "Hepatotoxicity developed after methotrexate."
    #       0123456789...
    spans = [(0, 14, "EFFECT"), (31, 43, "DRUG")]
    tokens, tags = to_bio(text, spans)
    assert tokens[0] == "hepatotoxicity"
    assert tags[0] == "B-EFFECT"
    assert "B-DRUG" in tags
    assert tags.count("B-DRUG") == 1


def test_multi_token_span_gets_B_then_I():
    text = "The patient developed chronic obstructive pulmonary disease."
    start = text.index("chronic")
    end = text.index("disease") + len("disease")
    tokens, tags = to_bio(text, [(start, end, "EFFECT")])
    entity_tags = [t for t in tags if t != "O"]
    assert entity_tags[0] == "B-EFFECT"
    assert all(t == "I-EFFECT" for t in entity_tags[1:])
    assert len(entity_tags) == 4


def test_tokens_and_tags_are_the_same_length():
    text = "5-fluorouracil 20 mg/kg caused rash (P<0.05)."
    tokens, tags = to_bio(text, [(0, 14, "DRUG")])
    assert len(tokens) == len(tags)


# --------------------------------------------------------------------------
# PRD defect 1: containment drops boundary-straddling tokens
# --------------------------------------------------------------------------

def test_mid_token_span_is_not_dropped():
    """A span ending mid-token must still tag that token.

    The PRD's containment test (`ts >= s and te <= e`) yields NO tags here,
    silently deleting the entity. Overlap keeps it.
    """
    text = "Severe hepatotoxicity was observed."
    start = text.index("hepato")
    end = start + len("hepatotox")          # deliberately mid-token
    tokens, tags = to_bio(text, [(start, end, "EFFECT")])
    assert "B-EFFECT" in tags, (tokens, tags)
    assert tokens[tags.index("B-EFFECT")] == "hepatotoxicity"


def test_span_starting_mid_token_is_not_dropped():
    text = "Post-methotrexate toxicity appeared."
    start = text.index("methotrexate")
    end = start + len("methotrexate")
    tokens, tags = to_bio(text, [(start, end, "DRUG")])
    assert any(t.endswith("DRUG") for t in tags), (tokens, tags)


def test_snapping_is_counted_not_hidden():
    text = "Severe hepatotoxicity was observed."
    start = text.index("hepato")
    stats = ConversionStats()
    to_bio(text, [(start, start + 9, "EFFECT")], stats=stats)
    assert stats.spans_snapped == 1
    assert stats.examples_snapped


def test_exact_span_is_not_counted_as_snapped():
    text = "Hepatotoxicity developed."
    stats = ConversionStats()
    to_bio(text, [(0, 14, "EFFECT")], stats=stats)
    assert stats.spans_snapped == 0


# --------------------------------------------------------------------------
# PRD defect 2: zero-match spans must not pass silently
# --------------------------------------------------------------------------

def test_span_matching_no_token_raises():
    """The PRD version returns all-O here and trains on empty labels."""
    text = "Hepatotoxicity developed."
    with pytest.raises(BIOConversionError):
        to_bio(text, [(500, 520, "DRUG")])


def test_unmatched_span_can_be_surveyed_without_raising():
    text = "Hepatotoxicity developed."
    stats = ConversionStats()
    tokens, tags = to_bio(text, [(500, 520, "DRUG")], stats=stats, strict=False)
    assert stats.spans_unmatched == 1
    assert set(tags) == {"O"}


def test_whitespace_only_span_raises():
    text = "Rash   developed."
    with pytest.raises(BIOConversionError):
        to_bio(text, [(4, 7, "EFFECT")])       # the gap between words


# --------------------------------------------------------------------------
# Adjacency: two entities must not merge
# --------------------------------------------------------------------------

def test_adjacent_same_label_entities_stay_separate():
    """`B-DRUG B-DRUG`, never `B-DRUG I-DRUG` - otherwise seqeval counts one."""
    text = "aspirin ibuprofen were given"
    tokens, tags = to_bio(text, [(0, 7, "DRUG"), (8, 17, "DRUG")])
    assert tags[0] == "B-DRUG"
    assert tags[1] == "B-DRUG"
    assert len(bio_to_entities(tokens, tags)) == 2


def test_multi_relation_sentence_keeps_all_spans():
    """The naproxen/oxaprozin case: 2 drugs + 2 effects from 4 relation rows."""
    text = ("A man taking naproxen and a woman on oxaprozin presented with "
            "tense bullae and cutaneous fragility.")
    spans = [
        (text.index("naproxen"), text.index("naproxen") + 8, "DRUG"),
        (text.index("oxaprozin"), text.index("oxaprozin") + 9, "DRUG"),
        (text.index("tense bullae"), text.index("tense bullae") + 12, "EFFECT"),
        (text.index("cutaneous fragility"),
         text.index("cutaneous fragility") + 19, "EFFECT"),
    ]
    tokens, tags = to_bio(text, spans)
    entities = bio_to_entities(tokens, tags)
    assert len(entities) == 4
    assert sum(1 for e in entities if e[2] == "DRUG") == 2
    assert sum(1 for e in entities if e[2] == "EFFECT") == 2


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------

def test_round_trip_recovers_entity_count():
    text = "Methotrexate caused severe hepatotoxicity and rash."
    spans = [
        (0, 12, "DRUG"),
        (text.index("severe hepatotoxicity"), text.index("severe hepatotoxicity") + 21, "EFFECT"),
        (text.index("rash"), text.index("rash") + 4, "EFFECT"),
    ]
    tokens, tags = to_bio(text, spans)
    assert len(bio_to_entities(tokens, tags)) == len(spans)


def test_no_entities_gives_all_O():
    tokens, tags = to_bio("The patient recovered fully.", [])
    assert set(tags) == {"O"}
    assert bio_to_entities(tokens, tags) == []


# --------------------------------------------------------------------------
# Illegal transitions - the CRF ablation metric (step 5.8)
# --------------------------------------------------------------------------

def test_illegal_transition_detected():
    assert count_illegal_transitions(["O", "I-DRUG", "O"]) == 1
    assert count_illegal_transitions(["B-DRUG", "I-DRUG"]) == 0
    assert count_illegal_transitions(["B-DRUG", "I-EFFECT"]) == 1
    assert count_illegal_transitions(["O", "O", "O"]) == 0


def test_valid_sequences_from_to_bio_are_never_illegal():
    """Gold labels must contain zero illegal transitions by construction."""
    text = "Methotrexate caused chronic obstructive pulmonary disease."
    start = text.index("chronic")
    _, tags = to_bio(text, [(0, 12, "DRUG"), (start, len(text) - 1, "EFFECT")])
    assert count_illegal_transitions(tags) == 0


# --------------------------------------------------------------------------
# Nested entities - the ADE corpus annotates drugs inside effect phrases
# --------------------------------------------------------------------------

def test_nested_drug_inside_effect_is_resolved():
    """'theophylline intoxication' EFFECT contains 'theophylline' DRUG.

    Flat BIO cannot hold both. Without resolution the inner span overwrites the
    middle of the outer one and orphans the tail into an illegal `I-EFFECT`.
    """
    text = "A case of theophylline intoxication was reported."
    outer = (text.index("theophylline"), text.index("intoxication") + 12, "EFFECT")
    inner = (text.index("theophylline"), text.index("theophylline") + 12, "DRUG")
    tokens, tags = to_bio(text, [outer, inner])
    assert count_illegal_transitions(tags) == 0
    assert "B-EFFECT" in tags and "I-EFFECT" in tags
    assert not any(t.endswith("DRUG") for t in tags)   # longest wins


def test_resolve_overlaps_longest_wins():
    spans = [(0, 25, "EFFECT"), (0, 12, "DRUG")]
    kept, dropped = resolve_overlaps(spans, "longest")
    assert kept == [(0, 25, "EFFECT")]
    assert dropped == [(0, 12, "DRUG")]


def test_resolve_overlaps_first_policy():
    spans = [(10, 20, "DRUG"), (5, 30, "EFFECT")]
    kept, _ = resolve_overlaps(spans, "first")
    assert kept == [(5, 30, "EFFECT")]


def test_resolve_overlaps_keeps_disjoint_spans():
    spans = [(0, 8, "DRUG"), (20, 30, "EFFECT"), (40, 44, "DRUG")]
    kept, dropped = resolve_overlaps(spans)
    assert len(kept) == 3 and dropped == []


def test_touching_spans_are_not_treated_as_overlapping():
    """End is exclusive, so [0,7) and [7,14) do not overlap."""
    kept, dropped = resolve_overlaps([(0, 7, "DRUG"), (7, 14, "EFFECT")])
    assert len(kept) == 2 and dropped == []


def test_unknown_overlap_policy_raises():
    with pytest.raises(ValueError):
        resolve_overlaps([(0, 5, "DRUG")], "nonsense")


def test_dropped_nested_spans_are_counted():
    text = "A case of theophylline intoxication was reported."
    outer = (text.index("theophylline"), text.index("intoxication") + 12, "EFFECT")
    inner = (text.index("theophylline"), text.index("theophylline") + 12, "DRUG")
    stats = ConversionStats()
    to_bio(text, [outer, inner], stats=stats)
    assert stats.spans_dropped_nested == 1
    assert stats.dropped_by_label["DRUG"] == 1
    assert stats.examples_dropped


# --------------------------------------------------------------------------
# The invariant: gold tags are always well-formed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("spans_factory", [
    lambda t: [(0, 12, "DRUG")],
    lambda t: [(0, 25, "EFFECT"), (0, 12, "DRUG")],
    lambda t: [(0, 12, "DRUG"), (5, 25, "EFFECT")],
    lambda t: [(0, 12, "DRUG"), (13, 25, "EFFECT"), (3, 20, "EFFECT")],
])
def test_gold_tags_never_contain_illegal_transitions(spans_factory):
    text = "theophylline intoxication and severe rash were reported."
    _, tags = to_bio(text, spans_factory(text), strict=False)
    assert count_illegal_transitions(tags) == 0, tags
