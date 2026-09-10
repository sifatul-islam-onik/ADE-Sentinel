"""Tests for steps 5.7-5.8 - entity scoring and the tag/id plumbing.

The strict-vs-lenient distinction is the point of most of these. Lenient seqeval
silently repairs malformed sequences, so a tagger that emits `O I-DRUG` still
scores an entity; strict mode does not. Runs 9 and 10 differ mainly in how often
they emit such sequences, so if the two modes were ever conflated the CRF
comparison would report nothing.
"""

from __future__ import annotations

import pytest

from src.bio_convert import TAG_TO_ID, TAGS, bio_to_entities, count_illegal_transitions
from src.models.encoding import encode_tokens_with_tags, load_vocab
from src.stage2_metrics import (
    entity_metrics,
    illegal_transition_stats,
    token_accuracy,
)


# --------------------------------------------------------------------------
# the tag inventory
# --------------------------------------------------------------------------

def test_tag_order_is_fixed():
    """A model trained under one ordering scores as noise under another, so this
    order is a contract between the three Stage 2 runs, not an implementation
    detail."""
    assert TAGS == ("O", "B-DRUG", "I-DRUG", "B-EFFECT", "I-EFFECT")
    assert TAG_TO_ID["O"] == 0
    assert all(TAG_TO_ID[t] == i for i, t in enumerate(TAGS))


# --------------------------------------------------------------------------
# strict vs lenient
# --------------------------------------------------------------------------

def test_lenient_repairs_what_strict_rejects():
    """`O I-EFFECT` is not a valid IOB2 entity, but lenient mode reads one."""
    gold = [["O", "B-EFFECT", "I-EFFECT"]]
    pred = [["O", "I-EFFECT", "I-EFFECT"]]

    m = entity_metrics(gold, pred)

    assert m["entity_f1_lenient"] == pytest.approx(1.0)
    assert m["entity_f1_strict"] == pytest.approx(0.0)
    assert m["strict_lenient_gap"] == pytest.approx(1.0)


def test_the_two_modes_agree_on_well_formed_output():
    gold = [["B-DRUG", "I-DRUG", "O", "B-EFFECT"]]
    pred = [["B-DRUG", "I-DRUG", "O", "B-EFFECT"]]

    m = entity_metrics(gold, pred)

    assert m["entity_f1_strict"] == pytest.approx(1.0)
    assert m["entity_f1_lenient"] == pytest.approx(1.0)
    assert m["strict_lenient_gap"] == pytest.approx(0.0)


def test_adjacent_entities_are_not_merged():
    """Two DRUGs back to back must count as two, or recall is overstated on
    every multi-drug sentence - which this corpus is full of."""
    gold = [["B-DRUG", "B-DRUG"]]

    assert len(bio_to_entities(["a", "b"], gold[0])) == 2

    m = entity_metrics(gold, [["B-DRUG", "I-DRUG"]])
    assert m["entity_f1_strict"] < 1.0


def test_per_label_scores_are_reported():
    gold = [["B-DRUG", "O", "B-EFFECT"]]
    pred = [["B-DRUG", "O", "O"]]

    m = entity_metrics(gold, pred)

    assert m["drug_f1"] == pytest.approx(1.0)
    assert m["effect_f1"] == pytest.approx(0.0)
    assert m["drug_support"] == 1


def test_entity_counts_are_reported():
    gold = [["B-DRUG", "I-DRUG", "O", "B-EFFECT"]]
    pred = [["B-DRUG", "O", "O", "O"]]

    m = entity_metrics(gold, pred)

    assert m["gold_entities"] == 2
    assert m["pred_entities"] == 1


# --------------------------------------------------------------------------
# guards against scoring padding
# --------------------------------------------------------------------------

def test_length_mismatch_raises():
    """Padding left in the predictions is the failure this catches. It would
    otherwise inflate the denominator and quietly lower every score."""
    with pytest.raises(ValueError, match="gold tags vs"):
        entity_metrics([["O", "B-DRUG"]], [["O", "B-DRUG", "O"]])


def test_sequence_count_mismatch_raises():
    with pytest.raises(ValueError, match="gold sequences vs"):
        entity_metrics([["O"], ["O"]], [["O"]])


# --------------------------------------------------------------------------
# step 5.8 - illegal transitions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tags,expected", [
    (["O", "B-DRUG", "I-DRUG"], 0),
    (["O", "I-DRUG"], 1),                       # I- with no B-
    (["B-DRUG", "I-EFFECT"], 1),                # I- of a different type
    (["I-DRUG", "I-DRUG"], 1),                  # only the first is illegal
    (["O", "I-DRUG", "O", "I-EFFECT"], 2),
])
def test_count_illegal_transitions(tags, expected):
    assert count_illegal_transitions(tags) == expected


def test_illegal_transition_stats_summarises_a_corpus():
    sequences = [["O", "I-DRUG"], ["B-DRUG", "I-DRUG"], ["I-EFFECT", "O", "I-DRUG"]]

    stats = illegal_transition_stats(sequences)

    assert stats["illegal_transitions"] == 3
    assert stats["illegal_sentences"] == 2
    assert stats["illegal_sentence_rate"] == pytest.approx(2 / 3)


def test_well_formed_corpus_has_no_illegal_transitions():
    """The gold tags must themselves be legal, or step 5.8 measures the
    converter rather than the model."""
    stats = illegal_transition_stats([["B-DRUG", "I-DRUG", "O", "B-EFFECT"]])

    assert stats["illegal_transitions"] == 0


# --------------------------------------------------------------------------
# token accuracy - reported, deprecated
# --------------------------------------------------------------------------

def test_token_accuracy_is_high_for_an_all_O_tagger():
    """The concrete reason this metric is in the table only to be argued with."""
    gold = [["O"] * 9 + ["B-DRUG"]]
    pred = [["O"] * 10]

    assert token_accuracy(gold, pred) == pytest.approx(0.9)
    assert entity_metrics(gold, pred)["entity_f1_strict"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# tokens and tags must be filtered together
# --------------------------------------------------------------------------

def test_encode_tokens_with_tags_filters_in_lockstep():
    _, index = load_vocab()
    tokens = ["patient", ",", "developed", "rash"]
    tags = ["O", "O", "O", "B-EFFECT"]

    ids, kept = encode_tokens_with_tags(tokens, tags, index)

    assert len(ids) == len(kept) == 3
    assert kept == ["O", "O", "B-EFFECT"]


def test_encode_tokens_with_tags_rejects_mismatched_input():
    _, index = load_vocab()

    with pytest.raises(ValueError, match="tokens but"):
        encode_tokens_with_tags(["a", "b"], ["O"], index)


def test_filtering_preserves_entities_on_the_real_corpus():
    """Verified across all Stage 2 splits: dropping punctuation never changes
    the decoded entity count. If that stops holding, the tagger is being trained
    on a different set of entities than the report claims."""
    pd = pytest.importorskip("pandas")
    import json as _json
    from pathlib import Path

    from src.bio_convert import to_bio

    _, index = load_vocab()
    splits = Path(__file__).resolve().parents[1] / "data" / "splits"
    before = after = 0

    for split in ("dev", "test"):
        df = pd.read_parquet(splits / f"stage2_{split}.parquet")
        for text, spans in zip(df.text, df.spans):
            raw = _json.loads(spans) if isinstance(spans, str) else spans
            parsed = [(int(s), int(e), str(label)) for s, e, label in raw]

            tokens, tags = to_bio(text, parsed, strict=False)
            before += len(bio_to_entities(tokens, tags))

            _, kept = encode_tokens_with_tags(tokens, tags, index)
            after += len(bio_to_entities([""] * len(kept), kept))

    assert before == after, f"{before} entities before filtering, {after} after"
