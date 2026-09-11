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


# --------------------------------------------------------------------------
# the native scorer - the remote path, where seqeval cannot be installed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tags,expected", [
    (["B-DRUG", "I-DRUG"], [(0, 2, "DRUG")]),
    (["O", "I-DRUG", "I-DRUG"], []),                 # orphan: no entity at all
    (["B-DRUG", "I-EFFECT"], [(0, 1, "DRUG")]),      # type change closes, orphan dropped
    (["B-DRUG", "B-DRUG"], [(0, 1, "DRUG"), (1, 2, "DRUG")]),
    (["O", "B-EFFECT", "O"], [(1, 2, "EFFECT")]),
    ([], []),
])
def test_strict_entities(tags, expected):
    from src.stage2_metrics import strict_entities

    assert strict_entities(tags) == expected


def test_strict_and_lenient_differ_exactly_on_orphans():
    """`bio_to_entities` repairs an orphan `I-X`; `strict_entities` does not.
    That difference is the whole strict/lenient distinction."""
    from src.stage2_metrics import strict_entities

    tags = ["O", "I-DRUG", "I-DRUG"]

    assert bio_to_entities([""] * 3, tags) == [(1, 3, "DRUG")]
    assert strict_entities(tags) == []


def test_prf_counts_duplicate_entities_once_each():
    """Two identical predictions must not both match one gold entity."""
    from src.stage2_metrics import prf

    p, r, f1 = prf([[(0, 1, "DRUG")]], [[(0, 1, "DRUG"), (0, 1, "DRUG")]])

    assert r == pytest.approx(1.0)
    assert p == pytest.approx(0.5)


def test_native_scorer_needs_no_seqeval(monkeypatch):
    """The remote runner has no seqeval at all (its sdist will not build on
    Python 3.12), so this path must not import it even indirectly."""
    import builtins

    from src.stage2_metrics import native_entity_scores

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("seqeval"):
            raise ImportError("seqeval is unavailable on the remote runner")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    scores = native_entity_scores(
        [["B-DRUG", "I-DRUG", "O", "B-EFFECT"]],
        [["B-DRUG", "I-DRUG", "O", "O"]],
    )

    assert scores["entity_f1_strict"] == pytest.approx(2 / 3)


def test_entity_metrics_degrades_gracefully_without_seqeval(monkeypatch):
    """Without seqeval the cross-check is skipped, not fatal - and the native
    numbers are still returned."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("seqeval"):
            raise ImportError("unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    m = entity_metrics([["B-DRUG"]], [["B-DRUG"]])

    assert m["seqeval_crosscheck"] == "unavailable"
    assert m["entity_f1_strict"] == pytest.approx(1.0)


def test_seqeval_crosscheck_agrees_locally():
    """Locally seqeval IS installed, so the two implementations are compared on
    every call. This is the assertion that makes the native scorer trustworthy
    on the remote runner where seqeval cannot check it."""
    pytest.importorskip("seqeval")

    gold = [["B-DRUG", "I-DRUG", "O", "B-EFFECT", "I-EFFECT"],
            ["O", "B-EFFECT", "O"], ["B-DRUG", "B-DRUG", "O"]]
    pred = [["B-DRUG", "O", "O", "I-EFFECT", "I-EFFECT"],
            ["O", "B-EFFECT", "O"], ["B-DRUG", "I-DRUG", "O"]]

    m = entity_metrics(gold, pred)

    assert m["seqeval_crosscheck"] == "agrees"
    assert m["seqeval_f1_strict"] == pytest.approx(m["entity_f1_strict"])
    assert m["seqeval_f1_lenient"] == pytest.approx(m["entity_f1_lenient"])


def test_native_matches_seqeval_on_the_real_corpus():
    """The strongest form of the check: agreement over every Stage 2 sentence,
    including deliberately malformed predictions."""
    pytest.importorskip("seqeval")
    pd = pytest.importorskip("pandas")

    import json as _json
    import random
    from pathlib import Path

    from seqeval.metrics import f1_score
    from seqeval.scheme import IOB2

    from src.bio_convert import to_bio
    from src.models.encoding import load_vocab
    from src.stage2_metrics import native_entity_scores

    _, index = load_vocab()
    splits = Path(__file__).resolve().parents[1] / "data" / "splits"
    gold = []
    for text, spans in zip(*(lambda d: (d.text, d.spans))(
            pd.read_parquet(splits / "stage2_test.parquet"))):
        raw = _json.loads(spans) if isinstance(spans, str) else spans
        tokens, tags = to_bio(
            text, [(int(s), int(e), str(label)) for s, e, label in raw], strict=False)
        _, kept = encode_tokens_with_tags(tokens, tags, index)
        if kept:
            gold.append(kept)

    rng = random.Random(11)
    pred = []
    for sequence in gold:
        row = list(sequence)
        for i, tag in enumerate(row):
            roll = rng.random()
            if roll < 0.12:
                row[i] = "O" if tag != "O" else "B-DRUG"
            elif tag.startswith("B-") and roll < 0.30:
                row[i] = "I-" + tag[2:]          # inject illegal transitions
        pred.append(row)

    native = native_entity_scores(gold, pred)

    assert native["entity_f1_strict"] == pytest.approx(
        float(f1_score(gold, pred, mode="strict", scheme=IOB2, zero_division=0)))
    assert native["entity_f1_lenient"] == pytest.approx(
        float(f1_score(gold, pred, zero_division=0)))


# --------------------------------------------------------------------------
# what lenient scoring adds, and where the entity errors are
# --------------------------------------------------------------------------

def _legal(tags):
    """Repair orphan `I-X` into `B-X`, giving a well-formed gold sequence."""
    fixed, previous = [], "O"
    for tag in tags:
        if tag.startswith("I-") and previous[2:] != tag[2:]:
            tag = "B-" + tag[2:]
        fixed.append(tag)
        previous = tag
    return fixed


def _random_corpus(seed, sentences=300, longest=12):
    """Well-formed gold against uniformly random - heavily malformed - output."""
    import random

    rng = random.Random(seed)
    gold, pred = [], []
    for _ in range(sentences):
        n = rng.randint(1, longest)
        gold.append(_legal([rng.choice(TAGS) for _ in range(n)]))
        pred.append([rng.choice(TAGS) for _ in range(n)])
    return gold, pred


def test_repaired_entities_are_exactly_the_illegal_transitions():
    """Strict entities are always a subset of lenient ones, and the difference
    is one entity per illegal transition. Checked on random sequences rather
    than on hand-picked cases, because the report's explanation of the
    strict/lenient gap rests on it."""
    from src.stage2_metrics import repaired_entities, strict_entities

    gold, pred = _random_corpus(seed=7)
    for p in pred:
        strict = set(strict_entities(p))
        lenient = set(bio_to_entities([""] * len(p), p))
        assert strict <= lenient
        assert len(lenient - strict) == count_illegal_transitions(p)

    stats = repaired_entities(gold, pred)
    assert stats["repaired_entities"] == sum(count_illegal_transitions(p) for p in pred)


def test_lenient_beats_strict_exactly_when_repairs_are_accurate_enough():
    """lenient F1 > strict F1  iff  c / k > strict F1 / 2, for k repaired
    entities of which c are correct. The Stage 2 report uses this to explain the
    sign of the gap, so it is checked in exact arithmetic and against the scorer
    itself rather than taken on trust."""
    from fractions import Fraction

    from src.stage2_metrics import repaired_entities, strict_entities

    def f1(gold_sets, pred_sets):
        tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets))
        total = sum(map(len, gold_sets)) + sum(map(len, pred_sets))
        return Fraction(2 * tp, total) if total else Fraction(0)

    for seed in range(40):
        gold, pred = _random_corpus(seed=seed, sentences=20, longest=8)
        gold_sets = [set(strict_entities(g)) for g in gold]
        strict = f1(gold_sets, [set(strict_entities(p)) for p in pred])
        lenient = f1(gold_sets, [set(bio_to_entities([""] * len(p), p)) for p in pred])

        m = entity_metrics(gold, pred)
        assert m["entity_f1_strict"] == pytest.approx(float(strict))
        assert m["entity_f1_lenient"] == pytest.approx(float(lenient))

        r = repaired_entities(gold, pred)
        k, c = r["repaired_entities"], r["repaired_correct"]
        if k == 0:
            assert lenient == strict
        else:
            assert (lenient > strict) == (Fraction(c, k) > strict / 2)


def test_error_breakdown_categories():
    from src.stage2_metrics import error_breakdown

    gold = [["B-EFFECT", "I-EFFECT", "O", "B-DRUG", "O", "O"]]
    pred = [["B-EFFECT", "O", "O", "B-EFFECT", "O", "B-DRUG"]]

    b = error_breakdown(gold, pred)

    # predicted: a truncated EFFECT, a DRUG labelled EFFECT, an invented DRUG
    assert (b["pred_exact"], b["pred_boundary"], b["pred_type"], b["pred_spurious"]) == (0, 1, 1, 1)
    assert (b["gold_found"], b["gold_boundary"], b["gold_type"], b["gold_missed"]) == (0, 1, 1, 0)
    # overlap match credits only the truncated EFFECT: 1 of 3 predicted, 1 of 2 gold
    assert b["overlap_precision"] == pytest.approx(1 / 3)
    assert b["overlap_recall"] == pytest.approx(1 / 2)
    assert b["overlap_f1"] == pytest.approx(0.4)


def test_error_breakdown_accounts_for_every_entity():
    from src.stage2_metrics import error_breakdown, strict_entities

    gold, pred = _random_corpus(seed=3)
    b = error_breakdown(gold, pred)

    assert (b["pred_exact"] + b["pred_boundary"] + b["pred_type"] + b["pred_spurious"]
            == sum(len(strict_entities(p)) for p in pred))
    assert (b["gold_found"] + b["gold_boundary"] + b["gold_type"] + b["gold_missed"]
            == sum(len(strict_entities(g)) for g in gold))
    assert b["pred_exact"] == b["gold_found"]
    # overlap match only ever adds hits, so it can never score below exact match
    assert b["overlap_f1"] >= entity_metrics(gold, pred)["entity_f1_strict"] - 1e-12
