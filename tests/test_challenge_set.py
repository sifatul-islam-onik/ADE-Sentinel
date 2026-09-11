"""Tests for step 6.3 - the rule-selected negation and hedging subset (PLAN F6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import challenge_set

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"


def test_cue_lists_are_frozen():
    """PLAN F6: the rule is fixed before the evaluation it defines. Editing a cue
    changes which sentences run 13 scores, so the edit must fail here first
    rather than silently move a reported result."""
    assert challenge_set.NEGATION_CUES == (
        r"\bno\b", r"\bnot\b", r"\bnone\b", r"\bneither\b", r"\bnor\b",
        r"\bwithout\b", r"\bdenied\b", r"\babsent\b", r"\bfailed to\b",
        r"\bruled out\b", r"\bnegative for\b", r"\bunremarkable\b")
    assert challenge_set.HEDGING_CUES == (
        r"\bmay\b", r"\bmight\b", r"\bpossibl", r"\bpotential",
        r"\bsuggest", r"\blikely\b", r"\bunclear\b", r"\bappears?\b",
        r"\bcould\b", r"\bprobable\b", r"\bcaution\b", r"\bunlikely\b")


def test_cues_contain_real_word_boundaries():
    """Shell quoting once turned `\\b` into a literal backspace byte, which made
    every cue match nothing and every rate 0.0%. Guard the escape itself."""
    for cue in challenge_set.NEGATION_CUES + challenge_set.HEDGING_CUES:
        assert "\x08" not in cue
        assert cue.startswith("\\b")


@pytest.mark.parametrize("text,negation,hedging", [
    ("No evidence of hepatotoxicity was found.", ["no"], []),
    ("Hepatotoxicity was ruled out.", ["ruled out"], []),
    ("This may be associated with rash.", [], ["may"]),
    ("The reaction was possibly drug-induced.", [], ["possibl"]),
    ("Knowledge of nothing unusual.", [], []),       # 'no' inside a word is not a cue
    ("Rash developed after penicillin.", [], []),
    ("It could not be excluded.", ["not"], ["could"]),
])
def test_cue_matches(text, negation, hedging):
    found = challenge_set.cue_matches(text)

    assert found["negation"] == negation
    assert found["hedging"] == hedging


def test_select_flags_every_sentence_in_order():
    df = challenge_set.select(["No rash.", "Rash may occur.", "Rash."], [1, 0, 1])

    assert list(df.negation) == [True, False, False]
    assert list(df.hedging) == [False, True, False]
    assert list(df.label) == [1, 0, 1]
    assert list(df.text) == ["No rash.", "Rash may occur.", "Rash."]


def test_frozen_subset_matches_the_rule():
    """The committed subset must be exactly what the cue lists select from the
    committed test split. If either moved, run 13's numbers no longer describe
    the file beside them."""
    pd = pytest.importorskip("pandas")
    frozen_path = SPLITS / "stage1_test_cues.parquet"
    if not frozen_path.exists():
        pytest.skip("subset not frozen yet - run scripts/negation_eval.py")

    test = pd.read_parquet(SPLITS / "stage1_test.parquet")
    frozen = pd.read_parquet(frozen_path)
    selected = challenge_set.select(test.text, test.label)

    assert list(frozen.text) == list(selected.text)
    assert list(frozen.negation) == list(selected.negation)
    assert list(frozen.hedging) == list(selected.hedging)
