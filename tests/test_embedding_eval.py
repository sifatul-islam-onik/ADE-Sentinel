"""Tests for the coverage / neighbour evidence functions (steps 3.4-3.5).

These produce report figures, so a silent error here becomes a wrong number in
the headline section. The token-vs-type distinction in particular is easy to
compute backwards and impossible to spot afterwards.
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedding_eval import (  # noqa: E402
    PROBE_TERMS, coverage, coverage_table, neighbour_table, neighbours,
    oov_terms, task_token_counts,
)


class FakeKV:
    """Minimal KeyedVectors stand-in: membership plus most_similar."""

    def __init__(self, vocab, sims=None):
        self._vocab = set(vocab)
        self._sims = sims or {}

    def __contains__(self, key):
        return key in self._vocab

    def most_similar(self, term, topn=5):
        return self._sims.get(term, [])[:topn]


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------

def test_token_and_type_coverage_differ_as_expected():
    """The whole point of reporting both: frequent words covered, rare not.

    `the` appears 100 times and is known; three domain terms appear once each
    and are not. Token coverage is high, type coverage is low.
    """
    counts = Counter({"the": 100, "hepatotoxicity": 1, "doxorubicin": 1, "rhabdomyolysis": 1})
    kv = FakeKV({"the"})
    c = coverage(counts, kv)

    assert c["type_coverage"] == pytest.approx(0.25)
    assert c["token_coverage"] == pytest.approx(100 / 103)
    assert c["token_coverage"] > c["type_coverage"]


def test_full_coverage():
    counts = Counter({"a": 2, "b": 3})
    c = coverage(counts, FakeKV({"a", "b"}))
    assert c["type_coverage"] == 1.0 and c["token_coverage"] == 1.0
    assert c["types_known"] == 2 and c["tokens_known"] == 5


def test_zero_coverage():
    c = coverage(Counter({"a": 2}), FakeKV(set()))
    assert c["type_coverage"] == 0.0 and c["token_coverage"] == 0.0


def test_empty_counts_do_not_divide_by_zero():
    c = coverage(Counter(), FakeKV({"a"}))
    assert c["type_coverage"] == 0.0 and c["token_coverage"] == 0.0


# --------------------------------------------------------------------------
# OOV listing
# --------------------------------------------------------------------------

def test_oov_terms_are_frequency_ordered():
    counts = Counter({"known": 50, "rare_oov": 1, "common_oov": 20, "mid_oov": 5})
    got = oov_terms(counts, FakeKV({"known"}))
    assert [t for t, _ in got] == ["common_oov", "mid_oov", "rare_oov"]


def test_oov_terms_respects_limit():
    counts = Counter({f"w{i}": 100 - i for i in range(50)})
    assert len(oov_terms(counts, FakeKV(set()), limit=10)) == 10


def test_oov_terms_empty_when_all_known():
    counts = Counter({"a": 1, "b": 2})
    assert oov_terms(counts, FakeKV({"a", "b"})) == []


# --------------------------------------------------------------------------
# Neighbours
# --------------------------------------------------------------------------

def test_neighbours_returns_none_for_missing_term():
    """None means 'no entry', which is a finding, not a failure."""
    assert neighbours(FakeKV({"aspirin"}), "doxorubicin") is None


def test_neighbours_returns_pairs():
    kv = FakeKV({"doxorubicin"}, {"doxorubicin": [("cisplatin", 0.9), ("etoposide", 0.8)]})
    got = neighbours(kv, "doxorubicin", topn=2)
    assert got == [("cisplatin", 0.9), ("etoposide", 0.8)]


def test_neighbours_respects_topn():
    kv = FakeKV({"x"}, {"x": [("a", 0.9), ("b", 0.8), ("c", 0.7)]})
    assert len(neighbours(kv, "x", topn=2)) == 2


# --------------------------------------------------------------------------
# Table rendering
# --------------------------------------------------------------------------

def test_neighbour_table_marks_missing_vocabulary():
    models = {
        "GloVe": FakeKV(set()),
        "Ours": FakeKV({"hepatotoxicity"},
                       {"hepatotoxicity": [("nephrotoxicity", 0.9)]}),
    }
    table = neighbour_table(models, probes=["hepatotoxicity"])
    assert "not in vocabulary" in table
    assert "nephrotoxicity" in table
    assert "| Probe | GloVe | Ours |" in table


def test_coverage_table_has_a_row_per_model():
    counts = Counter({"a": 1, "b": 1})
    table = coverage_table({"M1": FakeKV({"a"}), "M2": FakeKV({"a", "b"})}, counts)
    assert "M1" in table and "M2" in table
    assert "50.00%" in table and "100.00%" in table


def test_probe_list_covers_the_required_families():
    assert len(PROBE_TERMS) >= 15
    for required in ("doxorubicin", "hepatotoxicity", "thrombocytopenia",
                     "rash", "renal", "induced", "withdrawal"):
        assert required in PROBE_TERMS


# --------------------------------------------------------------------------
# Task token counting
# --------------------------------------------------------------------------

def test_task_token_counts_uses_the_given_tokenizer_and_drops_punctuation():
    from src.tokenizer import tokenize

    counts = task_token_counts(["Rash, fever.", "Rash again!"], tokenize)
    assert counts["rash"] == 2
    assert "," not in counts and "." not in counts


# --------------------------------------------------------------------------
# Case fallback - fairness of the GloVe baseline
# --------------------------------------------------------------------------

def test_case_fallback_finds_uncased_vocabulary():
    """Our tokenizer protects `HIV`; glove.6B is uncased and holds `hiv`.

    Without the retry, our own preprocessing would be scored as a GloVe failure.
    """
    from src.embedding_eval import lookup_key
    uncased = FakeKV({"hiv", "ct", "aml"})
    assert lookup_key("HIV", uncased) == "hiv"
    assert lookup_key("HIV", uncased, case_fallback=False) is None


def test_case_fallback_prefers_exact_match():
    from src.embedding_eval import lookup_key
    kv = FakeKV({"ALL", "all"})
    assert lookup_key("ALL", kv) == "ALL"


def test_case_fallback_changes_measured_coverage():
    counts = Counter({"HIV": 10, "methotrexate": 5})
    uncased = FakeKV({"hiv", "methotrexate"})
    assert coverage(counts, uncased, case_fallback=True)["type_coverage"] == 1.0
    assert coverage(counts, uncased, case_fallback=False)["type_coverage"] == pytest.approx(0.5)


def test_oov_respects_case_fallback():
    counts = Counter({"HIV": 10})
    uncased = FakeKV({"hiv"})
    assert oov_terms(counts, uncased) == []
    assert oov_terms(counts, uncased, case_fallback=False) == [("HIV", 10)]


def test_neighbours_use_the_case_folded_key():
    kv = FakeKV({"hiv"}, {"hiv": [("aids", 0.9)]})
    assert neighbours(kv, "HIV") == [("aids", 0.9)]
    assert neighbours(kv, "HIV", case_fallback=False) is None


def test_coverage_table_reports_both_case_columns():
    counts = Counter({"HIV": 1, "rash": 1})
    table = coverage_table({"GloVe": FakeKV({"hiv", "rash"})}, counts)
    assert "exact case" in table
    assert "100.00%" in table and "50.00%" in table
