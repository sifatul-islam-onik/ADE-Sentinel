"""Tests for step 6.4's manual columns - the frozen codebook and their carry-over."""

from __future__ import annotations

import pytest

from src import error_sample


def row(index, category="", notes="", text=None):
    return {"test_index": str(index), "manual_category": category, "notes": notes,
            "text": text if text is not None else f"sentence {index}"}


def test_codebook_is_frozen():
    """Adding a category after reading lets the taxonomy fit itself to the sample,
    so an edit here must fail before it can change a reported count."""
    assert error_sample.CATEGORIES == (
        "negation", "hedging", "multi-drug", "abbreviation", "boundary",
        "generic effect", "unrelated mention", "indirect statement",
        "gold questionable", "other")
    assert error_sample.BOUNDARY_SUBTYPES == ("modifier:", "coordination:", "nested:",
                                             "tokenisation:")


def test_carry_over_by_index():
    existing = [row(94, "boundary", "modifier: x"), row(52, "negation", ""),
                row(7, "", "a note only")]

    carried = error_sample.carry_over(existing, [7, 52, 94, 120])

    assert carried == {94: ("boundary", "modifier: x"), 52: ("negation", ""),
                       7: ("", "a note only")}


@pytest.mark.parametrize("category,notes", [("boundary", ""), ("", "read, not yet categorised")])
def test_carry_over_refuses_to_drop_a_reading(category, notes):
    existing = [row(52, "negation", "kept"), row(94, category, notes)]

    with pytest.raises(ValueError, match="94"):
        error_sample.carry_over(existing, [52])


def test_blank_rows_leave_the_sample_freely():
    existing = [row(52, "negation", "kept"), row(94), row(119, "  ", " ")]

    assert error_sample.carry_over(existing, [52]) == {52: ("negation", "kept")}


def test_carry_over_refuses_a_changed_sentence():
    existing = [row(52, "negation", "kept", text="old wording")]

    with pytest.raises(ValueError, match="text"):
        error_sample.carry_over(existing, [52], texts={52: "new wording"})
    assert error_sample.carry_over(existing, [52], texts={52: "old wording"}) == \
        {52: ("negation", "kept")}


def test_carry_over_refuses_a_duplicate_index():
    with pytest.raises(ValueError, match="twice"):
        error_sample.carry_over([row(52, "negation"), row(52)], [52])


@pytest.mark.parametrize("value,expected", [
    ("boundary", ("boundary",)),
    ("generic effect", ("generic effect",)),
    ("gold questionable; negation", ("gold questionable", "negation")),
    ("boundary; unrelated mention", ("boundary", "unrelated mention")),
])
def test_validate_accepts_one_or_two_categories(value, expected):
    assert error_sample.validate(value) == expected


@pytest.mark.parametrize("value", [
    "",
    "Boundary",                               # case matters
    "spelling",                               # not in the codebook
    "boundary; spelling",
    "boundary;hedging",                       # separator is "; "
    "boundary; hedging; negation",            # three
    "boundary; boundary",
    "negation; gold questionable",            # gold questionable is always primary
])
def test_validate_rejects(value):
    with pytest.raises(ValueError):
        error_sample.validate(value)


def test_low_confidence_marker():
    assert error_sample.is_low_confidence("[low confidence] modifier: x")
    assert not error_sample.is_low_confidence("modifier: x [low confidence]")


@pytest.mark.parametrize("k,lower,upper", [
    (0, 0.0, 0.113513),
    (30, 0.886487, 1.0),
    (15, 0.331541, 0.668459),
])
def test_wilson(k, lower, upper):
    lo, hi = error_sample.wilson(k, 30)

    assert lo == pytest.approx(lower, abs=1e-6)
    assert hi == pytest.approx(upper, abs=1e-6)
    assert 0.0 <= lo <= k / 30 <= hi <= 1.0


@pytest.mark.parametrize("k,n", [(1, 0), (-1, 30), (31, 30)])
def test_wilson_rejects_impossible_counts(k, n):
    with pytest.raises(ValueError):
        error_sample.wilson(k, n)
