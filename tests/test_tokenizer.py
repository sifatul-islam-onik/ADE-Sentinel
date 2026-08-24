"""Tests for the domain tokenizer.

The offset tests matter most. Phase 5 turns character spans into BIO tags by
intersecting them with these offsets, and an off-by-one there produces training
labels that are wrong but not obviously wrong -- the loss curve looks fine and
the entity-F1 is quietly garbage. PLAN F5.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tokenizer import (  # noqa: E402
    PROTECTED_ABBREVIATIONS, naive_tokenize, sentence_split, smart_lower, tokenize,
)


# --------------------------------------------------------------------------
# Offsets - the property Phase 5 depends on
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Hepatotoxicity developed after 3 weeks of methotrexate.",
    "5-fluorouracil 20 mg/kg was given (P<0.05).",
    "TNF-alpha rose; the patient's ALL relapsed.",
    "Rash, fever and thrombocytopenia were noted.",
])
def test_offsets_recover_original_substring(text):
    """Every offset pair must slice the exact token back out of the source."""
    tokens, offsets = tokenize(text, lower=False)
    for token, (start, end) in zip(tokens, offsets):
        assert text[start:end] == token


def test_offsets_are_sorted_and_non_overlapping():
    text = "5-fluorouracil 20 mg/kg was given (P<0.05)."
    _, offsets = tokenize(text)
    for (_, prev_end), (start, _) in zip(offsets, offsets[1:]):
        assert start >= prev_end


def test_offsets_are_stable_under_lowercasing():
    """Lowercasing must not shift offsets - they index the original string."""
    text = "The ALL patient received 5-Fluorouracil."
    _, raw_offsets = tokenize(text, lower=False)
    _, low_offsets = tokenize(text, lower=True)
    assert raw_offsets == low_offsets


# --------------------------------------------------------------------------
# The four things naive tokenisation destroys
# --------------------------------------------------------------------------

def test_hyphenated_chemical_names_stay_intact():
    tokens, _ = tokenize("5-fluorouracil and TNF-alpha were measured")
    assert "5-fluorouracil" in tokens
    assert "tnf-alpha" in tokens
    # the naive baseline shatters both
    naive = naive_tokenize("5-fluorouracil and TNF-alpha were measured")
    assert "5-fluorouracil" not in naive
    assert "5" in naive and "fluorouracil" in naive


def test_dosage_is_one_token():
    tokens, _ = tokenize("received 20 mg/kg daily")
    assert "20 mg/kg" in tokens


@pytest.mark.parametrize("dose", ["5mg", "1.5 g/day", "300 IU", "10 mL", "2.5 mg/m2"])
def test_dosage_variants(dose):
    tokens, _ = tokenize(f"a dose of {dose} was given")
    assert any(t.replace(" ", "") == dose.replace(" ", "").lower() for t in tokens), tokens


@pytest.mark.parametrize("stat", ["P<0.05", "p = 0.001", "P<.01"])
def test_statistics_are_one_token(stat):
    tokens, _ = tokenize(f"significant ({stat})")
    joined = [t.replace(" ", "") for t in tokens]
    assert stat.replace(" ", "").lower() in joined, tokens


def test_greek_letters_survive():
    tokens, _ = tokenize("TNF-α and IL-6 were elevated")
    assert "tnf-α" in tokens


# --------------------------------------------------------------------------
# Case handling - the ALL problem
# --------------------------------------------------------------------------

def test_ALL_is_not_lowercased_into_a_stopword():
    """The headline case: ALL is acute lymphoblastic leukemia, not `all`."""
    tokens, _ = tokenize("The patient with ALL received therapy")
    assert "ALL" in tokens
    assert "all" not in tokens
    # and the naive baseline gets it wrong
    assert "all" in naive_tokenize("The patient with ALL received therapy")


@pytest.mark.parametrize("abbr", ["ALL", "MS", "RA", "PT", "US", "IV", "CT"])
def test_protected_abbreviations_survive(abbr):
    assert smart_lower(abbr) == abbr


def test_ordinary_capitalised_words_are_lowercased():
    assert smart_lower("The") == "the"
    assert smart_lower("Methotrexate") == "methotrexate"


def test_long_acronyms_are_lowercased():
    """The rule is length <= 5; a longer all-caps run is not protected."""
    assert smart_lower("ABCDEFG") == "abcdefg"


def test_unlisted_short_acronym_is_lowercased():
    """Being all-caps and short is not enough - it must be in the list."""
    assert "ZZQ" not in PROTECTED_ABBREVIATIONS
    assert smart_lower("ZZQ") == "zzq"


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------

def test_sentence_split_basic():
    text = "Hepatotoxicity developed. The drug was withdrawn. Recovery followed."
    assert len(sentence_split(text)) == 3


@pytest.mark.parametrize("text", [
    "The drug was given i.v. to the patient without incident.",
    "Dosing was b.i.d. for two weeks in every case.",
    "See Fig. 2 for the full time course of the reaction.",
    "Reported by Smith et al. in a similar cohort of patients.",
])
def test_sentence_split_does_not_break_on_domain_abbreviations(text):
    assert len(sentence_split(text)) == 1, sentence_split(text)


@pytest.mark.parametrize("text", [
    "The drug was given i.v. Two hours later the rash appeared.",
    "Dosing was b.i.d. Recovery followed within a week.",
    "See Fig. 2 for the time course of the reaction.",
    "Reported by Smith et al. Similar cases have been described.",
    "Cultures grew E. coli after 48 hours of incubation.",
])
def test_abbreviation_guards_hold_before_capitals_and_digits(text):
    """The guards are only actually exercised when the next token is capitalised
    or numeric; a lowercase continuation is blocked by the lookahead anyway, so
    these cases are what prove the lookbehinds are not inert."""
    parts = sentence_split(text)
    assert len(parts) == 1, parts


def test_sentence_split_handles_empty():
    assert sentence_split("") == []
    assert sentence_split("   ") == []


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "...", "123", "α"])
def test_degenerate_inputs_do_not_raise(text):
    tokens, offsets = tokenize(text)
    assert len(tokens) == len(offsets)


def test_no_token_is_whitespace_only():
    text = "  spaced   out \n text with 20 mg/kg  "
    tokens, _ = tokenize(text)
    assert all(t.strip() for t in tokens)
