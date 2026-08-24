"""Step 2.1 - domain tokenizer for biomedical text.

Returns tokens AND their character offsets. The offsets are not a convenience:
Phase 5 converts the relation config's character spans into BIO tags, and that
conversion is only as correct as the offsets it is given (PLAN F5).

Four things standard whitespace/regex tokenisation destroys, all of which carry
the domain signal this project depends on:

  1. Hyphenated chemical names. `5-fluorouracil` split on the hyphen becomes
     `5` + `fluorouracil`, and `TNF-alpha` becomes `TNF` + `alpha`. Both lose the
     identity of the entity.
  2. Dosages. `20 mg/kg` split on whitespace and punctuation becomes four
     meaningless pieces.
  3. Parenthesised statistics. `P<0.05` becomes `P`, `<`, `0.05`.
  4. Case. Lowercasing everything turns `ALL` (acute lymphoblastic leukemia)
     into the stopword `all` -- the single most destructive default in this
     domain, because it silently converts a disease into noise.

The tokenizer is deliberately regex-based rather than model-based: it must be
deterministic, inspectable, and fast over ~150k abstracts, and the before/after
table in `results/figures/tokenizer_table.md` has to be explainable in a viva.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------
# Abbreviations that must survive lowercasing
# --------------------------------------------------------------------------
# The rule from the PRD: protect all-caps tokens of length <= 5 that appear in a
# medical abbreviation list. The list matters more than it looks -- `ALL`, `MS`,
# `PT` and `US` are all real English words when lowercased, so without this they
# become stopwords and the disease/measure they name disappears from the corpus.
PROTECTED_ABBREVIATIONS: frozenset[str] = frozenset({
    # haematology / oncology
    "ALL", "AML", "CML", "CLL", "NHL", "HL", "MM", "MDS", "CR", "PR",
    # neurology / rheumatology / immunology
    "MS", "RA", "SLE", "GBS", "ALS", "MG", "JIA",
    # infectious disease
    "HIV", "AIDS", "TB", "HBV", "HCV", "CMV", "EBV", "HSV", "MRSA", "UTI", "URI",
    # cardiopulmonary
    "MI", "CHF", "CAD", "COPD", "PE", "DVT", "AF", "HTN", "BP", "HR", "LVEF",
    # gastro / renal / hepatic
    "GI", "GU", "IBD", "GERD", "AKI", "CKD", "ESRD", "NASH", "LFT",
    # labs
    "AST", "ALT", "ALP", "GGT", "BUN", "CBC", "WBC", "RBC", "INR", "PT", "PTT",
    "ESR", "CRP", "LDH", "HbA1c", "eGFR",
    # drug classes and targets
    "NSAID", "SSRI", "SNRI", "MAOI", "ACE", "ARB", "CCB", "PPI", "TNF", "IL",
    "EGFR", "VEGF", "HER2", "COX", "GABA", "MTX", "AZT", "TKI",
    # imaging / procedures / care settings
    "CT", "MRI", "PET", "ECG", "EKG", "EEG", "ICU", "NICU", "ER", "OR", "US",
    # routes and schedules
    "IV", "IM", "SC", "PO", "PRN", "BID", "TID", "QID", "QD",
    # pharmacovigilance and bodies
    "ADR", "ADE", "AE", "SAE", "FDA", "EMA", "WHO", "CNS", "PNS", "DILI",
})

# --------------------------------------------------------------------------
# Token patterns, in priority order. First alternative to match wins, so the
# multi-character domain patterns must precede the generic word pattern.
# --------------------------------------------------------------------------
_UNIT = r"(?:mg|mcg|µg|ug|ng|g|kg|mL|ml|L|l|mmol|mol|IU|U|meq|mEq)"
_PER = r"(?:kg|m2|m\^2|day|d|hr?|h|wk|week|dose|mL|ml|L|l|min)"

_PATTERNS = [
    # P<0.05, p = 0.001, P <= 0.01  -- one token, not three
    ("stat", rf"[Pp]\s*[<>=≤≥]{{1,2}}\s*\d*\.?\d+"),
    # 20 mg/kg, 5mg, 1.5 g/day, 300 IU
    ("dose", rf"\d+(?:\.\d+)?\s*{_UNIT}(?:\s*/\s*{_PER})?\b"),
    # 5-fluorouracil, TNF-alpha, anti-inflammatory, non-Hodgkin's
    # Leading digits are part of the name, so the pattern starts at \w.
    ("word", r"\w+(?:[-'’]\w+)*"),
    # 0.05, 37.2, 1,200
    ("num", r"\d+(?:[.,]\d+)*"),
    ("punct", r"[^\s\w]"),
]

_MASTER = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _PATTERNS), re.UNICODE)

# Naive baseline, for the before/after comparison in step 2.2.
_NAIVE = re.compile(r"[A-Za-z0-9]+")


def smart_lower(token: str) -> str:
    """Lowercase, except for protected all-caps medical abbreviations.

    `ALL` -> `ALL` (acute lymphoblastic leukemia, not the determiner)
    `The` -> `the`
    `TNF-alpha` -> `tnf-alpha`  (hyphenated forms are not protected: the
                                 abbreviation is still recoverable from context,
                                 and protecting them would fragment the vocab)
    """
    if token.isupper() and len(token) <= 5 and token in PROTECTED_ABBREVIATIONS:
        return token
    return token.lower()


def tokenize(text: str, lower: bool = True) -> tuple[list[str], list[tuple[int, int]]]:
    """Split `text` into tokens and their (start, end) character offsets.

    Offsets index into the ORIGINAL string, before any lowercasing, so they stay
    valid for span alignment regardless of the `lower` flag.
    """
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []

    for match in _MASTER.finditer(text):
        raw = match.group()
        if not raw.strip():
            continue
        tokens.append(smart_lower(raw) if lower else raw)
        offsets.append((match.start(), match.end()))

    return tokens, offsets


def naive_tokenize(text: str, lower: bool = True) -> list[str]:
    """Whitespace/alphanumeric baseline - the 'before' column of step 2.2.

    This is what a default `re.findall(r'\\w+')` pipeline does, and it is the
    thing the domain tokenizer is measured against.
    """
    toks = _NAIVE.findall(text)
    return [t.lower() for t in toks] if lower else toks


def normalize_unicode(text: str) -> str:
    """NFKC-normalise, but keep Greek letters as themselves.

    NFKC folds ligatures and full-width forms, which is wanted, and leaves
    alpha/beta alone, which is also wanted -- `TNF-α` and `TNF-alpha` are
    different surface forms of the same entity, and collapsing them is a decision
    for the vocabulary builder, not the tokenizer.
    """
    return unicodedata.normalize("NFKC", text)


def sentence_split(text: str) -> list[str]:
    """Split an abstract into sentences (step 2.3).

    Regex, not a general sentence splitter, because biomedical abstracts break
    the usual heuristics: `i.v.`, `b.i.d.`, `Fig. 2`, `et al.`, `vs.`, `approx.`
    and species names like `E. coli` all contain a period that is not a sentence
    boundary. Splitting on those would produce fragments that pollute the
    Word2Vec context windows.
    """
    # Each lookbehind must include the trailing period. The split position sits
    # AFTER the ".", so a guard written as `(?<!\bFig)` inspects ".ig" and never
    # fires. Written without the period these guards are inert, and the mistake
    # hides: `i.v. to`, `b.i.d. for` and `et al. in` all survive anyway because
    # the following word is lowercase and the `(?=[A-Z0-9])` lookahead already
    # blocks the split. Only a capitalised or numeric continuation -- `Fig. 2`,
    # `i.v. Two` -- exposes it.
    protected = (
        r"(?<!\b[A-Za-z]\.)"                          # initials: "E. coli", "J. Smith"
        r"(?<!\bi\.v\.)(?<!\bi\.m\.)(?<!\bp\.o\.)"    # routes of administration
        r"(?<!\bb\.i\.d\.)(?<!\bt\.i\.d\.)(?<!\bq\.d\.)"
        r"(?<!\bet\sal\.)(?<!\bvs\.)(?<!\bFig\.)(?<!\bapprox\.)"
        r"(?<!\bNo\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bMrs\.)"
        r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bcf\.)"
    )
    parts = re.split(rf"{protected}(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]
