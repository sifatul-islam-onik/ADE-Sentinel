"""Word and sentence splitting for biomedical text.

A normal tokenizer damages exactly the words this project cares about:

    5-fluorouracil  ->  "5" + "fluorouracil"     (drug name destroyed)
    TNF-alpha       ->  "TNF" + "alpha"
    20 mg/kg        ->  four meaningless pieces
    P<0.05          ->  "P" + "<" + "0.05"
    ALL             ->  "all"                    (a leukemia becomes a stopword)

So the project uses its own regex tokenizer. It is deliberately simple and
deterministic: you can read it, and it runs fast over 160k abstracts.

`tokenize` returns character offsets as well as words. Stage 2 needs them, both
to turn the corpus's character spans into per-word tags and to paint the
predicted entities back onto the original sentence in the demo.
"""

from __future__ import annotations

import re
import unicodedata

# All-caps medical abbreviations that must survive lowercasing. Without this
# list, ALL (leukemia), MS, PT and US all turn into ordinary English stopwords
# and the thing they name disappears from the corpus.
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

_UNIT = r"(?:mg|mcg|µg|ug|ng|g|kg|mL|ml|L|l|mmol|mol|IU|U|meq|mEq)"
_PER = r"(?:kg|m2|m\^2|day|d|hr?|h|wk|week|dose|mL|ml|L|l|min)"

# Order matters: the first alternative that matches wins, so the multi-character
# domain patterns have to come before the generic word pattern.
_PATTERNS = [
    ("stat", rf"[Pp]\s*[<>=≤≥]{{1,2}}\s*\d*\.?\d+"),          # P<0.05, p = 0.001
    ("dose", rf"\d+(?:\.\d+)?\s*{_UNIT}(?:\s*/\s*{_PER})?\b"),  # 20 mg/kg, 5mg
    ("word", r"\w+(?:[-'’]\w+)*"),                             # 5-fluorouracil
    ("num", r"\d+(?:[.,]\d+)*"),                               # 0.05, 1,200
    ("punct", r"[^\s\w]"),
]

_MASTER = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _PATTERNS), re.UNICODE)

# The naive baseline the domain tokenizer is compared against in notebook 02.
_NAIVE = re.compile(r"[A-Za-z0-9]+")


def smart_lower(token: str) -> str:
    """Lowercase, except for protected all-caps medical abbreviations.

    ALL -> ALL, The -> the, TNF-alpha -> tnf-alpha (hyphenated forms are not
    protected; the abbreviation is still recoverable from context).
    """
    if token.isupper() and len(token) <= 5 and token in PROTECTED_ABBREVIATIONS:
        return token
    return token.lower()


def tokenize(text: str, lower: bool = True) -> tuple[list[str], list[tuple[int, int]]]:
    """Split `text` into words and their (start, end) character offsets.

    Offsets index into the ORIGINAL string, before any lowercasing, so they stay
    valid for span alignment whatever `lower` is set to.
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
    """What a default `re.findall(r'\\w+')` pipeline does - the "before" column."""
    toks = _NAIVE.findall(text)
    return [t.lower() for t in toks] if lower else toks


def normalize_unicode(text: str) -> str:
    """NFKC-normalise. Folds ligatures and full-width forms, leaves Greek alone."""
    return unicodedata.normalize("NFKC", text)


def sentence_split(text: str) -> list[str]:
    """Split an abstract into sentences.

    A regex, not a general sentence splitter, because biomedical abstracts are
    full of periods that are not sentence boundaries: `i.v.`, `b.i.d.`, `Fig. 2`,
    `et al.`, `vs.`, and species names like `E. coli`. Splitting on those would
    produce fragments that pollute the embedding training windows.
    """
    # Each lookbehind has to include the trailing period, because the split
    # position sits AFTER the ".".
    protected = (
        r"(?<!\b[A-Za-z]\.)"                          # initials: "E. coli"
        r"(?<!\bi\.v\.)(?<!\bi\.m\.)(?<!\bp\.o\.)"    # routes of administration
        r"(?<!\bb\.i\.d\.)(?<!\bt\.i\.d\.)(?<!\bq\.d\.)"
        r"(?<!\bet\sal\.)(?<!\bvs\.)(?<!\bFig\.)(?<!\bapprox\.)"
        r"(?<!\bNo\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bMrs\.)"
        r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bcf\.)"
    )
    parts = re.split(rf"{protected}(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]
