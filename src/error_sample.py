"""Step 6.4 - the manual half of the error taxonomy, kept across re-runs.

`scripts/error_taxonomy.py` classifies every pipeline failure by rule and
writes a seeded sample of 30 to `results/error_sample.csv` for reading. The
rules say what a sentence *contains*; the `manual_category` and `notes` columns
say what a reader found *caused* the error. The script regenerates the CSV on
every run, so this module is what stops a re-run from erasing that reading.

**The codebook is frozen.** `CATEGORIES` was fixed before the sample was read
and `tests/test_error_sample.py` pins it: adding a category after reading is
how a taxonomy quietly fits itself to its data. A cell holds one category, or
two joined by "; " with the primary first. "gold questionable" is always the
primary when it applies, because it puts in doubt whether the row is an error
at all.

**A reading is never dropped silently.** Carry-over is by test-set index. If a
row someone has filled in is no longer in the sample - the seed, the failure
set or the test split moved - `carry_over` raises rather than let the reading
vanish, and the same happens if the sentence at that index has changed.
"""

from __future__ import annotations

import math

CATEGORIES = ("negation", "hedging", "multi-drug", "abbreviation", "boundary",
              "generic effect", "unrelated mention", "indirect statement",
              "gold questionable", "other")
BOUNDARY_SUBTYPES = ("modifier:", "coordination:", "nested:", "tokenisation:")
SEPARATOR = "; "
LOW_CONFIDENCE = "[low confidence]"
Z95 = 1.959963984540054


def validate(value: str) -> tuple[str, ...]:
    """The categories in a `manual_category` cell, primary first; ValueError if invalid."""
    parts = tuple(value.split(SEPARATOR))
    unknown = [p for p in parts if p not in CATEGORIES]
    if unknown:
        raise ValueError(f"unknown category {unknown[0]!r} in {value!r} - "
                         f"use one of {', '.join(CATEGORIES)}")
    if len(parts) > 2:
        raise ValueError(f"{value!r} has {len(parts)} categories - at most two, primary first")
    if len(parts) == 2 and parts[0] == parts[1]:
        raise ValueError(f"{value!r} repeats a category")
    if len(parts) == 2 and parts[1] == "gold questionable":
        raise ValueError(f"{value!r}: 'gold questionable' must be the primary category")
    return parts


def is_low_confidence(notes: str) -> bool:
    return notes.lstrip().startswith(LOW_CONFIDENCE)


def carry_over(existing_rows, sample_indices, texts=None) -> dict[int, tuple[str, str]]:
    """{test_index: (manual_category, notes)} for existing rows still in the sample.

    `existing_rows` are dicts as read by csv.DictReader. A row with nothing in
    either manual column may leave the sample freely; a filled one may not. If
    `texts` maps test index to sentence, a filled row whose sentence differs is
    refused too, since its reading would attach to a different sentence.
    """
    sample = set(sample_indices)
    carried: dict[int, tuple[str, str]] = {}
    for row in existing_rows:
        index = int(row["test_index"])
        category = row.get("manual_category") or ""
        notes = row.get("notes") or ""
        filled = bool(category.strip() or notes.strip())
        if index in carried:
            raise ValueError(f"test sentence {index} appears twice in the existing sample")
        if index not in sample:
            if filled:
                raise ValueError(f"test sentence {index} has a manual reading but is no longer "
                                 "sampled - move or clear it by hand before re-running")
            continue
        if filled and texts is not None and row.get("text") != texts[index]:
            raise ValueError(f"test sentence {index} has a manual reading but its text no "
                             "longer matches the test split")
        carried[index] = (category, notes)
    return carried


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (95% by default)."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    # The bounds at k = 0 and k = n are exactly 0 and 1; floating point lands a hair off.
    lower = 0.0 if k == 0 else max(0.0, centre - half)
    upper = 1.0 if k == n else min(1.0, centre + half)
    return lower, upper
