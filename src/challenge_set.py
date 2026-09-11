"""Step 6.3 - the negation and hedging challenge subset, selected by rule.

PRD 8.3 asks for ~50-80 hand-picked test sentences containing negation or
hedging. PLAN F6 replaces hand-picking with a rule, because hand-picking after
the models exist is cherry-picking however well it is meant. The subset is
whatever the cue lists below match in the Stage 1 test split - nothing is added
or removed by eye - and `scripts/negation_eval.py` freezes it to
`data/splits/stage1_test_cues.parquet` the first time it is built, so a later
edit to a cue cannot quietly change what was evaluated.

**When the lists were written, stated plainly.** They were first written in
`scripts/document_stage1.py` during the Phase 4 documentation, after the Stage 1
models and their test predictions existed, and after four sentences that every
Stage 1 model misclassifies had been read. They were then used to test whether
those universally-missed sentences are enriched for either cue (report section
7.7: they are not), and they have not been changed since; they moved here
verbatim so the Stage 1 report and run 13 share one definition, and
`tests/test_challenge_set.py` pins them. The selection itself reads sentence
text only - never a prediction, never a label.

**What a match means.** These are broad surface patterns. `no` matches "no
evidence of" and also "No. 3"; `may` matches the modal and the month. A match
says a sentence *contains* a negation or hedge word, not that the word scopes
over the drug-effect relation, so the subset is a population in which scope
reasoning is more often needed - not one in which it always is.
"""

from __future__ import annotations

import re

NEGATION_CUES = (r"\bno\b", r"\bnot\b", r"\bnone\b", r"\bneither\b", r"\bnor\b",
                 r"\bwithout\b", r"\bdenied\b", r"\babsent\b", r"\bfailed to\b",
                 r"\bruled out\b", r"\bnegative for\b", r"\bunremarkable\b")
HEDGING_CUES = (r"\bmay\b", r"\bmight\b", r"\bpossibl", r"\bpotential",
                r"\bsuggest", r"\blikely\b", r"\bunclear\b", r"\bappears?\b",
                r"\bcould\b", r"\bprobable\b", r"\bcaution\b", r"\bunlikely\b")

_NEGATION = re.compile("|".join(NEGATION_CUES), re.IGNORECASE)
_HEDGING = re.compile("|".join(HEDGING_CUES), re.IGNORECASE)


def cue_matches(text: str) -> dict[str, list[str]]:
    """The negation and hedging cues a sentence contains, as matched surface strings."""
    return {
        "negation": sorted({m.group(0).lower() for m in _NEGATION.finditer(text)}),
        "hedging": sorted({m.group(0).lower() for m in _HEDGING.finditer(text)}),
    }


def select(texts, labels):
    """One row per sentence with its cue flags, in input order.

    Every sentence is kept, flagged or not, so the frozen file also defines the
    no-cue comparison group and cannot drift out of alignment with the split.
    """
    import pandas as pd

    rows = []
    for text, label in zip(texts, labels):
        found = cue_matches(text)
        rows.append({
            "text": text,
            "label": int(label),
            "negation": bool(found["negation"]),
            "hedging": bool(found["hedging"]),
            "negation_cues": "|".join(found["negation"]),
            "hedging_cues": "|".join(found["hedging"]),
        })
    return pd.DataFrame(rows)
