"""Step 5.3 - run BIO conversion over the whole Stage 2 corpus and eyeball it.

The PRD is explicit that 20 converted examples must be inspected by hand before
any tagger is trained. This produces those examples, and just as importantly the
corpus-wide diagnostics: how many spans needed boundary snapping, how many
aligned to no token at all, how many entities came out the far side.

The number that matters: entities tagged must equal spans in, or supervision has
gone missing between the split files and the tagger.

Usage:  .venv\\Scripts\\python scripts\\bio_spotcheck.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.bio_convert import ConversionStats, count_illegal_transitions, to_bio  # noqa: E402

SPLITS = REPO_ROOT / "data" / "splits"
OUT = REPO_ROOT / "results" / "bio_spotcheck.txt"
REPORT = REPO_ROOT / "results" / "bio_conversion_report.md"

N_EXAMPLES = 20


def main() -> int:
    import pandas as pd

    frames = {s: pd.read_parquet(SPLITS / f"stage2_{s}.parquet")
              for s in ("train", "dev", "test")}

    stats = ConversionStats()
    tag_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    illegal_total = 0
    examples: list[str] = []

    for split, frame in frames.items():
        for row in frame.itertuples():
            spans = [tuple(s) for s in json.loads(row.spans)]
            # strict=False: this is the survey pass, so a miss should be counted
            # across the whole corpus rather than aborting on the first one.
            tokens, tags = to_bio(row.text, spans, stats=stats, strict=False)

            tag_counts.update(tags)
            for tag in tags:
                if tag.startswith("B-"):
                    entity_counts[tag[2:]] += 1
            illegal_total += count_illegal_transitions(tags)

            if split == "train" and len(examples) < N_EXAMPLES:
                width = max((len(t) for t in tokens), default=1)
                block = [f"[{split}] {row.text}", ""]
                block += [f"    {t:<{width}}  {g}" for t, g in zip(tokens, tags)]
                examples.append("\n".join(block))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "BIO conversion spot check - inspect these before training any tagger.\n"
        "Step 5.3 of PLAN.md.\n\n" + ("\n\n" + "=" * 72 + "\n\n").join(examples) + "\n",
        encoding="utf-8")

    total_spans = sum(len(json.loads(r.spans)) for f in frames.values()
                      for r in f.itertuples())

    lines = [
        "# BIO conversion - corpus-wide diagnostics",
        "",
        "Produced by `scripts/bio_spotcheck.py` (PLAN step 5.3). These numbers exist",
        "because the PRD's reference converter fails silently in two ways, and",
        "\"probably zero\" is not a measurement (PLAN F5).",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Sentences converted | {stats.sentences:,} |",
        f"| Spans in | {total_spans:,} |",
        f"| Entities tagged | {stats.entities_tagged:,} |",
        f"| Spans needing boundary snapping | {stats.spans_snapped:,} |",
        f"| Spans matching no token | {stats.spans_unmatched:,} |",
        f"| Nested spans dropped (flat BIO cannot hold them) | {stats.spans_dropped_nested:,} |",
        f"| Illegal transitions in gold tags | {illegal_total:,} |",
        "",
        "## Entity counts",
        "",
        "| Label | Entities |",
        "|---|---|",
    ]
    for label, count in sorted(entity_counts.items()):
        lines.append(f"| {label} | {count:,} |")

    lines += [
        "",
        "## Tag distribution",
        "",
        "| Tag | Tokens | Share |",
        "|---|---|---|",
    ]
    total_tags = sum(tag_counts.values())
    for tag, count in sorted(tag_counts.items()):
        lines.append(f"| `{tag}` | {count:,} | {100 * count / total_tags:.2f}% |")

    lines += [
        "",
        f"`O` accounts for {100 * tag_counts['O'] / total_tags:.1f}% of tokens, which is "
        "exactly why token accuracy is not reported for Stage 2: a model predicting `O` "
        "everywhere would score that high while finding nothing. Entity-level "
        "precision/recall/F1 via `seqeval` is the metric.",
        "",
        "## Interpretation",
        "",
    ]

    if stats.spans_unmatched:
        lines += [
            f"**{stats.spans_unmatched} spans matched no token.** These contribute no "
            "supervision. Examples:",
            "",
        ] + [f"- `{e}`" for e in stats.examples_unmatched] + [""]
    else:
        lines += ["No span failed to align - every annotation reached the tagger.", ""]

    if stats.spans_dropped_nested:
        lines += [
            f"**{stats.spans_dropped_nested} spans were dropped as nested.** The ADE "
            "corpus annotates drug names inside effect phrases - `theophylline` inside "
            "`theophylline intoxication`, `lead` inside `high blood lead level`. Flat "
            "BIO gives each token exactly one tag, so the two cannot coexist; letting "
            "the inner span overwrite the outer one orphans the remainder into an "
            "illegal `I-EFFECT`. The longer span wins, since it is the more complete "
            "annotation of the adverse event. Dropped by label: "
            + ", ".join(f"{k} {v}" for k, v in sorted(stats.dropped_by_label.items()))
            + ".",
            "",
            "The general fix is one BIO plane per entity type - all observed nesting is "
            "DRUG-inside-EFFECT - which needs a two-head tagger and is outside the "
            "PRD's model ladder. The loss is therefore measured and reported rather "
            "than hidden, and it caps Stage 2 recall by this amount.",
            "",
        ] + [f"- `{e}`" for e in stats.examples_dropped] + [""]

    if stats.spans_snapped:
        lines += [
            f"**{stats.spans_snapped} spans did not fall on token boundaries** and were "
            "widened to the enclosing tokens. The PRD's containment test would have "
            "dropped these entirely. Examples (annotation -> tagged tokens):",
            "",
        ] + [f"- `{e}`" for e in stats.examples_snapped] + [""]

    lines += [
        "Gold tag sequences contain "
        f"{'no' if illegal_total == 0 else str(illegal_total)} illegal transitions, "
        "as they must by construction. That zero is the baseline the CRF ablation in "
        "step 5.8 measures model output against.",
        "",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"sentences         : {stats.sentences:,}")
    print(f"spans in          : {total_spans:,}")
    print(f"entities tagged   : {stats.entities_tagged:,}")
    print(f"snapped           : {stats.spans_snapped:,}")
    print(f"unmatched         : {stats.spans_unmatched:,}")
    print(f"nested dropped    : {stats.spans_dropped_nested:,} {dict(stats.dropped_by_label)}")
    print(f"illegal (gold)    : {illegal_total:,}")
    print(f"O share           : {100 * tag_counts['O'] / total_tags:.1f}%")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)} ({N_EXAMPLES} examples)")
    print(f"wrote {REPORT.relative_to(REPO_ROOT)}")

    # Every input span must be accounted for as one of: tagged, unmatched, or
    # deliberately dropped as nested. Anything else means supervision vanished
    # somewhere between the split files and the tagger.
    accounted = (stats.entities_tagged + stats.spans_unmatched
                 + stats.spans_dropped_nested)
    if accounted != total_spans:
        print(f"\n[!!] {accounted:,} spans accounted for vs {total_spans:,} in "
              "- supervision lost without explanation")
        return 1

    if illegal_total:
        print(f"\n[!!] {illegal_total} illegal transitions in GOLD tags; the CRF "
              "ablation in step 5.8 assumes this is zero")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
