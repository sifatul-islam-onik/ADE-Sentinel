"""Step 0.3 - the day-1 sanity script from PRD section 6.1.

Runs the PRD's script verbatim (the prints below are unchanged), then writes the
numbers to results/dataset_stats.md so the report quotes measured figures rather
than the PRD's approximations.

It also measures three quantities the PRD's version prints but never acts on,
each of which is load-bearing later:

  * duplicate counts per config  -> the dedup protocol (PLAN F3)
  * cross-config sentence overlap -> why Stage 1 and Stage 2 must share ONE
    split, or run 12 evaluates a Stage 1 model on sentences it trained on
    (PLAN F4)
  * relations per sentence        -> the ceiling that naive row-dedup would
    silently impose on Stage 2 recall (PLAN F3)

Usage:  .venv\\Scripts\\python scripts\\sanity_check.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUT_PATH = REPO_ROOT / "results" / "dataset_stats.md"
REPO = "ade-benchmark-corpus/ade_corpus_v2"


def main() -> int:
    from datasets import load_dataset

    # ---- PRD section 6.1 script, verbatim ----------------------------------
    cls = load_dataset(REPO, "Ade_corpus_v2_classification")["train"]
    rel = load_dataset(REPO, "Ade_corpus_v2_drug_ade_relation")["train"]

    print("classification rows:", len(cls), "| unique:", len(set(cls["text"])))
    print("label distribution:", Counter(cls["label"]))
    print("relation rows:", len(rel), "| unique:", len(set(rel["text"])))
    print("overlap between configs:", len(set(cls["text"]) & set(rel["text"])))
    print(cls[0])
    print(rel[0])
    # ---- end verbatim block ------------------------------------------------

    cls_texts, rel_texts = list(cls["text"]), list(rel["text"])
    cls_unique, rel_unique = set(cls_texts), set(rel_texts)
    labels = Counter(cls["label"])
    overlap = cls_unique & rel_unique
    union_unique = cls_unique | rel_unique

    # How many relations does a single sentence carry? This is exactly what a
    # naive drop-duplicates on the relation config would throw away.
    rel_per_sentence = Counter(rel_texts)
    multi = Counter(v for v in rel_per_sentence.values())
    spans_lost_by_naive_dedup = len(rel_texts) - len(rel_unique)

    pos = labels.get(1, 0)
    total = sum(labels.values())
    pos_pct = 100.0 * pos / total if total else 0.0

    # Class balance AFTER dedup is the number Stage 1 actually trains against,
    # and it differs from the raw balance because the duplication is not spread
    # evenly across classes. Compute it here rather than discovering it in Phase 4.
    first_label: dict[str, int] = {}
    for text, label in zip(cls_texts, cls["label"]):
        first_label.setdefault(text, label)
    dedup_labels = Counter(first_label.values())
    dedup_total = sum(dedup_labels.values())
    dedup_pos_pct = 100.0 * dedup_labels.get(1, 0) / dedup_total if dedup_total else 0.0

    # A sentence carrying two different labels would make dedup ill-defined:
    # "keep the first row" would then silently pick a label. Check it in one pass.
    labels_by_text: dict[str, set] = {}
    for text, label in zip(cls_texts, cls["label"]):
        labels_by_text.setdefault(text, set()).add(label)
    conflicting = sum(1 for labs in labels_by_text.values() if len(labs) > 1)

    lines = [
        "# Dataset statistics - measured, not estimated",
        "",
        f"Source: `{REPO}` (Hugging Face Hub, parquet-native).",
        "Produced by `scripts/sanity_check.py` (PLAN step 0.3).",
        "**Quote these numbers in the report, not the PRD's approximations.**",
        "",
        "## Row counts",
        "",
        "| Config | Rows | Unique sentences | Duplicate rows |",
        "|---|---|---|---|",
        f"| `Ade_corpus_v2_classification` | {len(cls_texts):,} | {len(cls_unique):,} | "
        f"{len(cls_texts) - len(cls_unique):,} |",
        f"| `Ade_corpus_v2_drug_ade_relation` | {len(rel_texts):,} | {len(rel_unique):,} | "
        f"{spans_lost_by_naive_dedup:,} |",
        "",
        "## Class balance (classification config)",
        "",
        "| Label | Rows | Share |",
        "|---|---|---|",
    ]
    for label, count in sorted(labels.items()):
        name = "1 (ADE)" if label == 1 else "0 (not ADE)"
        lines.append(f"| {name} | {count:,} | {100.0 * count / total:.2f}% |")
    lines += [
        "",
        "### After deduplication - the balance Stage 1 actually trains on",
        "",
        "| Label | Unique sentences | Share |",
        "|---|---|---|",
    ]
    for label, count in sorted(dedup_labels.items()):
        name = "1 (ADE)" if label == 1 else "0 (not ADE)"
        lines.append(f"| {name} | {count:,} | {100.0 * count / dedup_total:.2f}% |")
    lines += [
        "",
        f"Raw positive share is {pos_pct:.2f}%, but after dedup it is "
        f"**{dedup_pos_pct:.2f}%**. The duplication is concentrated in the positive "
        "class, so dedup is not label-neutral - it shifts the balance materially. "
        f"The PRD's \"roughly 1 in 5\" describes the *post-dedup* corpus, which is the "
        "correct one to design against.",
        "",
        "Either way, **macro-F1 and per-class F1 are the primary metrics**; raw accuracy "
        "is misleading at this imbalance (PRD 6.1 fact 3).",
        "",
        f"Sentences carrying conflicting labels across duplicate rows: **{conflicting}**. "
        + ("Dedup is well-defined - any duplicate row can be kept."
           if conflicting == 0 else
           "**Resolve these explicitly before dedup**; keeping the first row would "
           "silently pick a label."),
        "",
        "## Cross-config overlap (PLAN F4)",
        "",
        f"- Sentences in both configs: **{len(overlap):,}**",
        f"- Unique sentences across the union of both configs: **{len(union_unique):,}**",
        "",
        "Because the configs share sentences, splitting them independently would put a "
        "sentence in Stage-1 *train* and Stage-2 *test*. The split must be built once "
        "over the union and projected onto each config, or the run-12 error-propagation "
        "number is invalid.",
        "",
        "## Relations per sentence (PLAN F3)",
        "",
        "| Relations in one sentence | Number of sentences |",
        "|---|---|",
    ]
    for n, count in sorted(multi.items()):
        lines.append(f"| {n} | {count:,} |")
    lines += [
        "",
        f"Naive `drop_duplicates` on the relation config would discard "
        f"**{spans_lost_by_naive_dedup:,} drug-effect pairs** "
        f"({100.0 * spans_lost_by_naive_dedup / len(rel_texts):.1f}% of all relations), "
        "permanently capping Stage 2 recall. Dedup for Stage 2 must GROUP BY sentence "
        "and UNION the spans.",
        "",
        "## First row of each config",
        "",
        "```python",
        f"cls[0] = {cls[0]!r}",
        "",
        f"rel[0] = {rel[0]!r}",
        "```",
        "",
        "## Reference check",
        "",
        "Published NER benchmark statistics for the positive portion: 4,272 sentences, "
        "86,865 tokens, 12,264 ADE tags, 5,544 Drug tags. Use these to sanity-check "
        "Stage 2 numbers against the literature (PRD 6.1 fact 4).",
        "",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"[ok] wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"[!!] naive relation dedup would discard {spans_lost_by_naive_dedup:,} "
          f"drug-effect pairs - see PLAN F3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
