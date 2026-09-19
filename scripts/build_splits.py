"""Steps 1.1-1.3 - deduplicate both configs and build ONE global split.

Three things here matter more than the code, and each is a PLAN finding:

F3 - Dedup means different things for the two configs.
     The classification config is one row per sentence, so duplicates are true
     duplicates and dropping them is right. The relation config is one row per
     drug-effect RELATION, so a sentence with four pairs appears four times with
     different spans. Measured: 6,821 rows collapse to 4,271 sentences, so a
     naive drop_duplicates would discard 2,550 pairs - 37.4% of all supervision -
     and permanently cap Stage 2 recall. Here it is a GROUP BY plus a span UNION.

F4 - Both stages must share one split.
     Measured cross-config overlap is 4,271, i.e. every relation sentence is also
     a classification row. Splitting each config independently would put a
     sentence in Stage-1 train and Stage-2 test, and run 12's error-propagation
     number - the report's most distinctive result - would be measuring
     memorisation. So the split is computed once over the union of unique
     sentences and projected onto each config.

F7 - The output is small (~3 MB) and is COMMITTED to git, which is what makes
     "never re-split at runtime" enforceable rather than aspirational.

Usage:  .venv\\Scripts\\python scripts\\build_splits.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import DEFAULT_SEED, set_seed  # noqa: E402

REPO = "ade-benchmark-corpus/ade_corpus_v2"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
REPORT_PATH = REPO_ROOT / "results" / "split_report.md"

TRAIN_FRAC, DEV_FRAC = 0.70, 0.15   # test takes the remainder


# --------------------------------------------------------------------------
# 1.2  relation config -> one row per sentence, spans unioned
# --------------------------------------------------------------------------

def extract_spans(row: dict) -> list[tuple[int, int, str]]:
    """Pull (start, end, label) triples out of one relation row.

    `indexes` stores parallel start_char/end_char LISTS, not scalars: an entity
    mentioned twice in the sentence carries both offsets. Measured distribution
    of list lengths - drug: 6,235 rows with 1, 534 with 2, 39 with 3, 4 with 4,
    and 9 with ZERO; effect: 6,645 with 1, 142 with 2, 6 with 3, 28 with zero.

    The zero-length case is why this returns a list rather than a pair: those
    rows name an entity but give no offsets, so there is nothing to tag. They are
    counted and reported rather than silently yielding empty labels, which is the
    failure mode PLAN F5 exists to prevent.
    """
    spans: list[tuple[int, int, str]] = []
    for field, label in (("drug", "DRUG"), ("effect", "EFFECT")):
        idx = row["indexes"][field]
        for start, end in zip(idx["start_char"], idx["end_char"]):
            spans.append((int(start), int(end), label))
    return spans


def group_relations(rel) -> tuple[dict[str, list[tuple[int, int, str]]], dict]:
    by_text: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
    rows_without_spans = 0
    total_span_slots = 0
    # Per-FIELD, not just per-row: a row whose drug has no offsets but whose
    # effect does still yields spans, so an all-empty-row counter reports zero
    # while entity mentions are quietly going unlabelled.
    missing_offsets = Counter()

    for row in rel:
        for field in ("drug", "effect"):
            if not row["indexes"][field]["start_char"]:
                missing_offsets[field] += 1
        spans = extract_spans(row)
        total_span_slots += len(spans)
        if not spans:
            rows_without_spans += 1
        # A set: the 4-relation sentence "naproxen/oxaprozin x tense bullae/
        # cutaneous fragility" is 4 rows but only 4 distinct spans, not 8.
        by_text[row["text"]].update(spans)

    grouped = {t: sorted(s) for t, s in by_text.items()}
    stats = {
        "relation_rows": len(rel),
        "unique_sentences": len(grouped),
        "rows_discarded_by_naive_dedup": len(rel) - len(grouped),
        "rows_with_no_offsets": rows_without_spans,
        "drug_mentions_missing_offsets": missing_offsets["drug"],
        "effect_mentions_missing_offsets": missing_offsets["effect"],
        "span_slots_before_dedup": total_span_slots,
        "unique_spans_after_union": sum(len(v) for v in grouped.values()),
    }
    return grouped, stats


# --------------------------------------------------------------------------
# 1.3  one global split over the union, stratified on the Stage 1 label
# --------------------------------------------------------------------------

def assign_splits(texts_by_label: dict[int, list[str]], rng) -> dict[str, str]:
    """Stratified 70/15/15 by label, assigning each unique sentence exactly once."""
    assignment: dict[str, str] = {}
    for label, texts in sorted(texts_by_label.items()):
        texts = sorted(texts)          # sort first so the shuffle is seed-reproducible
        rng.shuffle(texts)
        n = len(texts)
        n_train = int(round(TRAIN_FRAC * n))
        n_dev = int(round(DEV_FRAC * n))
        for i, text in enumerate(texts):
            if i < n_train:
                assignment[text] = "train"
            elif i < n_train + n_dev:
                assignment[text] = "dev"
            else:
                assignment[text] = "test"
    return assignment


def main() -> int:
    import pandas as pd
    from datasets import load_dataset

    seed = set_seed(DEFAULT_SEED)
    import random
    rng = random.Random(seed)

    print(f"seed = {seed}\n")

    cls = load_dataset(REPO, "Ade_corpus_v2_classification")["train"]
    rel = load_dataset(REPO, "Ade_corpus_v2_drug_ade_relation")["train"]

    # ---- 1.1 classification: true duplicates, safe to drop ----------------
    label_by_text: dict[str, int] = {}
    conflicts = 0
    for text, label in zip(cls["text"], cls["label"]):
        if text in label_by_text and label_by_text[text] != label:
            conflicts += 1
        label_by_text.setdefault(text, label)

    if conflicts:
        sys.exit(f"{conflicts} sentences carry conflicting labels; dedup is ill-defined.")

    print(f"1.1 classification : {len(cls):,} rows -> {len(label_by_text):,} unique "
          f"({len(cls) - len(label_by_text):,} duplicates dropped, {conflicts} conflicts)")

    # ---- 1.2 relations: GROUP BY + span UNION (F3) ------------------------
    grouped, rel_stats = group_relations(rel)
    print(f"1.2 relations      : {rel_stats['relation_rows']:,} rows -> "
          f"{rel_stats['unique_sentences']:,} sentences, "
          f"{rel_stats['unique_spans_after_union']:,} unique spans")
    print(f"    naive dedup would have discarded "
          f"{rel_stats['rows_discarded_by_naive_dedup']:,} relation rows")
    missing = (rel_stats["drug_mentions_missing_offsets"]
               + rel_stats["effect_mentions_missing_offsets"])
    if missing:
        print(f"    [!] {missing} entity mentions name an entity but carry no char "
              f"offsets (drug {rel_stats['drug_mentions_missing_offsets']}, "
              f"effect {rel_stats['effect_mentions_missing_offsets']}) - no BIO supervision")

    # ---- 1.3 one split over the union (F4) --------------------------------
    relation_only = set(grouped) - set(label_by_text)
    if relation_only:
        # Measured as empty, but assert rather than assume: if it ever changes,
        # those sentences have no Stage 1 label to stratify on.
        sys.exit(f"{len(relation_only)} relation sentences absent from the "
                 "classification config; they have no label to stratify on.")

    texts_by_label: dict[int, list[str]] = defaultdict(list)
    for text, label in label_by_text.items():
        texts_by_label[label].append(text)

    assignment = assign_splits(texts_by_label, rng)
    print(f"\n1.3 global split   : {len(assignment):,} unique sentences assigned")

    # ---- write stage 1 ----------------------------------------------------
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    stage1 = pd.DataFrame(
        {"text": t, "label": label_by_text[t], "split": assignment[t]}
        for t in sorted(label_by_text)
    )
    for split in ("train", "dev", "test"):
        part = stage1[stage1.split == split].drop(columns="split").reset_index(drop=True)
        part.to_parquet(SPLITS_DIR / f"stage1_{split}.parquet", index=False)

    # ---- write stage 2 ----------------------------------------------------
    stage2 = pd.DataFrame(
        {
            "text": t,
            "spans": json.dumps(grouped[t]),   # parquet-safe; list[tuple] is not
            "n_spans": len(grouped[t]),
            "split": assignment[t],
        }
        for t in sorted(grouped)
    )
    for split in ("train", "dev", "test"):
        part = stage2[stage2.split == split].drop(columns="split").reset_index(drop=True)
        part.to_parquet(SPLITS_DIR / f"stage2_{split}.parquet", index=False)

    # ---- assertions: the whole point of doing it this way ------------------
    print("\nassertions")
    problems = []

    seen: dict[str, str] = {}
    for frame, stage in ((stage1, "stage1"), (stage2, "stage2")):
        for text, split in zip(frame.text, frame.split):
            if text in seen and seen[text] != split:
                problems.append(f"{text[:60]!r} in both {seen[text]} and {split}")
            seen[text] = split
    print(f"  [{'ok' if not problems else '!!'}] no sentence in two splits, "
          f"across BOTH configs ({len(seen):,} checked)")

    overall = len(stage1[stage1.label == 1]) / len(stage1)
    for split in ("train", "dev", "test"):
        part = stage1[stage1.split == split]
        ratio = len(part[part.label == 1]) / len(part)
        drift = abs(ratio - overall) * 100
        flag = "ok" if drift <= 1.0 else "!!"
        if drift > 1.0:
            problems.append(f"{split} positive ratio drifts {drift:.2f}pp")
        print(f"  [{flag}] stage1 {split:5s} {len(part):>6,} rows, "
              f"{ratio * 100:5.2f}% positive (drift {drift:.2f}pp)")

    for split in ("train", "dev", "test"):
        part = stage2[stage2.split == split]
        print(f"  [ok] stage2 {split:5s} {len(part):>6,} sentences, "
              f"{part.n_spans.sum():>6,} spans")

    # every stage2 sentence must be a stage1 positive
    s1pos = set(stage1[stage1.label == 1].text)
    stray = set(stage2.text) - s1pos
    if stray:
        problems.append(f"{len(stray)} stage2 sentences are not stage1 positives")
    print(f"  [{'ok' if not stray else '!!'}] every stage2 sentence is a stage1 positive")

    # ---- report ------------------------------------------------------------
    lines = [
        "# Split protocol - measured",
        "",
        f"Produced by `scripts/build_splits.py`, seed **{seed}**, "
        f"{int(TRAIN_FRAC*100)}/{int(DEV_FRAC*100)}/"
        f"{100-int(TRAIN_FRAC*100)-int(DEV_FRAC*100)} stratified on the Stage 1 label.",
        "",
        "## Deduplication",
        "",
        "| Config | Rows | Unique sentences | Note |",
        "|---|---|---|---|",
        f"| classification | {len(cls):,} | {len(label_by_text):,} | true duplicates, dropped |",
        f"| drug-ade relation | {rel_stats['relation_rows']:,} | "
        f"{rel_stats['unique_sentences']:,} | one row per *relation*: grouped, spans unioned |",
        "",
        f"Naive `drop_duplicates` on the relation config would have discarded "
        f"**{rel_stats['rows_discarded_by_naive_dedup']:,}** drug-effect pairs "
        f"({100*rel_stats['rows_discarded_by_naive_dedup']/rel_stats['relation_rows']:.1f}% "
        "of all Stage 2 supervision). Grouping preserves them: "
        f"{rel_stats['unique_spans_after_union']:,} unique spans across "
        f"{rel_stats['unique_sentences']:,} sentences.",
        "",
        f"{rel_stats['drug_mentions_missing_offsets']} drug mentions and "
        f"{rel_stats['effect_mentions_missing_offsets']} effect mentions name an entity "
        "but carry no character offsets, so they contribute no BIO supervision. They are "
        "reported here rather than silently producing all-`O` sequences (PLAN F5). "
        f"{rel_stats['rows_with_no_offsets']} rows lack offsets for *both* fields.",
        "",
        "## One split, shared by both stages",
        "",
        "Cross-config overlap is total: every relation sentence is also a classification "
        "row. Independent splits would leak Stage-1 training sentences into the Stage-2 "
        "test set and invalidate the run-12 error-propagation measurement, so the "
        "assignment is computed once over the union and projected onto each config.",
        "",
        "| Split | Stage 1 rows | % positive | Stage 2 sentences | Stage 2 spans |",
        "|---|---|---|---|---|",
    ]
    for split in ("train", "dev", "test"):
        p1 = stage1[stage1.split == split]
        p2 = stage2[stage2.split == split]
        lines.append(
            f"| {split} | {len(p1):,} | {100*len(p1[p1.label==1])/len(p1):.2f}% | "
            f"{len(p2):,} | {p2.n_spans.sum():,} |"
        )
    lines += [
        "",
        "## Assertions",
        "",
        "- No sentence appears in more than one split, checked across **both** configs.",
        "- Per-split positive ratio stays within 1pp of the corpus ratio.",
        "- Every Stage 2 sentence is a Stage 1 positive.",
        "",
        "These files are committed to git deliberately: they are small, and committing "
        "them is what makes \"never re-split at runtime\" enforceable.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    (INTERIM_DIR / "relation_stats.json").write_text(
        json.dumps(rel_stats, indent=2), encoding="utf-8")

    print(f"\nwrote {SPLITS_DIR.relative_to(REPO_ROOT)}/stage{{1,2}}_{{train,dev,test}}.parquet")
    print(f"wrote {REPORT_PATH.relative_to(REPO_ROOT)}")

    if problems:
        print("\nFAILED:")
        for p in problems:
            print("  -", p)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
