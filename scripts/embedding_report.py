"""Steps 3.4-3.5 - generate the coverage and nearest-neighbour report figures.

Loads whichever embeddings are present in `models/` and writes
`results/figures/coverage.md` and `results/figures/neighbours.md`. Missing
models are skipped with a note, so this is useful before FastText exists as well
as after.

Vectors are memory-mapped (`mmap='r'`). That is not a micro-optimisation: on a
machine whose Windows commit limit is nearly exhausted, loading GloVe's
400,000 x 300 float32 array outright raises MemoryError even with GBs of
physical RAM free. Mapping the file sidesteps the commit charge entirely.

Usage:  .venv\\Scripts\\python scripts\\embedding_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embedding_eval import (  # noqa: E402
    PROBE_TERMS, coverage, coverage_table, neighbour_table, oov_terms, task_token_counts,
)
from src.tokenizer import tokenize  # noqa: E402

MODELS = REPO_ROOT / "models"
SPLITS = REPO_ROOT / "data" / "splits"
FIGURES = REPO_ROOT / "results" / "figures"

# Display name -> filename. Order is the order of the report columns.
CANDIDATES = [
    ("E1 GloVe (general)", "glove.kv"),
    ("E2 Word2Vec (ours)", "w2v.kv"),
    ("E3 FastText (ours)", "ft.kv"),
]


def main() -> int:
    import pandas as pd
    from gensim.models import KeyedVectors

    texts = pd.concat([
        pd.read_parquet(SPLITS / f"stage1_{s}.parquet") for s in ("train", "dev", "test")
    ]).text.tolist()
    counts = task_token_counts(texts, tokenize)

    models: dict[str, object] = {}
    missing: list[str] = []
    for name, filename in CANDIDATES:
        path = MODELS / filename
        if path.exists():
            models[name] = KeyedVectors.load(str(path), mmap="r")
        else:
            missing.append(name)

    if not models:
        sys.exit("No embeddings in models/ - train them first.")

    print(f"{len(texts):,} sentences | {sum(counts.values()):,} tokens | "
          f"{len(counts):,} types")
    for name, kv in models.items():
        print(f"  loaded {name:24s} {len(kv):>8,} vectors")
    if missing:
        print(f"  missing: {', '.join(missing)}")

    FIGURES.mkdir(parents=True, exist_ok=True)

    # ---- coverage (3.4) ---------------------------------------------------
    glove_name = "E1 GloVe (general)"
    lines = [
        "# Vocabulary coverage (PRD 7.2)",
        "",
        f"Measured over the deduplicated ADE corpus: {len(texts):,} sentences, "
        f"{sum(counts.values()):,} tokens, {len(counts):,} distinct types, using the "
        "project's own tokenizer.",
        "",
        coverage_table(models, counts),
        "",
        "## Reading this table",
        "",
        "**Token coverage is high for every embedding; type coverage is not.** That gap "
        "is the finding. General-purpose vectors cover the common English scaffolding - "
        "`the`, `patient`, `was`, `after` - which is most of the running text, so token "
        "coverage flatters them. Type coverage counts each distinct word once, and it is "
        "the distinct words that carry the domain: a drug name appearing three times is "
        "as informative as one appearing three hundred.",
        "",
        "The `exact case` column exists for fairness, not decoration. Our tokenizer "
        "protects medical abbreviations from lowercasing (`ALL`, `HIV`, `CT`), while "
        "`glove.6B` is an uncased release holding `hiv` and `ct`. Scoring those as GloVe "
        "misses would penalise the baseline for **our** preprocessing choice, so every "
        "lookup retries case-folded. The headline columns use that retry; the last column "
        "shows what a naive exact-match comparison would have reported instead.",
        "",
    ]

    if glove_name in models:
        strict = coverage(counts, models[glove_name], case_fallback=False)
        fair = coverage(counts, models[glove_name], case_fallback=True)
        delta = (fair["type_coverage"] - strict["type_coverage"]) * 100
        lines += [
            f"For GloVe that retry is worth {delta:+.2f} percentage points "
            f"({strict['type_coverage'] * 100:.2f}% -> "
            f"{fair['type_coverage'] * 100:.2f}%). Small, but now measured rather than "
            "assumed - and the domain advantage survives it, which is the point.",
            "",
            "## Most frequent task terms GloVe has never seen",
            "",
            "Case-folded, so these are genuinely absent rather than artefacts of casing.",
            "",
            "| Term | Frequency in the ADE corpus |",
            "|---|---|",
        ]
        for term, freq in oov_terms(counts, models[glove_name], limit=30):
            lines.append(f"| `{term}` | {freq:,} |")
        lines += [
            "",
            "Two kinds of miss appear here. Most are **domain vocabulary** - "
            "`intravitreal`, `endophthalmitis`, `rechallenge`, `capecitabine`, "
            "`l-asparaginase` - words a general-purpose corpus simply never contains. "
            "A few are **possessives** (`patient's`, `crohn's`, `parkinson's`), where "
            "GloVe's tokenisation split the clitic and ours does not. The second group "
            "is a tokenisation difference rather than a vocabulary gap, and is called out "
            "here so the table is not over-read.",
            "",
        ]

    (FIGURES / "coverage.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- neighbours (3.5) -------------------------------------------------
    nn = [
        "# Nearest neighbours (PRD 7.3)",
        "",
        "Top-5 cosine neighbours for a fixed probe list, side by side. Probes span four "
        "families so the table shows more than one kind of failure: drug names, toxicity "
        "morphology, everyday words used in a clinical sense, and the causation cues "
        "Stage 1 depends on.",
        "",
        neighbour_table(models, PROBE_TERMS, topn=5),
        "",
        "**not in vocabulary** is a result, not a gap in the experiment: it is the "
        "strongest possible statement about coverage, and PRD 7.3 asks for at least one "
        "such probe.",
        "",
    ]
    if missing:
        nn += [f"*Pending: {', '.join(missing)}. Rerun once trained.*", ""]

    (FIGURES / "neighbours.md").write_text("\n".join(nn), encoding="utf-8")

    print(f"\nwrote {(FIGURES / 'coverage.md').relative_to(REPO_ROOT)}")
    print(f"wrote {(FIGURES / 'neighbours.md').relative_to(REPO_ROOT)}")
    if missing:
        print(f"\n[note] rerun after training: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
