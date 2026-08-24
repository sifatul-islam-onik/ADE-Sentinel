"""Step 3.6 - embedding matrices aligned to one shared task vocabulary.

Produces `models/emb_matrices/{E0_random,E1,E2,E3}.npy` plus `vocab.json`. These
are what runs 3-6 consume: the BiLSTM is trained four times, and the ONLY thing
that changes between runs is which of these files is loaded (PLAN F8).

Two decisions that make the ablation clean:

**One shared vocabulary index.** Every matrix has the same number of rows in the
same order, so swapping the file swaps the representation and nothing else. Row
0 is `<pad>` (always zero), row 1 is `<unk>`.

**One shared random base.** A naive implementation would draw fresh random
values for each matrix, so E1's uncovered 34% and E2's uncovered 20% would hold
*different* noise. Any measured difference would then partly reflect that noise
rather than the embeddings. Here a single seeded base is drawn once and each
representation overwrites only the rows it covers, so uncovered rows are
byte-identical across E0-E3 and the difference between runs is attributable to
the vectors alone.

Lookups use the same case-folding retry as the coverage measurement
(`src.embedding_eval.lookup_key`), or the matrices would disagree with the
numbers reported in `results/figures/coverage.md`.

Usage:  .venv\\Scripts\\python scripts\\build_embedding_matrices.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embedding_eval import build_matrix, lookup_key, task_token_counts  # noqa: E402
from src.tokenizer import tokenize  # noqa: E402

MODELS = REPO_ROOT / "models"
SPLITS = REPO_ROOT / "data" / "splits"
OUT = MODELS / "emb_matrices"
REPORT = REPO_ROOT / "results" / "figures" / "embedding_matrices.md"

DIM = 300
SEED = 42
MIN_FREQ = 2          # a token seen once cannot be learned by any of these models
PAD, UNK = "<pad>", "<unk>"

SOURCES = [
    ("E1", "glove.kv", "GloVe 300d (general purpose)"),
    ("E2", "w2v.kv", "Word2Vec skip-gram 300d (ours)"),
    ("E3", "ft.kv", "FastText skip-gram 300d (ours)"),
]


def main() -> int:
    import numpy as np
    import pandas as pd
    from gensim.models import KeyedVectors

    texts = pd.concat([
        pd.read_parquet(SPLITS / f"stage1_{s}.parquet") for s in ("train", "dev", "test")
    ]).text.tolist()
    counts = task_token_counts(texts, tokenize)

    vocab = [PAD, UNK] + sorted(t for t, c in counts.items() if c >= MIN_FREQ)
    index = {t: i for i, t in enumerate(vocab)}
    dropped = len(counts) - (len(vocab) - 2)

    print(f"{len(texts):,} sentences | {len(counts):,} types "
          f"| vocab {len(vocab):,} (min_freq={MIN_FREQ}, dropped {dropped:,} singletons)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")

    # The single shared base. Uniform(-0.25, 0.25) matches the usual embedding
    # layer default, and is drawn ONCE so uncovered rows match across matrices.
    rng = np.random.default_rng(SEED)
    base = rng.uniform(-0.25, 0.25, (len(vocab), DIM)).astype(np.float32)
    base[index[PAD]] = 0.0

    np.save(OUT / "E0_random.npy", base)
    rows = [("E0_random", "Randomly initialised (ablation floor)", 0, 0.0)]
    print(f"  E0_random  {len(vocab):>6,} rows, 0 filled  (the floor)")

    for key, filename, description in SOURCES:
        path = MODELS / filename
        if not path.exists():
            print(f"  {key}  SKIPPED - {filename} not found")
            continue

        kv = KeyedVectors.load(str(path), mmap="r")
        matrix, hits = build_matrix(index, base, kv, skip=(PAD, UNK))
        np.save(OUT / f"{key}.npy", matrix)
        pct = 100.0 * hits / len(vocab)
        rows.append((key, description, hits, pct))
        print(f"  {key}         {len(vocab):>6,} rows, {hits:,} filled ({pct:.1f}%)")
        del kv, matrix

    # ---- verify the invariants the ablation depends on --------------------
    print("\nassertions")
    problems = []
    saved = sorted(OUT.glob("*.npy"))
    shapes = {p.stem: np.load(p, mmap_mode="r").shape for p in saved}

    if len(set(shapes.values())) != 1:
        problems.append(f"matrices differ in shape: {shapes}")
    print(f"  [{'ok' if len(set(shapes.values())) == 1 else '!!'}] "
          f"all {len(shapes)} matrices are {next(iter(shapes.values()))}")

    for name, path in ((p.stem, p) for p in saved):
        m = np.load(path, mmap_mode="r")
        if not np.allclose(m[index[PAD]], 0.0):
            problems.append(f"{name}: <pad> row is not zero")
    print(f"  [{'ok' if not problems else '!!'}] <pad> row is zero in every matrix")

    # Uncovered rows must be identical across matrices - the point of the shared base.
    e0 = np.load(OUT / "E0_random.npy", mmap_mode="r")
    if (OUT / "E2.npy").exists():
        e2 = np.load(OUT / "E2.npy", mmap_mode="r")
        kv = KeyedVectors.load(str(MODELS / "w2v.kv"), mmap="r")
        uncovered = [i for t, i in index.items()
                     if t not in (PAD, UNK) and lookup_key(t, kv) is None]
        if uncovered:
            same = np.allclose(e0[uncovered], e2[uncovered])
            if not same:
                problems.append("uncovered rows differ between E0 and E2")
            print(f"  [{'ok' if same else '!!'}] {len(uncovered):,} uncovered rows are "
                  "identical to E0 (shared base holds)")
        del kv

    # ---- report -----------------------------------------------------------
    lines = [
        "# Embedding matrices (step 3.6)",
        "",
        f"Built by `scripts/build_embedding_matrices.py`, seed {SEED}. These are the "
        "files runs 3-6 load; the BiLSTM code and every hyperparameter stay fixed, and "
        "only the matrix changes.",
        "",
        f"Shared vocabulary: **{len(vocab):,} rows** ({DIM}d), built from tokens "
        f"appearing at least {MIN_FREQ} times in the ADE corpus. Row 0 is `<pad>` "
        f"(zero), row 1 is `<unk>`. {dropped:,} singleton types were dropped - no model "
        "here can learn a useful vector from one occurrence.",
        "",
        "| Matrix | Representation | Rows filled | Coverage |",
        "|---|---|---|---|",
    ]
    for key, description, hits, pct in rows:
        lines.append(f"| `{key}.npy` | {description} | {hits:,} | {pct:.1f}% |")

    lines += [
        "",
        "## Why one shared random base",
        "",
        "Every matrix starts from the *same* seeded `Uniform(-0.25, 0.25)` draw, and each "
        "representation overwrites only the rows it covers. Uncovered rows are therefore "
        "byte-identical across E0-E3.",
        "",
        "This matters for the validity of runs 3-6. Drawing fresh noise per matrix would "
        "mean E1's uncovered third and E2's uncovered fifth held *different* random "
        "values, so part of any measured F1 difference would be attributable to that "
        "noise rather than to the embeddings. With a shared base, the only thing that "
        "differs between the four runs is the vectors themselves.",
        "",
        "## Note on E3",
        "",
        "FastText fills 100% of rows because it synthesises a vector for any string from "
        "character n-grams. That is not evidence of superiority - see the caveat in "
        "`coverage.md`. It does mean E3 is the only representation with no random rows "
        "at all, which is itself a difference between the runs and should be mentioned "
        "when interpreting them.",
        "",
        "## Consumption",
        "",
        "```python",
        "import json, numpy as np",
        "vocab = json.load(open('models/emb_matrices/vocab.json'))",
        "index = {t: i for i, t in enumerate(vocab)}",
        "weights = np.load('models/emb_matrices/E2.npy')   # swap E0/E1/E2/E3 here",
        "```",
        "",
        "Row order is identical across all four files. Loading a different file must be "
        "the *only* change between runs 3-6.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}/ ({len(saved)} matrices + vocab.json)")
    print(f"wrote {REPORT.relative_to(REPO_ROOT)}")

    if problems:
        print("\nFAILED:")
        for p in problems:
            print("  -", p)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
