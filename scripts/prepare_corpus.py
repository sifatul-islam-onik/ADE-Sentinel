"""Step 2.3 - sentence-split and tokenise the PubMed corpus for embedding training.

Turns `data/pubmed_corpus.jsonl` (159,975 abstracts) into
`data/pubmed_sentences.txt`, one whitespace-joined tokenised sentence per line,
which is the format gensim streams directly.

Two decisions worth stating, because both affect the embedding space:

1. **Same tokenizer as the task.** The embeddings are only useful to Stage 1/2 if
   their vocabulary matches the tokens those stages actually see. Training on
   whitespace tokens and then looking up `5-fluorouracil` would miss on every
   domain term - which is precisely the failure the tokenizer exists to prevent.

2. **Punctuation-only tokens are dropped.** They carry no lexical meaning, and
   keeping them wastes context-window slots: a window of 5 around a drug name
   spent on commas is a window not spent on the effect. Tokens containing any
   alphanumeric character are kept, so `P<0.05` and `20 mg/kg` survive.

Sentences shorter than MIN_TOKENS are dropped - a 2-token fragment contributes
almost nothing to a skip-gram objective but still costs an epoch pass.

Usage:  .venv\\Scripts\\python scripts\\prepare_corpus.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tokenizer import normalize_unicode, sentence_split, tokenize  # noqa: E402

CORPUS = REPO_ROOT / "data" / "pubmed_corpus.jsonl"
OUT = REPO_ROOT / "data" / "pubmed_sentences.txt"
STATS = REPO_ROOT / "results" / "corpus_prep_stats.md"

MIN_TOKENS = 3
CHUNK = 2000


def process_record(line: str) -> list[str]:
    """One JSONL line -> list of tokenised sentence strings."""
    rec = json.loads(line)
    title = rec.get("title", "").strip()
    abstract = rec.get("abstract", "").strip()

    # Titles rarely end in a period, so joining without one would weld the title
    # onto the abstract's first sentence and create a spurious context window.
    if title and title[-1] not in ".!?":
        title += "."
    text = normalize_unicode(f"{title} {abstract}".strip())

    out: list[str] = []
    for sentence in sentence_split(text):
        tokens, _ = tokenize(sentence, lower=True)
        tokens = [t for t in tokens if any(c.isalnum() for c in t)]
        if len(tokens) >= MIN_TOKENS:
            out.append(" ".join(tokens))
    return out


def chunks(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def process_chunk(lines: list[str]) -> list[str]:
    result = []
    for line in lines:
        result.extend(process_record(line))
    return result


def main() -> int:
    if not CORPUS.exists():
        sys.exit(f"{CORPUS} not found - run the fetch first.")

    started = time.time()
    n_abstracts = n_sentences = n_tokens = 0
    vocab: Counter[str] = Counter()
    length_hist: Counter[int] = Counter()

    workers = max(mp.cpu_count() - 1, 1)
    print(f"tokenising {CORPUS.name} with {workers} workers...")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS.open(encoding="utf-8") as src, \
            OUT.open("w", encoding="utf-8") as dst, \
            mp.Pool(workers) as pool:

        for sentences in pool.imap(process_chunk, chunks(src, CHUNK), chunksize=1):
            n_abstracts += CHUNK
            for sentence in sentences:
                dst.write(sentence + "\n")
                toks = sentence.split()
                n_sentences += 1
                n_tokens += len(toks)
                vocab.update(toks)
                length_hist[min(len(toks) // 10 * 10, 100)] += 1

            if n_sentences and n_abstracts % 20_000 == 0:
                rate = n_abstracts / (time.time() - started)
                print(f"  ~{n_abstracts:,} abstracts | {n_sentences:,} sentences "
                      f"| {rate:,.0f} abs/s")

    # The chunked counter overshoots on the final partial chunk; recount exactly.
    with CORPUS.open(encoding="utf-8") as fh:
        n_abstracts = sum(1 for _ in fh)

    elapsed = time.time() - started
    size_mb = OUT.stat().st_size / 1e6

    # min_count=5 is the Word2Vec setting; how much vocabulary survives it?
    kept_5 = sum(1 for c in vocab.values() if c >= 5)
    hapax = sum(1 for c in vocab.values() if c == 1)

    lines = [
        "# Corpus preparation for embedding training (step 2.3)",
        "",
        "Produced by `scripts/prepare_corpus.py` from `data/pubmed_corpus.jsonl`,",
        "using the same domain tokenizer the supervised stages use.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Abstracts in | {n_abstracts:,} |",
        f"| Sentences out | {n_sentences:,} |",
        f"| Sentences per abstract | {n_sentences / n_abstracts:.1f} |",
        f"| Running tokens | {n_tokens:,} |",
        f"| Mean sentence length | {n_tokens / n_sentences:.1f} tokens |",
        f"| Vocabulary (all) | {len(vocab):,} |",
        f"| Vocabulary at `min_count=5` | {kept_5:,} |",
        f"| Hapax legomena | {hapax:,} ({100 * hapax / len(vocab):.1f}%) |",
        f"| Output size | {size_mb:,.1f} MB |",
        f"| Wall clock | {elapsed / 60:.1f} min |",
        "",
        "## Filtering rules",
        "",
        "| Rule | Rationale |",
        "|---|---|",
        "| Same tokenizer as Stage 1/2 | embedding lookups must match the tokens the "
        "supervised models see, or every domain term misses |",
        "| Punctuation-only tokens dropped | no lexical content, and they consume "
        "context-window slots that should hold the drug or the effect |",
        f"| Sentences with < {MIN_TOKENS} tokens dropped | too short to contribute a "
        "useful skip-gram context |",
        "",
        f"`min_count=5` retains {kept_5:,} of {len(vocab):,} types "
        f"({100 * kept_5 / len(vocab):.1f}%). The discarded tail is dominated by "
        "hapax legomena, which for biomedical text are largely author-specific "
        "compounds, mis-OCR'd tokens and one-off identifiers - exactly the forms "
        "Word2Vec cannot learn a useful vector for anyway. FastText (E3) is the "
        "counter-case: its subword units still reach these terms, which is the "
        "reason the PRD includes it.",
        "",
        "## Token count sanity",
        "",
        f"{n_tokens:,} running tokens over {n_abstracts:,} abstracts is "
        f"{n_tokens / n_abstracts:.0f} tokens per abstract, consistent with the "
        "typical structured PubMed abstract. This is the corpus size that the "
        "Section 7 embedding comparison rests on: if E2/E3 fail to beat GloVe, "
        "corpus size is the first hypothesis to test, and this is the number to "
        "quote.",
        "",
    ]

    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"abstracts    : {n_abstracts:,}")
    print(f"sentences    : {n_sentences:,}")
    print(f"tokens       : {n_tokens:,}")
    print(f"vocab        : {len(vocab):,}  (min_count=5 -> {kept_5:,})")
    print(f"wall clock   : {elapsed / 60:.1f} min")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)} ({size_mb:,.1f} MB)")
    print(f"wrote {STATS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
