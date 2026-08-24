"""Step 2.2 - before/after evidence for the domain tokenizer.

Produces `results/figures/tokenizer_table.md`, which is a report figure, not
throwaway diagnostics. It measures the domain tokenizer against a naive
alphanumeric baseline on the ADE corpus and reports vocabulary size, OOV
behaviour, and concrete examples of tokens the naive pipeline destroys.

The argument the table has to make: the tokens naive tokenisation breaks are
precisely the informative ones. Splitting `5-fluorouracil` into `5` and
`fluorouracil` does not merely add a token, it deletes a drug.

Usage:  .venv\\Scripts\\python scripts\\tokenizer_report.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tokenizer import naive_tokenize, tokenize  # noqa: E402

SPLITS = REPO_ROOT / "data" / "splits"
OUT = REPO_ROOT / "results" / "figures" / "tokenizer_table.md"


def load_corpus_texts() -> list[str]:
    import pandas as pd

    frames = [pd.read_parquet(SPLITS / f"stage1_{s}.parquet")
              for s in ("train", "dev", "test")]
    return pd.concat(frames).text.tolist()


def main() -> int:
    if not (SPLITS / "stage1_train.parquet").exists():
        sys.exit("Splits not found - run scripts/build_splits.py first.")

    texts = load_corpus_texts()

    domain_vocab: Counter[str] = Counter()
    naive_vocab: Counter[str] = Counter()
    domain_tokens = naive_tokens = 0

    # Tokens the domain tokenizer keeps whole that the naive one fragments.
    kept_whole: Counter[str] = Counter()

    # The naive regex drops punctuation entirely; the domain tokenizer emits it.
    # Comparing raw counts would therefore measure punctuation, not tokenisation,
    # so the word-only count is tracked separately for the honest comparison.
    domain_word_tokens = 0

    for text in texts:
        d_toks, _ = tokenize(text)
        n_toks = naive_tokenize(text)
        domain_vocab.update(d_toks)
        naive_vocab.update(n_toks)
        domain_tokens += len(d_toks)
        naive_tokens += len(n_toks)
        domain_word_tokens += sum(1 for t in d_toks if any(c.isalnum() for c in t))

        for tok in d_toks:
            # A multi-part token the naive regex could never produce.
            if ("-" in tok or " " in tok or "<" in tok or "=" in tok) and len(tok) > 2:
                kept_whole[tok] += 1

    # Vocabulary items that exist only because of domain tokenisation.
    domain_only = [t for t in domain_vocab if t not in naive_vocab]
    # Case-protected forms: present as uppercase in domain, lowercased in naive.
    case_saved = sorted(
        t for t in domain_vocab
        if t.isupper() and t.lower() in naive_vocab and t not in naive_vocab
    )

    hapax_d = sum(1 for c in domain_vocab.values() if c == 1)
    hapax_n = sum(1 for c in naive_vocab.values() if c == 1)

    lines = [
        "# Domain tokenizer - before/after",
        "",
        f"Measured on the deduplicated ADE corpus ({len(texts):,} sentences) by",
        "`scripts/tokenizer_report.py` (PLAN step 2.2).",
        "",
        "**Baseline** is `re.findall(r'[A-Za-z0-9]+')` plus unconditional lowercasing -",
        "the default pipeline in most tutorials.",
        "",
        "## Aggregate",
        "",
        "| Measure | Naive baseline | Domain tokenizer | Delta |",
        "|---|---|---|---|",
        f"| Running tokens (all) | {naive_tokens:,} | {domain_tokens:,} | "
        f"{domain_tokens - naive_tokens:+,} |",
        f"| Running tokens (words only) | {naive_tokens:,} | {domain_word_tokens:,} | "
        f"{domain_word_tokens - naive_tokens:+,} |",
        f"| Vocabulary (types) | {len(naive_vocab):,} | {len(domain_vocab):,} | "
        f"{len(domain_vocab) - len(naive_vocab):+,} |",
        f"| Hapax legomena | {hapax_n:,} | {hapax_d:,} | {hapax_d - hapax_n:+,} |",
        f"| Type/token ratio | {len(naive_vocab)/naive_tokens:.4f} | "
        f"{len(domain_vocab)/domain_tokens:.4f} | |",
        "",
        "Read the *words only* row, not the first one: the naive regex discards "
        "punctuation while the domain tokenizer emits it, so the raw totals differ "
        "mostly by punctuation and say nothing about tokenisation quality.",
        "",
        f"On words alone the domain tokenizer emits **{naive_tokens - domain_word_tokens:,} "
        f"fewer tokens** ({domain_word_tokens:,} vs {naive_tokens:,}), because it "
        "refuses to shatter multi-part terms.",
        "",
        f"Vocabulary moves the other way: **{len(domain_vocab):,} types vs "
        f"{len(naive_vocab):,}**, because `long-term` and `drug-induced` are single "
        "types here and recycled fragments there. That is a real cost - a larger "
        "vocabulary means more embedding rows and more rare types - and it is the "
        f"right trade: **{len(domain_only):,} of those types cannot be represented by "
        "the naive pipeline at all**, and they are the domain-bearing ones. It also "
        "sharpens the Section 7 experiment rather than blunting it, since these are "
        "exactly the terms general-purpose GloVe vectors will fail to cover.",
        "",
        "## What the baseline destroys",
        "",
        "The 25 most frequent multi-part tokens the domain tokenizer keeps whole, each",
        "of which the naive baseline splits into meaningless fragments:",
        "",
        "| Token kept whole | Freq | Naive baseline produces |",
        "|---|---|---|",
    ]
    for tok, freq in kept_whole.most_common(25):
        pieces = " + ".join(f"`{p}`" for p in naive_tokenize(tok))
        lines.append(f"| `{tok}` | {freq:,} | {pieces or '-'} |")

    lines += [
        "",
        "## Case protection",
        "",
        "All-caps medical abbreviations that unconditional lowercasing would collapse",
        "into ordinary English words. `ALL` is the canonical example: lowercased it",
        "becomes a determiner and acute lymphoblastic leukemia disappears from the",
        "corpus entirely.",
        "",
        f"Protected forms observed in this corpus: **{len(case_saved)}**",
        "",
        "| Abbreviation | Occurrences | Collapses to |",
        "|---|---|---|",
    ]
    for tok in sorted(case_saved, key=lambda t: -domain_vocab[t])[:20]:
        lines.append(f"| `{tok}` | {domain_vocab[tok]:,} | `{tok.lower()}` |")

    lines += [
        "",
        "## Why this matters downstream",
        "",
        "Every fragmented token is a vocabulary entry the embedding layer must learn",
        "separately, and a token whose meaning has been destroyed. `5` and",
        "`fluorouracil` carry neither the identity of the drug nor its relationship to",
        "the effect it causes. Since the project's headline claim is about the quality",
        "of the embedding space (PRD section 7), the tokenizer sets the ceiling on what",
        "any embedding can achieve - which is why this table precedes the embedding",
        "experiment rather than following it.",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"sentences         : {len(texts):,}")
    print(f"tokens  naive/dom : {naive_tokens:,} / {domain_tokens:,}")
    print(f"vocab   naive/dom : {len(naive_vocab):,} / {len(domain_vocab):,}")
    print(f"domain-only types : {len(domain_only):,}")
    print(f"case-protected    : {len(case_saved)}")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
