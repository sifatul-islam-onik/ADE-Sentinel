"""Loading and inspecting the trained word vectors, without a big memory bill.

`gensim`'s own `most_similar` normalises the whole matrix in one go, which for
GloVe means a 400,000 x 300 temporary - 458 MB, on top of the 458 MB the vectors
already occupy. On a laptop that is enough to fail.

`nearest` below computes the same cosine neighbours in blocks, so peak extra
memory is a few tens of MB whatever the vocabulary size, and `load` memory-maps
the vectors rather than reading them in. Answers are identical to gensim's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = REPO_ROOT / "models"

# The three embeddings notebook 3 compares.
FILES = {"E1 GloVe (general)": "glove.kv",
         "E2 Word2Vec (ours)": "w2v.kv",
         "E3 FastText (ours)": "ft.kv"}


def load(name_or_file: str):
    """Load one of the saved embeddings, memory-mapped.

    Takes either a key of `FILES` ("E2 Word2Vec (ours)") or a filename ("w2v.kv").
    """
    from gensim.models import KeyedVectors

    filename = FILES.get(name_or_file, name_or_file)
    return KeyedVectors.load(str(MODELS / filename), mmap="r")


def nearest(kv, word: str, k: int = 5, block: int = 20_000) -> list[tuple[str, float]]:
    """The `k` nearest words by cosine similarity, or [] if `word` is unknown."""
    if word not in kv:
        return []

    probe = np.asarray(kv[word], dtype=np.float32)
    probe = probe / (np.linalg.norm(probe) or 1.0)

    vectors = kv.vectors
    scores = np.empty(len(vectors), dtype=np.float32)

    for start in range(0, len(vectors), block):
        chunk = np.asarray(vectors[start:start + block], dtype=np.float32)
        norms = np.sqrt(np.einsum("ij,ij->i", chunk, chunk))
        np.maximum(norms, 1e-12, out=norms)
        scores[start:start + len(chunk)] = (chunk @ probe) / norms

    # k + 1 because the word is its own nearest neighbour.
    top = np.argpartition(-scores, k + 1)[:k + 1]
    top = top[np.argsort(-scores[top])]
    return [(kv.index_to_key[i], float(scores[i])) for i in top
            if kv.index_to_key[i] != word][:k]


def lookup_key(word: str, kv, case_fallback: bool = True) -> str | None:
    """Resolve `word` against `kv`, retrying case-folded. None if absent.

    **Why the retry is not optional.** Our tokenizer deliberately protects
    medical abbreviations from lowercasing, so `HIV`, `CT` and `AML` stay
    uppercase. `glove.6B` is an UNCASED release holding `hiv`, `ct`, `aml`.
    Comparing the two without a case-folded retry counts those as GloVe
    failures when they are really artefacts of *our* preprocessing — which
    would inflate GloVe's miss rate and flatter the domain vectors.

    The project's headline claim has to survive a hostile reading, so the
    baseline is measured at its strongest fair version.
    """
    if word in kv:
        return word
    if case_fallback:
        lowered = word.lower()
        if lowered != word and lowered in kv:
            return lowered
    return None


def covers(kv, word: str) -> bool:
    """Whether the embedding has a vector for `word`, case-folding allowed."""
    return lookup_key(word, kv) is not None


def task_token_counts(texts) -> "Counter[str]":
    """Word frequencies over the task corpus, using the project's tokenizer.

    Must be the same tokenizer the supervised models use, or coverage is
    measured against words that never reach an embedding lookup at all.
    """
    from collections import Counter

    from src.tokenizer import tokenize
    from src.vocab import is_indexable

    counts: Counter[str] = Counter()
    for text in texts:
        tokens, _ = tokenize(text)
        counts.update(t for t in tokens if is_indexable(t))
    return counts


def build_matrix(index: dict[str, int], base, kv, skip: tuple[str, ...] = ()):
    """Fill a copy of `base` with vectors from `kv`. Returns (matrix, rows filled).

    `base` is shared across every representation on purpose. Drawing fresh noise
    per matrix would leave E1's uncovered rows and E2's uncovered rows holding
    DIFFERENT random values, so part of any downstream F1 difference would be
    that noise rather than the embeddings. Copying one seeded base keeps
    uncovered rows byte-identical, which is what makes runs 3-6 a clean
    single-variable ablation.
    """
    matrix = base.copy()
    hits = 0
    for word, row in index.items():
        if word in skip:
            continue
        key = lookup_key(word, kv)
        if key is not None:
            matrix[row] = kv[key]
            hits += 1
    return matrix, hits
