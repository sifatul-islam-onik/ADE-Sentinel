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


def covers(kv, word: str) -> bool:
    """Whether the embedding has a vector for `word`.

    A case-folded retry is allowed because GloVe is an uncased release. Without
    it, GloVe would be penalised for *our* choice to protect medical
    abbreviations from lowercasing (`HIV` stays `HIV`, GloVe holds `hiv`).
    """
    return word in kv or word.lower() in kv
