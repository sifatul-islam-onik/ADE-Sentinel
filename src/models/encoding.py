"""Text -> integer ids, against the frozen vocabulary from step 3.6.

This module is deliberately torch-free. Two reasons:

1. The local environment has no torch (PLAN F7), and the encoding has to be
   unit-testable on the machine that authors it.
2. Stage 1 (runs 3-6) and Stage 2 (runs 9-10) must encode text *identically*.
   Sharing one tested function is the only way to guarantee that; two
   near-identical loops in two notebooks is how the vocabularies drift apart.

The one subtlety is which tokens are allowed to exist. `models/emb_matrices/
vocab.json` was built by `task_token_counts`, which keeps only tokens containing
an alphanumeric character - so commas, parentheses and full stops are not in it.
Encoding must apply the same filter. If it did not, every punctuation mark would
resolve to `<unk>`, and `<unk>` would come to mean "a comma" far more often than
"a rare drug name" - which is exactly the signal the embedding ablation is
trying to measure.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.tokenizer import tokenize

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOCAB = REPO_ROOT / "models" / "emb_matrices" / "vocab.json"


def load_vocab(path: str | Path | None = None) -> tuple[list[str], dict[str, int]]:
    """Load the frozen vocabulary, returning (vocab list, token -> id index).

    Asserts the reserved rows are where every consumer assumes they are: an
    embedding matrix whose row 0 is not `<pad>` would train on a padding vector
    that receives gradient, and the bug is invisible in the loss curve.
    """
    path = Path(path or DEFAULT_VOCAB)
    vocab = json.loads(path.read_text(encoding="utf-8"))

    if vocab[PAD_ID] != PAD or vocab[UNK_ID] != UNK:
        raise ValueError(
            f"{path}: expected row {PAD_ID}={PAD!r} and row {UNK_ID}={UNK!r}, "
            f"found {vocab[PAD_ID]!r} and {vocab[UNK_ID]!r}"
        )

    return vocab, {t: i for i, t in enumerate(vocab)}


def is_indexable(token: str) -> bool:
    """The filter `task_token_counts` applied when the vocabulary was built."""
    return any(ch.isalnum() for ch in token)


def encode_tokens(tokens: list[str], index: dict[str, int]) -> list[int]:
    """Map already-tokenised text to ids, dropping non-indexable tokens."""
    return [index.get(t, UNK_ID) for t in tokens if is_indexable(t)]


def encode(text: str, index: dict[str, int], max_len: int | None = None) -> list[int]:
    """Tokenise and encode one string. Truncates, never pads."""
    tokens, _ = tokenize(text)
    ids = encode_tokens(tokens, index)
    return ids[:max_len] if max_len else ids


def encode_batch(
    texts, index: dict[str, int], max_len: int, pad_id: int = PAD_ID
) -> tuple[list[list[int]], list[int]]:
    """Encode many strings to a padded rectangle.

    Returns (ids, lengths). `lengths` is the pre-padding length clamped to at
    least 1: `pack_padded_sequence` rejects a zero length, and a sentence whose
    every token was filtered out would otherwise crash the first epoch rather
    than the first batch. Such a row is all padding and contributes nothing.
    """
    ids, lengths = [], []
    for text in texts:
        row = encode(text, index, max_len)
        lengths.append(max(len(row), 1))
        ids.append(row + [pad_id] * (max_len - len(row)))
    return ids, lengths


def unk_rate(texts, index: dict[str, int]) -> tuple[float, int, int]:
    """Fraction of indexable tokens that fall through to `<unk>`.

    Worth printing before a run: the four embedding matrices share one
    vocabulary, so this number is identical across runs 3-6 by construction. If
    it is not, the wrong vocab.json has been loaded.
    """
    total = unks = 0
    for text in texts:
        for i in encode(text, index):
            total += 1
            unks += i == UNK_ID
    return (unks / total if total else 0.0), unks, total
