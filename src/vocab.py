"""Words -> integer ids, against the one frozen vocabulary.

`models/emb_matrices/vocab.json` is a list of 12,220 words. Row 0 is `<pad>`,
row 1 is `<unk>`, and every embedding matrix (E0/E1/E2/E3) uses this exact row
order. That is what makes the embedding comparison in notebook 04 fair: the four
runs differ only in the numbers, never in the word list.

The vocabulary was built from words containing at least one alphanumeric
character, so commas and brackets are not in it. Encoding applies the same
filter - otherwise every punctuation mark would resolve to `<unk>`, and `<unk>`
would come to mean "a comma" far more often than "a rare drug name", which is
the signal the comparison is trying to measure.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.tokenizer import tokenize

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB = REPO_ROOT / "models" / "emb_matrices" / "vocab.json"


def load_vocab(path: str | Path | None = None) -> tuple[list[str], dict[str, int]]:
    """Load the frozen vocabulary as (word list, word -> id)."""
    path = Path(path or DEFAULT_VOCAB)
    vocab = json.loads(path.read_text(encoding="utf-8"))

    if vocab[PAD_ID] != PAD or vocab[UNK_ID] != UNK:
        raise ValueError(f"{path}: row 0 must be {PAD!r} and row 1 {UNK!r}")

    return vocab, {t: i for i, t in enumerate(vocab)}


def is_indexable(token: str) -> bool:
    """The filter used when the vocabulary was built: keep anything alphanumeric."""
    return any(ch.isalnum() for ch in token)


def encode_tokens(tokens: list[str], index: dict[str, int]) -> list[int]:
    """Map already-split words to ids, dropping punctuation."""
    return [index.get(t, UNK_ID) for t in tokens if is_indexable(t)]


def encode_tokens_with_tags(
    tokens: list[str], tags: list[str], index: dict[str, int]
) -> tuple[list[int], list[str]]:
    """Encode words and drop the matching tags in lockstep (Stage 2).

    Filtering the ids without filtering the tags shifts every sentence
    containing a comma by one from that comma onwards. It does not raise; it
    just trains the tagger on wrong labels. Hence one function doing both.
    """
    if len(tokens) != len(tags):
        raise ValueError(f"{len(tokens)} tokens but {len(tags)} tags")

    kept = [(index.get(t, UNK_ID), tag) for t, tag in zip(tokens, tags) if is_indexable(t)]
    if not kept:
        return [], []

    ids, out_tags = zip(*kept)
    return list(ids), list(out_tags)


def encode(text: str, index: dict[str, int], max_len: int | None = None) -> list[int]:
    """Tokenise and encode one string. Truncates, never pads."""
    tokens, _ = tokenize(text)
    ids = encode_tokens(tokens, index)
    return ids[:max_len] if max_len else ids


def encode_batch(
    texts, index: dict[str, int], max_len: int, pad_id: int = PAD_ID
) -> tuple[list[list[int]], list[int]]:
    """Encode many strings into a padded rectangle. Returns (ids, lengths).

    Lengths are clamped to at least 1 because `pack_padded_sequence` rejects a
    zero length; such a row is all padding and contributes nothing anyway.
    """
    ids, lengths = [], []
    for text in texts:
        row = encode(text, index, max_len)
        lengths.append(max(len(row), 1))
        ids.append(row + [pad_id] * (max_len - len(row)))
    return ids, lengths


def unk_rate(texts, index: dict[str, int]) -> tuple[float, int, int]:
    """Fraction of words that fall through to `<unk>`. Returns (rate, unks, total)."""
    total = unks = 0
    for text in texts:
        for i in encode(text, index):
            total += 1
            unks += i == UNK_ID
    return (unks / total if total else 0.0), unks, total
