"""Tests for `src.models.encoding` - the shared text -> ids path.

Runs 3-6 and runs 9-10 both encode through this module. If it drifts, the
embedding matrices stop lining up with the ids that index them, and the symptom
is not a crash but a quietly worse F1 - which would be read as a result.
"""

from __future__ import annotations

import json

import pytest

from src.models.encoding import (
    DEFAULT_VOCAB,
    PAD,
    PAD_ID,
    UNK,
    UNK_ID,
    encode,
    encode_batch,
    encode_tokens,
    is_indexable,
    load_vocab,
    unk_rate,
)


@pytest.fixture
def toy_index():
    vocab = [PAD, UNK, "patient", "developed", "hepatotoxicity", "after"]
    return vocab, {t: i for i, t in enumerate(vocab)}


# --------------------------------------------------------------------------
# vocabulary contract
# --------------------------------------------------------------------------

def test_load_vocab_reserved_rows(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps([PAD, UNK, "aspirin"]), encoding="utf-8")

    vocab, index = load_vocab(path)

    assert vocab[PAD_ID] == PAD
    assert vocab[UNK_ID] == UNK
    assert index["aspirin"] == 2


def test_load_vocab_rejects_wrong_reserved_rows(tmp_path):
    """An embedding whose row 0 is a real word trains on a padding vector that
    receives gradient. Loud failure beats a silent 1pp."""
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(["aspirin", UNK, PAD]), encoding="utf-8")

    with pytest.raises(ValueError, match="expected row 0"):
        load_vocab(path)


def test_real_vocab_matches_embedding_matrices():
    """The committed vocab and every matrix present must agree on row count.

    Skips rather than fails when the matrices are absent. They are gitignored by
    design (PLAN F7 - they live in the versioned Kaggle Dataset), so a fresh
    clone legitimately has `vocab.json` and no `.npy` files, and a test that
    failed there would be reporting the plan rather than a defect. Where the
    matrices *are* present, the row counts still have to line up - a mismatch
    means the vocabulary and the matrices came from different builds, and every
    embedding lookup after that is off by some number of rows.
    """
    numpy = pytest.importorskip("numpy")

    matrices = sorted(DEFAULT_VOCAB.parent.glob("E*.npy"))
    if not matrices:
        pytest.skip("no embedding matrices here - attach the Kaggle Dataset, or "
                    "run scripts/build_embedding_matrices.py")

    vocab, _ = load_vocab()
    for path in matrices:
        rows = numpy.load(path, mmap_mode="r").shape[0]
        assert rows == len(vocab), f"{path.name}: {rows} rows vs vocab {len(vocab)}"


# --------------------------------------------------------------------------
# the punctuation filter
# --------------------------------------------------------------------------

def test_is_indexable_matches_vocab_construction():
    assert is_indexable("aspirin")
    assert is_indexable("5-fluorouracil")
    assert is_indexable("P<0.05")
    assert not is_indexable(",")
    assert not is_indexable("(")
    assert not is_indexable("--")


def test_encode_drops_punctuation(toy_index):
    """Punctuation is absent from the vocabulary, so encoding it would make
    `<unk>` mean 'a comma' far more often than 'a rare drug name'."""
    _, index = toy_index

    ids = encode("patient developed hepatotoxicity, after.", index)

    assert ids == [index["patient"], index["developed"],
                   index["hepatotoxicity"], index["after"]]
    assert UNK_ID not in ids


def test_unknown_words_become_unk(toy_index):
    _, index = toy_index

    ids = encode("patient developed rhabdomyolysis", index)

    assert ids == [index["patient"], index["developed"], UNK_ID]


def test_encode_tokens_and_encode_agree(toy_index):
    from src.tokenizer import tokenize

    _, index = toy_index
    text = "patient developed hepatotoxicity after 20 mg/kg."
    tokens, _ = tokenize(text)

    assert encode_tokens(tokens, index) == encode(text, index)


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------

def test_encode_batch_pads_to_rectangle(toy_index):
    _, index = toy_index

    ids, lengths = encode_batch(["patient developed", "after"], index, max_len=5)

    assert [len(row) for row in ids] == [5, 5]
    assert lengths == [2, 1]
    assert ids[0][2:] == [PAD_ID] * 3
    assert ids[1][1:] == [PAD_ID] * 4


def test_encode_batch_truncates(toy_index):
    _, index = toy_index

    ids, lengths = encode_batch(["patient developed hepatotoxicity after"],
                                index, max_len=2)

    assert lengths == [2]
    assert ids[0] == [index["patient"], index["developed"]]


def test_empty_sentence_gets_length_one(toy_index):
    """`pack_padded_sequence` rejects length 0. A sentence of pure punctuation
    encodes to nothing, and must not take down the whole run."""
    _, index = toy_index

    ids, lengths = encode_batch(["...", ""], index, max_len=4)

    assert lengths == [1, 1]
    assert all(row == [PAD_ID] * 4 for row in ids)


def test_truncation_applies_before_padding(toy_index):
    _, index = toy_index
    max_len = 3

    ids, lengths = encode_batch(
        ["patient developed hepatotoxicity after", "patient"], index, max_len
    )

    assert all(len(row) == max_len for row in ids)
    assert lengths == [3, 1]


# --------------------------------------------------------------------------
# corpus-level sanity
# --------------------------------------------------------------------------

def test_unk_rate_counts_only_indexable_tokens(toy_index):
    _, index = toy_index

    rate, unks, total = unk_rate(["patient developed rhabdomyolysis, ..."], index)

    assert (unks, total) == (1, 3)
    assert rate == pytest.approx(1 / 3)


def test_real_corpus_stays_in_range():
    """Every id the encoder can emit must index a row that exists."""
    pd = pytest.importorskip("pandas")

    vocab, index = load_vocab()
    df = pd.read_parquet(
        DEFAULT_VOCAB.parents[2] / "data" / "splits" / "stage1_test.parquet"
    )

    ids, lengths = encode_batch(df.text.tolist()[:500], index, max_len=96)

    assert max(max(row) for row in ids) < len(vocab)
    assert all(1 <= n <= 96 for n in lengths)
