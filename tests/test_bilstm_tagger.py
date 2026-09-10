"""Tests for the Stage 2 tagger (steps 5.4-5.5).

torch is remote-only (PLAN F7), so this module skips locally and runs in the
Kaggle session before runs 9-11 are launched. The CRF tests additionally need
`pytorch-crf`, pinned in requirements-remote.txt.

The two properties worth the most:

* **`decode` returns exactly `lengths[i]` tags.** A tagger that returns padded
  rows scores padding as `O`, which inflates token accuracy, leaves entity-F1
  almost untouched, and therefore looks like a modelling result rather than a
  bug.
* **The CRF cannot emit an illegal transition.** That is the entire claim of run
  10, and it is a structural property of Viterbi decoding over learned
  transitions - so it should hold even at random initialisation, before any
  training. If it does not, the mask or the decode is wrong.
"""

from __future__ import annotations

import pytest

from src.bio_convert import TAGS, count_illegal_transitions

torch = pytest.importorskip("torch", reason="torch is remote-only (PLAN F7)")

from src.models.bilstm_tagger import BiLSTMTagger  # noqa: E402

VOCAB, DIM = 40, 16


@pytest.fixture
def matrix():
    generator = torch.Generator().manual_seed(0)
    m = torch.rand(VOCAB, DIM, generator=generator).numpy()
    m[0] = 0.0
    return m


@pytest.fixture
def softmax_tagger(matrix):
    net = BiLSTMTagger(matrix, hidden_dim=8, dropout=0.0, use_crf=False)
    net.eval()
    return net


def crf_tagger(matrix):
    pytest.importorskip("torchcrf", reason="pytorch-crf is remote-only")
    net = BiLSTMTagger(matrix, hidden_dim=8, dropout=0.0, use_crf=True)
    net.eval()
    return net


# --------------------------------------------------------------------------
# shapes and padding
# --------------------------------------------------------------------------

def test_emissions_have_one_score_per_tag_per_token(softmax_tagger):
    ids = torch.randint(2, VOCAB, (3, 10))
    lengths = torch.tensor([10, 6, 1])

    assert softmax_tagger(ids, lengths).shape == (3, 10, len(TAGS))


def test_decode_returns_exactly_the_real_tokens(softmax_tagger):
    """Ragged output by construction - padding cannot reach the scorer."""
    ids = torch.zeros(3, 12, dtype=torch.long)
    lengths = torch.tensor([12, 5, 1])
    ids[0, :12] = torch.randint(2, VOCAB, (12,))
    ids[1, :5] = torch.randint(2, VOCAB, (5,))
    ids[2, :1] = torch.randint(2, VOCAB, (1,))

    paths = softmax_tagger.decode(ids, lengths)

    assert [len(p) for p in paths] == [12, 5, 1]


def test_padding_does_not_change_the_emissions(softmax_tagger):
    """The same sentence padded to two widths must score identically."""
    sentence = torch.randint(2, VOCAB, (1, 6))
    length = torch.tensor([6])

    short = softmax_tagger(sentence, length)[:, :6]
    padded = softmax_tagger(
        torch.cat([sentence, torch.zeros(1, 14, dtype=torch.long)], dim=1), length
    )[:, :6]

    assert torch.allclose(short, padded, atol=1e-5)


def test_batching_does_not_change_the_decode(softmax_tagger):
    ids = torch.zeros(2, 9, dtype=torch.long)
    ids[0, :4] = torch.randint(2, VOCAB, (4,))
    ids[1, :9] = torch.randint(2, VOCAB, (9,))
    lengths = torch.tensor([4, 9])

    batched = softmax_tagger.decode(ids, lengths)
    alone = softmax_tagger.decode(ids[:1], lengths[:1])

    assert batched[0] == alone[0]


# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------

def test_softmax_loss_ignores_padding(softmax_tagger):
    """Whatever sits in the padded tag slots must not affect the loss."""
    ids = torch.zeros(2, 8, dtype=torch.long)
    ids[:, :3] = torch.randint(2, VOCAB, (2, 3))
    lengths = torch.tensor([3, 3])

    tags_a = torch.zeros(2, 8, dtype=torch.long)
    tags_b = torch.zeros(2, 8, dtype=torch.long)
    tags_b[:, 3:] = 4                      # garbage beyond the real tokens

    assert torch.allclose(softmax_tagger(ids, lengths, tags_a),
                          softmax_tagger(ids, lengths, tags_b))


def test_loss_is_a_scalar(softmax_tagger):
    ids = torch.randint(2, VOCAB, (4, 7))
    lengths = torch.tensor([7, 7, 7, 7])
    tags = torch.randint(0, len(TAGS), (4, 7))

    assert softmax_tagger(ids, lengths, tags).ndim == 0


# --------------------------------------------------------------------------
# the CRF - run 10's whole claim
# --------------------------------------------------------------------------

def test_crf_never_emits_an_illegal_transition(matrix):
    """Structural, so it holds even untrained. If this fails, run 10's headline
    number is measuring a bug rather than structured prediction."""
    net = crf_tagger(matrix)
    torch.manual_seed(0)
    ids = torch.randint(2, VOCAB, (16, 12))
    lengths = torch.randint(1, 13, (16,))

    for path in net.decode(ids, lengths):
        tags = [TAGS[i] for i in path]
        assert count_illegal_transitions(tags) == 0, tags


def test_crf_decode_respects_lengths(matrix):
    net = crf_tagger(matrix)
    ids = torch.randint(2, VOCAB, (3, 10))
    lengths = torch.tensor([10, 4, 2])

    assert [len(p) for p in net.decode(ids, lengths)] == [10, 4, 2]


def test_crf_loss_is_a_positive_scalar(matrix):
    """It is a negative log-likelihood, so it must be finite and non-negative."""
    net = crf_tagger(matrix)
    ids = torch.randint(2, VOCAB, (4, 6))
    lengths = torch.tensor([6, 6, 6, 6])
    tags = torch.zeros(4, 6, dtype=torch.long)

    loss = net(ids, lengths, tags)

    assert loss.ndim == 0
    assert torch.isfinite(loss) and loss >= 0


def test_crf_and_softmax_differ_only_in_the_head(matrix):
    """Runs 9 and 10 must share an architecture, or the comparison is not about
    structured prediction."""
    softmax = BiLSTMTagger(matrix, hidden_dim=8, use_crf=False)
    crf = crf_tagger(matrix)

    shared = {k: v for k, v in softmax.config.items() if k != "use_crf"}
    assert shared == {k: v for k, v in crf.config.items() if k != "use_crf"}
    assert softmax.crf is None and crf.crf is not None


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------

def test_frozen_embeddings_receive_no_gradient(matrix):
    net = BiLSTMTagger(matrix, hidden_dim=8, freeze_embeddings=True)

    assert not net.embedding.weight.requires_grad
    assert net.embedding.padding_idx == 0


def test_pad_row_stays_zero(matrix):
    net = BiLSTMTagger(matrix, hidden_dim=8, freeze_embeddings=False)

    assert torch.allclose(net.embedding.weight[0], torch.zeros(DIM))
