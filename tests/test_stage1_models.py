"""Tests for the Stage 1 BiLSTM (step 4.2).

torch is remote-only (PLAN F7), so this module skips locally and runs in the
Kaggle session before the four ablation runs are launched. The F8 arithmetic is
tested separately in `test_effective_batch.py`, which needs no torch.

The property worth the most here is padding invariance. A masking bug in the
attention, or a missing `pack_padded_sequence`, does not raise - it just makes
short sentences worse, which reads as a modelling result rather than a defect.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is remote-only (PLAN F7)")

from src.models.bilstm import BiLSTMClassifier, class_weights  # noqa: E402


# --------------------------------------------------------------------------
# BiLSTM
# --------------------------------------------------------------------------

VOCAB, DIM = 40, 16


@pytest.fixture
def matrix():
    generator = torch.Generator().manual_seed(0)
    m = torch.rand(VOCAB, DIM, generator=generator).numpy()
    m[0] = 0.0          # <pad>
    return m


@pytest.fixture
def model(matrix):
    net = BiLSTMClassifier(matrix, hidden_dim=8, dropout=0.0, attn_dim=8)
    net.eval()          # dropout off: these are determinism tests
    return net


def test_forward_shape(model):
    ids = torch.randint(2, VOCAB, (4, 12))
    lengths = torch.tensor([12, 9, 5, 1])

    assert model(ids, lengths).shape == (4, 2)


def test_padding_does_not_change_the_output(model):
    """The core correctness property.

    The same sentence, padded to two different widths, must produce the same
    logits. If it does not, padding is reaching either the recurrence or the
    attention softmax - and because padding is the majority of positions in a
    batch of short sentences, that leak is large, not negligible.
    """
    sentence = torch.randint(2, VOCAB, (1, 6))
    length = torch.tensor([6])

    short = model(sentence, length)
    padded = model(torch.cat([sentence, torch.zeros(1, 20, dtype=torch.long)], dim=1),
                   length)

    assert torch.allclose(short, padded, atol=1e-5)


def test_batching_does_not_change_the_output(model):
    """A sentence must score the same alone as it does beside a longer one."""
    ids = torch.zeros(2, 10, dtype=torch.long)
    ids[0, :3] = torch.randint(2, VOCAB, (3,))
    ids[1, :10] = torch.randint(2, VOCAB, (10,))
    lengths = torch.tensor([3, 10])

    batched = model(ids, lengths)
    alone = model(ids[:1], lengths[:1])

    assert torch.allclose(batched[0], alone[0], atol=1e-5)


def test_attention_ignores_padding(model):
    ids = torch.zeros(1, 10, dtype=torch.long)
    ids[0, :4] = torch.randint(2, VOCAB, (4,))
    lengths = torch.tensor([4])

    _, weights = model(ids, lengths, return_attention=True)

    assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1e-5)
    assert torch.allclose(weights[0, 4:], torch.zeros(6), atol=1e-6)


def test_frozen_embeddings_receive_no_gradient(matrix):
    """Runs 3-6 in the frozen condition measure the vectors themselves. If the
    embedding layer trains, E0's random rows learn and the floor lifts."""
    net = BiLSTMClassifier(matrix, hidden_dim=8, freeze_embeddings=True)

    assert not net.embedding.weight.requires_grad
    assert net.trainable_parameters() < sum(p.numel() for p in net.parameters())


def test_unfrozen_embeddings_do_receive_gradient(matrix):
    net = BiLSTMClassifier(matrix, hidden_dim=8, freeze_embeddings=False)

    assert net.embedding.weight.requires_grad
    assert net.trainable_parameters() == sum(p.numel() for p in net.parameters())


def test_pad_row_stays_zero_when_fine_tuning(matrix):
    """`padding_idx=0` must survive `from_pretrained(freeze=False)`, or the pad
    vector drifts away from zero during training."""
    net = BiLSTMClassifier(matrix, hidden_dim=8, freeze_embeddings=False)

    assert net.embedding.padding_idx == 0
    assert torch.allclose(net.embedding.weight[0], torch.zeros(DIM))


def test_config_records_the_ablation_variable(matrix):
    net = BiLSTMClassifier(matrix, hidden_dim=8, freeze_embeddings=True)

    assert net.config["freeze_embeddings"] is True
    assert net.config["vocab_size"] == VOCAB
    assert net.config["embed_dim"] == DIM


# --------------------------------------------------------------------------
# class weights
# --------------------------------------------------------------------------

def test_class_weights_favour_the_minority():
    labels = [0] * 80 + [1] * 20

    w = class_weights(labels)

    assert w[1] > w[0]
    assert w.mean() == pytest.approx(1.0)


def test_class_weights_are_flat_when_balanced():
    w = class_weights([0] * 50 + [1] * 50)

    assert torch.allclose(w, torch.ones(2))


def test_class_weights_survive_an_absent_class():
    """`clamp(min=1)` keeps a division by zero out of the loss."""
    w = class_weights([0] * 10)

    assert torch.isfinite(w).all()
