"""Step 4.2 - the BiLSTM classifier trained four times for runs 3-6.

One class, embedding matrix injected. That injection is the whole point: runs
3-6 load `E0_random.npy`, `E1.npy`, `E2.npy`, `E3.npy` into the *same* code with
the *same* hyperparameters and the *same* seed, so any difference in macro-F1 is
attributable to the vectors and nothing else (PLAN F8).

Architecture, and why each piece is here:

    ids -> Embedding(frozen or fine-tuned) -> BiLSTM -> additive attention -> MLP -> 2

**Additive attention rather than last-hidden-state or mean pooling.** PRD 8.1
tier T3 asks for it, and it earns its place beyond that: an ADE sentence is
usually positive because of a short span ("developed hepatotoxicity") inside a
long clinical description, so a pooling function that can concentrate on a few
positions is a better match for the label than one that averages 18 tokens.
The attention weights are also directly inspectable, which makes the Phase 6
error taxonomy something you can look at rather than guess at.

**`pack_padded_sequence`.** Without it the LSTM consumes padding as though it
were text, and the final states of short sentences are dominated by whatever the
pad embedding drifts to. Row 0 of every matrix is zero and `padding_idx=0` keeps
it that way, but packing is what makes it structurally impossible for padding to
reach the recurrence at all.

**Frozen vs fine-tuned embeddings is a flag, not a default.** See
`notebooks/stage1_remote.ipynb`: the ablation is run in both conditions, because
they answer different questions. Frozen asks "how good are these vectors";
fine-tuned asks "does the initialisation still matter once the task has had its
say". Reporting only one of the two would be choosing the answer in advance.

torch is imported at module scope: this file only ever executes on Kaggle/Colab
(PLAN F7), and the local environment tests it via `pytest.importorskip`.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD_ID = 0


class AdditiveAttention(nn.Module):
    """Bahdanau-style attention pooling over time.

    Masking is not optional here. Softmax over unmasked padding positions would
    hand real probability mass to embeddings of `<pad>`, and because padding is
    the *majority* of positions in a batch of short sentences, that leak is
    large rather than negligible.
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.project = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, states: torch.Tensor, mask: torch.Tensor):
        """states (B, T, H), mask (B, T) with True on real tokens.

        Returns (pooled (B, H), weights (B, T)).
        """
        scores = self.score(torch.tanh(self.project(states))).squeeze(-1)   # (B, T)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), states).squeeze(1)          # (B, H)
        return pooled, weights


class BiLSTMClassifier(nn.Module):
    """BiLSTM + attention sentence classifier with an injected embedding matrix.

    Args:
        embedding_matrix: (vocab, dim) float array. Row 0 must be `<pad>`.
        num_classes: 2 for Stage 1.
        hidden_dim: units *per direction*; the pooled representation is 2x this.
        freeze_embeddings: if True the matrix receives no gradient and the run
            measures the pretrained vectors themselves.
    """

    def __init__(
        self,
        embedding_matrix,
        num_classes: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.5,
        attn_dim: int = 128,
        freeze_embeddings: bool = True,
    ):
        super().__init__()

        weights = torch.as_tensor(embedding_matrix, dtype=torch.float32)
        vocab_size, embed_dim = weights.shape

        self.embedding = nn.Embedding.from_pretrained(
            weights, freeze=freeze_embeddings, padding_idx=PAD_ID
        )
        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = AdditiveAttention(2 * hidden_dim, attn_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

        self.config = {
            "vocab_size": int(vocab_size), "embed_dim": int(embed_dim),
            "hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout,
            "attn_dim": attn_dim, "freeze_embeddings": bool(freeze_embeddings),
            "num_classes": num_classes, "pooling": "additive_attention",
        }

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor,
                return_attention: bool = False):
        """ids (B, T) int64, lengths (B,) int64 of pre-padding lengths."""
        mask = torch.arange(ids.size(1), device=ids.device)[None, :] < lengths[:, None]

        embedded = self.embed_dropout(self.embedding(ids))

        # enforce_sorted=False lets the DataLoader stay in split order, which
        # keeps predictions aligned with the parquet rows for error analysis.
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        states, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=ids.size(1)
        )

        pooled, weights = self.attention(states, mask)
        logits = self.classifier(self.dropout(pooled))

        return (logits, weights) if return_attention else logits

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_matrix_file(cls, path: str | Path, **kwargs) -> "BiLSTMClassifier":
        """Build from one of `models/emb_matrices/*.npy` - how runs 3-6 differ."""
        import numpy as np

        return cls(np.load(Path(path)), **kwargs)


def class_weights(labels, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency weights for the loss, normalised to mean 1.

    The split is roughly 1:4, and macro-F1 is the primary metric, so an unweighted
    loss optimises the wrong thing: it buys accuracy on `not-ADE` at the cost of
    exactly the recall the report is about. Normalising to mean 1 keeps the loss
    on a comparable scale to an unweighted run, so the learning rate does not
    have to be retuned when the flag changes.

    Computed from the TRAINING labels only, and identical across runs 3-6.
    """
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long),
                            minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights / weights.mean()
