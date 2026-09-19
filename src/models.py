"""The two BiLSTM architectures: a sentence classifier and a word tagger.

Stage 1 (`BiLSTMClassifier`) reads a whole sentence and answers ADE / not-ADE.
Stage 2 (`BiLSTMTagger`) emits one tag per word.

Both take the embedding matrix as an argument rather than building one. That
injection is the entire point of the project's headline experiment: runs 3-6
load E0/E1/E2/E3 into the *same* code with the *same* hyperparameters and the
*same* seed, so any difference in the score comes from the vectors and nothing
else.

The transformer models (runs 7, 8, 11) are not here - they are `transformers`
one-liners, `AutoModelForSequenceClassification` and
`AutoModelForTokenClassification`, and notebook 04 shows the call.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.bio import TAGS

PAD_ID = 0
IGNORE_INDEX = -100     # what CrossEntropyLoss skips


class AdditiveAttention(nn.Module):
    """Bahdanau-style attention pooling over time.

    Masking is not optional. A softmax over unmasked padding positions hands
    real probability mass to the `<pad>` vector, and in a batch of short
    sentences padding is the *majority* of positions.
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.project = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, states: torch.Tensor, mask: torch.Tensor):
        """states (B, T, H), mask (B, T) True on real words. Returns (pooled, weights)."""
        scores = self.score(torch.tanh(self.project(states))).squeeze(-1)   # (B, T)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), states).squeeze(1)          # (B, H)
        return pooled, weights


class BiLSTMClassifier(nn.Module):
    """Stage 1: ids -> Embedding -> BiLSTM -> attention -> linear -> 2 classes.

    Attention rather than mean pooling because an ADE sentence is usually
    positive because of a short span ("developed hepatotoxicity") inside a long
    clinical description. A pooling function that can concentrate on a few
    positions matches the label better than one that averages 18 words.

    `freeze_embeddings=True` means the matrix gets no gradient, so the run
    measures the pretrained vectors themselves. Setting it False is the "u"
    (unfrozen) condition - notebook 04 reports both.
    """

    def __init__(
        self,
        embedding_matrix,
        num_classes: int = 2,
        hidden_dim: int = 256,      # per direction; the pooled vector is 2x this
        num_layers: int = 1,
        dropout: float = 0.5,
        attn_dim: int = 128,
        freeze_embeddings: bool = True,
    ):
        super().__init__()

        weights = torch.as_tensor(embedding_matrix, dtype=torch.float32)
        vocab_size, embed_dim = weights.shape

        self.embedding = nn.Embedding.from_pretrained(
            weights, freeze=freeze_embeddings, padding_idx=PAD_ID)
        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)

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

        # Packing is what makes it structurally impossible for padding to reach
        # the recurrence at all.
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        states, _ = pad_packed_sequence(packed_out, batch_first=True,
                                        total_length=ids.size(1))

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


class BiLSTMTagger(nn.Module):
    """Stage 2: the same encoder, one output per word, optionally a CRF on top.

    Runs 9 and 10 are this class with the same matrix, hyperparameters and seed;
    `use_crf` is the only difference between them.

    What the CRF is for: a per-word softmax decides each position independently,
    so nothing stops it emitting `I-DRUG` straight after `O` - a sequence that
    cannot be an entity. A CRF learns transition scores and decodes with
    Viterbi, so those paths become very unlikely. Note "unlikely", not
    "impossible": the constraint is learned from data, and notebook 05 shows the
    count dropping from 82 to 5, not to 0.
    """

    def __init__(
        self,
        embedding_matrix,
        num_tags: int = len(TAGS),
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.5,
        use_crf: bool = False,
        freeze_embeddings: bool = True,
    ):
        super().__init__()

        weights = torch.as_tensor(embedding_matrix, dtype=torch.float32)
        vocab_size, embed_dim = weights.shape

        self.embedding = nn.Embedding.from_pretrained(
            weights, freeze=freeze_embeddings, padding_idx=PAD_ID)
        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)

        self.dropout = nn.Dropout(dropout)
        self.emissions = nn.Linear(2 * hidden_dim, num_tags)

        self.use_crf = use_crf
        self.crf = None
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(num_tags, batch_first=True)

        self.config = {
            "vocab_size": int(vocab_size), "embed_dim": int(embed_dim),
            "hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout,
            "num_tags": num_tags, "use_crf": bool(use_crf),
            "freeze_embeddings": bool(freeze_embeddings),
        }

    def _encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """ids (B, T) -> emission scores (B, T, num_tags)."""
        embedded = self.embed_dropout(self.embedding(ids))

        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        states, _ = pad_packed_sequence(packed_out, batch_first=True,
                                        total_length=ids.size(1))

        return self.emissions(self.dropout(states))

    def forward(self, ids, lengths, tags=None):
        """Returns the loss when `tags` is given, else the emission scores."""
        emissions = self._encode(ids, lengths)
        mask = torch.arange(ids.size(1), device=ids.device)[None, :] < lengths[:, None]

        if tags is None:
            return emissions

        if self.use_crf:
            # torchcrf sums log-likelihood over the batch; negate for a loss and
            # divide by the batch so the scale is batch-size independent.
            log_likelihood = self.crf(emissions, tags, mask=mask, reduction="sum")
            return -log_likelihood / ids.size(0)

        # Padding is excluded by label, not by mask: CrossEntropyLoss has no
        # mask argument.
        targets = tags.masked_fill(~mask, IGNORE_INDEX)
        return nn.functional.cross_entropy(
            emissions.reshape(-1, emissions.size(-1)),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX)

    @torch.no_grad()
    def decode(self, ids: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        """Best tag sequence per sentence, padding stripped.

        Viterbi for the CRF, per-position argmax otherwise. Returning ragged
        lists rather than a padded tensor makes it impossible to accidentally
        score a padding position - a bug that inflates word accuracy, leaves
        entity-F1 untouched, and so looks like a modelling result.
        """
        emissions = self._encode(ids, lengths)
        mask = torch.arange(ids.size(1), device=ids.device)[None, :] < lengths[:, None]

        if self.use_crf:
            return self.crf.decode(emissions, mask=mask)

        best = emissions.argmax(dim=-1)
        return [best[i, : lengths[i]].tolist() for i in range(ids.size(0))]

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_matrix_file(cls, path: str | Path, **kwargs) -> "BiLSTMTagger":
        import numpy as np
        return cls(np.load(Path(path)), **kwargs)


def class_weights(labels, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency weights for the loss, normalised to mean 1.

    The split is about 1:4 and macro-F1 is the metric, so an unweighted loss
    optimises the wrong thing - it buys accuracy on not-ADE at the cost of
    exactly the ADE recall the project is about. Computed from the TRAINING
    labels only, and identical across runs 3-6.
    """
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long),
                            minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights / weights.mean()
