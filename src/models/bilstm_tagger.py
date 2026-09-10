"""Steps 5.4-5.5 - the Stage 2 sequence tagger, with and without a CRF.

One class, one flag. Runs 9 and 10 load the same embedding matrix into the same
architecture with the same hyperparameters and the same seed; `use_crf` is the
only thing that differs. That mirrors the Stage 1 ablation deliberately - the
question "what does structured prediction buy" is only answerable if nothing
else moved.

**What the CRF is actually for.** A per-token softmax decides each position
independently, so nothing stops it emitting `I-DRUG` straight after `O` - a
sequence that cannot correspond to any entity. A CRF learns transition scores
between tags and decodes with Viterbi, so illegal transitions become
vanishingly unlikely. Step 5.8 counts them in both conditions, and that count is
the concrete evidence for the architectural claim; entity-F1 alone tends to move
only a little and does not explain *why*.

**Why this is not the Stage 1 classifier with a different head.** The classifier
pools the sequence to one vector through attention; a tagger must keep one
output per token, so there is no pooling step at all. Sharing a base class would
mean a base class whose only content is an embedding lookup and an LSTM - the
duplication is four lines, and the coupling would cost more than it saves.

**`DataParallel` and CRF do not mix** (PLAN F8): the CRF computes its loss inside
`forward`, so two GPUs return a per-device loss vector that needs `.mean()`.
Run this on a single GPU; the training script pins one by default.

torch is imported at module scope: this file only ever executes remotely
(PLAN F7), and the local environment tests it via `pytest.importorskip`.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.bio_convert import TAGS

PAD_ID = 0
IGNORE_INDEX = -100     # what CrossEntropyLoss skips; also what HF expects


class BiLSTMTagger(nn.Module):
    """BiLSTM token tagger, optionally topped with a CRF.

    Args:
        embedding_matrix: (vocab, dim) float array; row 0 must be `<pad>`.
        num_tags: size of the BIO inventory (5 for DRUG/EFFECT).
        hidden_dim: units *per direction*.
        use_crf: replace the per-token softmax loss with a linear-chain CRF.
        freeze_embeddings: if True the matrix receives no gradient.
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
            weights, freeze=freeze_embeddings, padding_idx=PAD_ID
        )
        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.emissions = nn.Linear(2 * hidden_dim, num_tags)

        self.use_crf = use_crf
        self.crf = None
        if use_crf:
            try:
                from torchcrf import CRF
            except ImportError as exc:      # pragma: no cover - remote-only path
                raise ImportError(
                    "pytorch-crf is required for run 10. Install the pinned "
                    "version: pip install pytorch-crf==0.7.2"
                ) from exc
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

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        states, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=ids.size(1)
        )

        return self.emissions(self.dropout(states))

    def forward(self, ids, lengths, tags=None):
        """Returns loss when `tags` is given, else emission scores.

        The CRF branch and the softmax branch must agree on what padding means.
        The mask is built from `lengths` in both, so a position beyond the real
        sentence contributes to neither the loss nor the decode.
        """
        emissions = self._encode(ids, lengths)
        mask = torch.arange(ids.size(1), device=ids.device)[None, :] < lengths[:, None]

        if tags is None:
            return emissions

        if self.use_crf:
            # torchcrf sums log-likelihood over the batch; negate for a loss and
            # divide by the batch so the scale does not depend on batch size.
            log_likelihood = self.crf(emissions, tags, mask=mask, reduction="sum")
            return -log_likelihood / ids.size(0)

        # Padding positions are excluded by label, not by mask, because
        # CrossEntropyLoss has no mask argument.
        targets = tags.masked_fill(~mask, IGNORE_INDEX)
        return nn.functional.cross_entropy(
            emissions.reshape(-1, emissions.size(-1)),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    @torch.no_grad()
    def decode(self, ids: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        """Best tag sequence per sentence, padding stripped.

        Viterbi for the CRF, per-position argmax otherwise. Returning ragged
        lists rather than a padded tensor is deliberate: it makes it impossible
        to score a padding position by forgetting to trim, which is a bug that
        inflates token accuracy and leaves entity-F1 untouched - so it looks like
        a modelling result.
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
