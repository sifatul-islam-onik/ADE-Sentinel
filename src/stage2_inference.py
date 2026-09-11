"""Stage 2 inference outside the training scripts - steps 5.9 and 6.1, and the demo.

The training scripts decode the test split as a side effect of training. Two
later steps need a tagger applied to text it did not decode then: Phase 6.1 runs
Stage 2 over Stage 1's false positives, which are not in the Stage 2 split at
all, and step 5.9 runs an external reference tagger over our test split. Both
must see exactly the word stream runs 9-11 were scored on, or their entity
counts stop being comparable with the logged ones.

So the word-level preparation lives here once, and `reproduce_saved` checks this
path against the predictions the remote session wrote before anything
downstream is allowed to trust it. CPU-only; needs `torch`, plus `transformers`
for the BERT taggers (requirements-dev.txt).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.bio_convert import to_bio
from src.models.encoding import is_indexable

BILSTM_MAX_LEN = 96     # scripts/train_bilstm_tagger.py MAX_LEN - what runs 9-10 saw
BERT_MAX_LEN = 192      # scripts/train_bert_tagger.py --max-len - what run 11 saw


def words_of(text: str) -> list[str]:
    """The word stream runs 9-11 were trained and scored on, for any sentence.

    `to_bio` with no spans is just the project tokenizer; the punctuation filter
    is the one both training scripts apply. A sentence Stage 2 never saw at
    training time goes through exactly this and nothing else.
    """
    tokens, _ = to_bio(text, [], strict=False)
    return [t for t in tokens if is_indexable(t)]


class BertTagger:
    """A token-classification checkpoint, read at word level.

    Each word's tag is taken from its first subword, exactly as
    `train_bert_tagger.decode_to_words` does, and words past the subword budget
    are tagged `O`, as they were when run 11 was scored.

    `label_map` renames the checkpoint's labels onto the project inventory, for
    reference models whose config names them only `LABEL_0`..`LABEL_4`.
    """

    def __init__(self, name_or_path, *, label_map=None, revision=None,
                 max_len: int = BERT_MAX_LEN, batch_size: int = 32):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, revision=revision)
        self.model = AutoModelForTokenClassification.from_pretrained(
            name_or_path, revision=revision)
        self.model.eval()
        self.max_len, self.batch_size = max_len, batch_size

        id2label = self.model.config.id2label
        raw = [id2label[i] for i in range(len(id2label))]
        self.raw_labels = raw
        self.labels = [label_map[label] for label in raw] if label_map else raw

    def tag(self, sentences: list[list[str]]) -> list[list[str]]:
        out: list[list[str]] = []
        for start in range(0, len(sentences), self.batch_size):
            chunk = sentences[start:start + self.batch_size]
            tagged = iter(self._tag_batch([s for s in chunk if s]) if any(chunk) else [])
            out.extend(next(tagged) if words else [] for words in chunk)
        return out

    def _tag_batch(self, chunk: list[list[str]]) -> list[list[str]]:
        encoded = self.tokenizer(chunk, is_split_into_words=True, truncation=True,
                                 max_length=self.max_len, padding=True,
                                 return_tensors="pt")
        with self._torch.no_grad():
            best = self.model(**encoded).logits.argmax(-1).tolist()

        results = []
        for i, words in enumerate(chunk):
            by_word, previous = {}, None
            for position, word_id in enumerate(encoded.word_ids(batch_index=i)):
                if word_id is not None and word_id != previous:
                    by_word[word_id] = self.labels[best[i][position]]
                previous = word_id
            results.append([by_word.get(w, "O") for w in range(len(words))])
        return results


class BiLSTMTaggerRunner:
    """A saved run 9/10 checkpoint, rebuilt against the frozen embedding matrix."""

    def __init__(self, checkpoint, matrices, *, max_len: int = BILSTM_MAX_LEN,
                 batch_size: int = 64):
        import numpy as np
        import torch

        from src.models.bilstm_tagger import BiLSTMTagger
        from src.models.encoding import load_vocab

        self._torch = torch
        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        config = saved["config"]
        matrix = np.load(Path(matrices) / f"{saved['embedding']}.npy")

        self.model = BiLSTMTagger(
            matrix, num_tags=config["num_tags"], hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"], dropout=config["dropout"],
            use_crf=config["use_crf"], freeze_embeddings=config["freeze_embeddings"])
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

        _, self.index = load_vocab(Path(matrices) / "vocab.json")
        self.tags = list(saved["tags"])
        self.max_len, self.batch_size = max_len, batch_size

    def tag(self, sentences: list[list[str]]) -> list[list[str]]:
        from src.models.encoding import encode_tokens

        torch = self._torch
        out: list[list[str]] = []
        for start in range(0, len(sentences), self.batch_size):
            chunk = sentences[start:start + self.batch_size]
            ids = [encode_tokens(words, self.index)[:self.max_len] for words in chunk]
            live = [i for i, row in enumerate(ids) if row]

            decoded = {}
            if live:
                width = max(len(ids[i]) for i in live)
                matrix = torch.tensor([ids[i] + [0] * (width - len(ids[i])) for i in live])
                lengths = torch.tensor([len(ids[i]) for i in live])
                for i, path in zip(live, self.model.decode(matrix, lengths)):
                    decoded[i] = [self.tags[t] for t in path]

            for i, words in enumerate(chunk):
                tags = decoded.get(i, [])
                out.append(tags + ["O"] * (len(words) - len(tags)))
        return out


def reproduce_saved(tagger, predictions_path) -> dict:
    """Re-decode the sentences a saved predictions file covers, and compare.

    This is the gate for Phase 6: a local inference path that disagreed with
    what the remote session decoded would turn run 12 into a measurement of the
    inference code rather than of the pipeline.
    """
    saved = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    words = [words_of(text) for text in saved["text"]]
    misaligned = sum(len(w) != len(g) for w, g in zip(words, saved["gold"]))

    pred = tagger.tag(words)
    return {
        "sentences": len(words),
        "misaligned_sentences": misaligned,
        "identical_sentences": sum(a == b for a, b in zip(pred, saved["pred"])),
        "differing_tokens": sum(x != y for a, b in zip(pred, saved["pred"])
                                for x, y in zip(a, b)),
        "tokens": sum(len(g) for g in saved["gold"]),
        "gold": saved["gold"],
        "pred": pred,
    }
