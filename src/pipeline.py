"""The two-stage pipeline, applied live to any text. This is what the demo runs.

    from src.pipeline import load_pipeline
    pipeline = load_pipeline("bilstm")
    for s in pipeline.analyse("Hepatotoxicity developed after isoniazid."):
        print(s.is_ade, s.confidence, s.entities)

The rules are the ones the reported end-to-end numbers were measured under:

* the gate's decision is the argmax (not a 0.5 threshold), and its confidence is
  the softmax probability of the class it chose;
* Stage 2 tags only the sentences the gate accepts;
* entities are read strictly, so what the demo highlights is what the scorer
  would have counted.

Text goes through the project tokenizer and the punctuation filter, with
character offsets kept, so entities can be painted back onto the original
string. CPU only - torch and transformers are imported inside the model classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.bio import strict_entities
from src.tokenizer import sentence_split, tokenize
from src.vocab import is_indexable

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = REPO_ROOT / "models"
MATRICES = MODELS / "emb_matrices"

# Input limits the saved checkpoints were trained and scored under.
GATE_BILSTM_MAX_LEN = 96
GATE_BERT_MAX_LEN = 128
TAGGER_BILSTM_MAX_LEN = 96
TAGGER_BERT_MAX_LEN = 192


@dataclass(frozen=True)
class Pair:
    """A Stage 1 gate and a Stage 2 tagger the demo can load together."""

    key: str
    label: str
    gate_run: str
    tagger_run: str
    pipeline_run: str       # the runs.csv row scoring this exact pair end to end
    description: str


PAIRS = {
    "bilstm": Pair(
        "bilstm", "BiLSTM gate + BiLSTM-CRF tagger", "6", "10", "12c",
        "Both checkpoints load in well under a second on CPU, so a cold start is "
        "mostly Streamlit and torch starting up."),
    "biomedbert": Pair(
        "biomedbert", "BiomedBERT gate + BiomedBERT tagger", "8", "11", "12",
        "The most accurate pair. Two 420 MB checkpoints take several seconds to "
        "load, so it is loaded only when selected."),
}
DEFAULT_PAIR = "bilstm"


@dataclass(frozen=True)
class Example:
    """A preloaded sentence. All five are held-out Stage 1 test sentences.

    Chosen to show one behaviour each, from sentences the default pair handles
    correctly - they illustrate the pipeline rather than measure it.
    `test_index` is the row in `data/splits/stage1_test.parquet`.
    """

    button: str
    test_index: int
    ade: bool                    # the corpus label
    note: str
    text: str


EXAMPLES = (
    Example("One drug, two effects", 36, True,
            "A typical case-report sentence: two effects, one drug, and a dose and an "
            "indication that are not entities.",
            "A 55-year-old woman presented an episode of acute urticaria and labial angioedema "
            "60 minutes after ingesting 500 mg of cloxacillin for a skin abscess."),
    Example("Two drugs", 73, True,
            "Both drugs in the combination are tagged as causes of the one effect.",
            "A case of toxic hepatitis caused by combination therapy with methotrexate and "
            "etretinate in the treatment of severe psoriasis is presented in a 47-year-old woman."),
    Example("Negated", 2446, False,
            "The negation covers the drug-effect relation, so there is no ADE. The tagger on its "
            "own would still mark sertraline and the incontinence; the gate is what stops it.",
            "The patient was then treated with sertraline without experiencing any incontinence "
            "episodes."),
    Example("Negation elsewhere", 34, True,
            "'without' negates the patient's history, not the relation, so this is still an ADE.",
            "A 53-year-old male, without any prior history of psychosis, developed schizophrenia "
            "4 days after starting low-dose bromocriptine therapy for a macroprolactinoma."),
    Example("Drugs, no ADE", 13, False,
            "Three drugs, but only as treatment. The gate rejects it and Stage 2 never runs.",
            "A 23-year-old woman with metastatic Sertoli-Leydig cell tumor was treated with "
            "cisplatin, vinblastine, and bleomycin."),
)


def logged_scores(pair: Pair, runs_csv: Path | None = None) -> dict:
    """The pair's test scores from `results/runs.csv`, None where not logged.

    stage1: macro-F1 of the gate. stage2: strict entity-F1 of the tagger on gold
    ADE sentences. pipeline: strict entity-F1 of the two chained, on every test
    sentence.
    """
    import csv

    path = runs_csv or REPO_ROOT / "results" / "runs.csv"
    latest = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                latest[row["run_id"]] = row

    def number(run_id, column):
        value = latest.get(run_id, {}).get(column, "")
        return float(value) if value else None

    return {"stage1": number(pair.gate_run, "macro_f1"),
            "stage2": number(pair.tagger_run, "entity_f1_strict"),
            "pipeline": number(pair.pipeline_run, "entity_f1_strict")}


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class Entity:
    label: str      # "DRUG" or "EFFECT"
    start: int      # character offsets into the sentence, end exclusive
    end: int
    text: str


@dataclass
class SentenceResult:
    text: str
    p_ade: float                 # the gate's probability of the ADE class
    is_ade: bool                 # the gate's argmax decision
    entities: list[Entity] = field(default_factory=list)
    truncated: bool = False      # longer than a model's input limit

    @property
    def confidence(self) -> float:
        """Probability of the class the gate chose."""
        return self.p_ade if self.is_ade else 1.0 - self.p_ade


def words_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """The word stream the models were scored on, with each word's character span."""
    tokens, offsets = tokenize(text, lower=True)
    kept = [(t, o) for t, o in zip(tokens, offsets) if is_indexable(t)]
    return [t for t, _ in kept], [o for _, o in kept]


def words_of(text: str) -> list[str]:
    """Just the words, for a tagger that does not need offsets."""
    return words_with_offsets(text)[0]


def entities_from_tags(text: str, offsets, tags: list[str]) -> list[Entity]:
    """Strict entities, mapped from word positions back onto the sentence."""
    out = []
    for first, last, label in strict_entities(tags):
        start, end = offsets[first][0], offsets[last - 1][1]
        out.append(Entity(label, start, end, text[start:end]))
    return out


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

class Pipeline:
    """Gate, then tagger - on each sentence of the input, in order."""

    def __init__(self, gate, tagger, pair: Pair | None = None):
        self.gate, self.tagger, self.pair = gate, tagger, pair

    def analyse(self, text: str) -> list[SentenceResult]:
        sentences = sentence_split(text)
        if not sentences:
            return []

        words = [words_with_offsets(s) for s in sentences]
        decisions = self.gate.classify(sentences)

        # Only the accepted sentences reach Stage 2 - that is the whole point of
        # a gate, and it is where the end-to-end error propagation comes from.
        accepted = [i for i, (_, is_ade) in enumerate(decisions) if is_ade]
        tagged = dict(zip(accepted, self.tagger.tag([words[i][0] for i in accepted])))

        results = []
        for i, (sentence, (p_ade, is_ade)) in enumerate(zip(sentences, decisions)):
            sentence_words, offsets = words[i]
            entities = entities_from_tags(sentence, offsets, tagged[i]) if i in tagged else []
            truncated = (_exceeds(self.gate, sentence, sentence_words)
                         or (is_ade and _exceeds(self.tagger, sentence, sentence_words)))
            results.append(SentenceResult(sentence, float(p_ade), bool(is_ade), entities,
                                          truncated))
        return results


def _exceeds(model, sentence: str, words: list[str]) -> bool:
    """Whether a sentence is longer than the model's input limit.

    BERT limits count subwords with special tokens; BiLSTM limits count words.
    """
    limit = getattr(model, "max_len", None)
    if limit is None:
        return False
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return len(words) > limit
    if hasattr(model, "tag"):      # a tagger reads pre-split words
        return bool(words) and len(tokenizer(words, is_split_into_words=True)["input_ids"]) > limit
    return len(tokenizer(sentence)["input_ids"]) > limit


def _decisions(torch, logits) -> list[tuple[float, bool]]:
    """(P(ADE), argmax is ADE) per row - the argmax, not a 0.5 threshold."""
    p_ade = torch.softmax(logits.float(), dim=1)[:, 1].tolist()
    chosen = logits.argmax(dim=1).tolist()
    return [(p, c == 1) for p, c in zip(p_ade, chosen)]


# --------------------------------------------------------------------------
# Stage 1 gates
# --------------------------------------------------------------------------

class BiLSTMGate:
    """A saved run 3-6 checkpoint, rebuilt against its frozen embedding matrix."""

    def __init__(self, checkpoint, matrices=MATRICES, *, max_len: int = GATE_BILSTM_MAX_LEN,
                 batch_size: int = 64):
        import numpy as np
        import torch

        from src.models import BiLSTMClassifier
        from src.vocab import load_vocab

        self._torch = torch
        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        config = saved["config"]
        matrix = np.load(Path(matrices) / f"{saved['embedding']}.npy")

        self.model = BiLSTMClassifier(
            matrix, num_classes=config["num_classes"], hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"], dropout=config["dropout"],
            attn_dim=config["attn_dim"], freeze_embeddings=config["freeze_embeddings"])
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

        _, self.index = load_vocab(Path(matrices) / "vocab.json")
        self.max_len, self.batch_size = max_len, batch_size

    def classify(self, sentences: list[str]) -> list[tuple[float, bool]]:
        from src.vocab import encode_batch

        torch = self._torch
        out = []
        for start in range(0, len(sentences), self.batch_size):
            ids, lengths = encode_batch(sentences[start:start + self.batch_size], self.index,
                                        self.max_len)
            with torch.no_grad():
                logits = self.model(torch.tensor(ids), torch.tensor(lengths))
            out += _decisions(torch, logits)
        return out


class BertGate:
    """A saved run 7-8 sequence-classification checkpoint."""

    def __init__(self, path, *, max_len: int = GATE_BERT_MAX_LEN, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()
        self.max_len, self.batch_size = max_len, batch_size

    def classify(self, sentences: list[str]) -> list[tuple[float, bool]]:
        torch = self._torch
        out = []
        for start in range(0, len(sentences), self.batch_size):
            encoded = self.tokenizer(sentences[start:start + self.batch_size], truncation=True,
                                     max_length=self.max_len, padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**encoded).logits
            out += _decisions(torch, logits)
        return out


# --------------------------------------------------------------------------
# Stage 2 taggers
# --------------------------------------------------------------------------

class BiLSTMTaggerRunner:
    """A saved run 9/10 checkpoint, rebuilt against the frozen embedding matrix."""

    def __init__(self, checkpoint, matrices=MATRICES, *, max_len: int = TAGGER_BILSTM_MAX_LEN,
                 batch_size: int = 64):
        import numpy as np
        import torch

        from src.models import BiLSTMTagger
        from src.vocab import load_vocab

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
        from src.vocab import encode_tokens

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


class BertTagger:
    """A token-classification checkpoint, read back at word level.

    Each word's tag is taken from its first subword, and words past the subword
    budget are tagged `O` - exactly how run 11 was scored, so the entity counts
    stay comparable with the BiLSTM taggers'.
    """

    def __init__(self, name_or_path, *, max_len: int = TAGGER_BERT_MAX_LEN,
                 batch_size: int = 32):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(name_or_path)
        self.model.eval()
        self.max_len, self.batch_size = max_len, batch_size

        id2label = self.model.config.id2label
        self.labels = [id2label[i] for i in range(len(id2label))]

    def tag(self, sentences: list[list[str]]) -> list[list[str]]:
        out: list[list[str]] = []
        for start in range(0, len(sentences), self.batch_size):
            chunk = sentences[start:start + self.batch_size]
            tagged = iter(self._tag_batch([s for s in chunk if s]) if any(chunk) else [])
            out.extend(next(tagged) if words else [] for words in chunk)
        return out

    def _tag_batch(self, chunk: list[list[str]]) -> list[list[str]]:
        encoded = self.tokenizer(chunk, is_split_into_words=True, truncation=True,
                                 max_length=self.max_len, padding=True, return_tensors="pt")
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


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _quiet_transformers() -> None:
    """transformers draws a progress bar per weight while loading - noise in a log."""
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()


def load_gate(run_id: str):
    if run_id == "6":
        return BiLSTMGate(MODELS / "stage1" / "run6_E3" / "checkpoint.pt")
    if run_id == "8":
        _quiet_transformers()
        return BertGate(MODELS / "stage1" / "run8_biomedbert" / "best")
    raise ValueError(f"no demo loader for Stage 1 run {run_id}")


def load_tagger(run_id: str):
    if run_id == "9":
        return BiLSTMTaggerRunner(MODELS / "stage2" / "run9_softmax" / "checkpoint.pt")
    if run_id == "10":
        return BiLSTMTaggerRunner(MODELS / "stage2" / "run10_crf" / "checkpoint.pt")
    if run_id == "11":
        _quiet_transformers()
        return BertTagger(MODELS / "stage2" / "run11_biomedbert" / "best")
    raise ValueError(f"no demo loader for Stage 2 run {run_id}")


def load_pipeline(key: str = DEFAULT_PAIR) -> Pipeline:
    pair = PAIRS[key]
    return Pipeline(load_gate(pair.gate_run), load_tagger(pair.tagger_run), pair)
