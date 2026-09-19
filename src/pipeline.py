"""The two-stage pipeline, applied live to any text. This is what the demo runs.

    from src.pipeline import load_pipeline
    pipeline = load_pipeline()
    for s in pipeline.analyse("Hepatotoxicity developed after isoniazid."):
        print(s.is_ade, s.confidence, s.entities)

Two saved BiLSTM checkpoints, both on CPU:

    Stage 1 gate   run 6   BiLSTM + attention on the E3 (FastText) matrix
    Stage 2 tagger run 10  BiLSTM + CRF on the same matrix

The rules below are the ones the reported numbers were measured under:

* the gate's decision is the argmax (not a 0.5 threshold), and its confidence is
  the softmax probability of the class it chose;
* Stage 2 tags only the sentences the gate accepts;
* entities are read strictly, so what the demo highlights is what the scorer
  would have counted.

Text goes through the project tokenizer and the punctuation filter, with
character offsets kept, so entities can be painted back onto the original string.
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
GATE_MAX_LEN = 96
TAGGER_MAX_LEN = 96

# Which logged runs these checkpoints are, for reading their scores back out of
# results/runs.csv.
GATE_RUN = "6"          # BiLSTM + attention, E3 embeddings
TAGGER_RUN = "10"       # BiLSTM + CRF, E3 embeddings
PIPELINE_RUN = "12c"    # the two of them chained, scored end to end
MODEL_LABEL = "BiLSTM gate + BiLSTM-CRF tagger"


@dataclass(frozen=True)
class Example:
    """A preloaded sentence. All five are held-out Stage 1 test sentences.

    Chosen to show one behaviour each, from sentences the model handles
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


def logged_scores(runs_csv: Path | None = None) -> dict:
    """The model's test scores from `results/runs.csv`, None where not logged.

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

    return {"stage1": number(GATE_RUN, "macro_f1"),
            "stage2": number(TAGGER_RUN, "entity_f1_strict"),
            "pipeline": number(PIPELINE_RUN, "entity_f1_strict")}


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

    def __init__(self, gate, tagger):
        self.gate, self.tagger = gate, tagger

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
            # Both models count words, so one length check covers them.
            truncated = len(sentence_words) > max(self.gate.max_len, self.tagger.max_len)
            results.append(SentenceResult(sentence, float(p_ade), bool(is_ade), entities,
                                          truncated))
        return results


def _decisions(torch, logits) -> list[tuple[float, bool]]:
    """(P(ADE), argmax is ADE) per row - the argmax, not a 0.5 threshold."""
    p_ade = torch.softmax(logits.float(), dim=1)[:, 1].tolist()
    chosen = logits.argmax(dim=1).tolist()
    return [(p, c == 1) for p, c in zip(p_ade, chosen)]


# --------------------------------------------------------------------------
# Stage 1: the gate
# --------------------------------------------------------------------------

class BiLSTMGate:
    """A saved run 3-6 checkpoint, rebuilt against its frozen embedding matrix."""

    def __init__(self, checkpoint, matrices=MATRICES, *, max_len: int = GATE_MAX_LEN,
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


# --------------------------------------------------------------------------
# Stage 2: the tagger
# --------------------------------------------------------------------------

class BiLSTMTaggerRunner:
    """A saved run 10 checkpoint, rebuilt against the frozen embedding matrix."""

    def __init__(self, checkpoint, matrices=MATRICES, *, max_len: int = TAGGER_MAX_LEN,
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


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_gate(run_id: str = GATE_RUN):
    """One of the four ablation checkpoints. Run 6 (E3) is the one the demo uses."""
    folder = {"3": "run3_E0_random", "4": "run4_E1",
              "5": "run5_E2", "6": "run6_E3"}.get(run_id)
    if folder is None:
        raise ValueError(f"no checkpoint for Stage 1 run {run_id} - this project "
                         f"ships runs 3, 4, 5 and 6")
    return BiLSTMGate(MODELS / "stage1" / folder / "checkpoint.pt")


def load_tagger(run_id: str = TAGGER_RUN):
    if run_id != "10":
        raise ValueError(f"no checkpoint for Stage 2 run {run_id} - this project "
                         f"ships run 10, the BiLSTM-CRF tagger")
    return BiLSTMTaggerRunner(MODELS / "stage2" / "run10_crf" / "checkpoint.pt")


def load_pipeline() -> Pipeline:
    """The demo's pipeline: run 6 gate, run 10 tagger, both on CPU."""
    return Pipeline(load_gate(), load_tagger())
