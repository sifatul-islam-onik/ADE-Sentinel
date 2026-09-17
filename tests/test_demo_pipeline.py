"""Tests for Phase 7 - the demo's pipeline logic and the app built on it.

The pipeline tests use stand-in models, so they need no torch and no checkpoint:
they pin the rules run 12 scored by - the gate's argmax decides, only accepted
sentences are tagged, entities are read strictly. The app test drives the real
Streamlit script and skips unless streamlit and the fetched checkpoints exist.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src import demo_pipeline as dp
from src.stage2_inference import words_of

REPO_ROOT = Path(__file__).resolve().parents[1]


class Gate:
    """Accepts a sentence when it contains `trigger`, with fixed probabilities."""

    def __init__(self, trigger="induced"):
        self.trigger = trigger
        self.seen = []

    def classify(self, sentences):
        self.seen.append(list(sentences))
        return [(0.9, True) if self.trigger in s else (0.2, False) for s in sentences]


class Tagger:
    """Tags a word DRUG when it ends in '-induced', EFFECT when it is 'hepatitis'."""

    max_len = 96

    def __init__(self):
        self.seen = []

    def tag(self, sentences):
        self.seen.append([list(words) for words in sentences])
        return [["B-DRUG" if w.endswith("-induced") else "B-EFFECT" if w == "hepatitis" else "O"
                 for w in words] for words in sentences]


@pytest.mark.parametrize("text", [
    "A 77-year-old man developed ankle, hand, and facial swelling after rosiglitazone.",
    "5-fluorouracil (5-FU) at 20 mg/kg caused TNF-alpha release; P<0.05.",
    "Thrombocytopenia with or without microangiopathy following quinine: \"TTP/HUS\".",
])
def test_words_are_the_stream_stage2_was_scored_on(text):
    words, offsets = dp.words_with_offsets(text)
    assert words == words_of(text)
    assert len(offsets) == len(words)
    for word, (start, end) in zip(words, offsets):
        assert text[start:end].lower() == word.lower()


def test_entities_map_back_to_characters():
    text = "Isoniazid-induced hepatitis, severe."
    words, offsets = dp.words_with_offsets(text)
    tags = ["B-DRUG", "B-EFFECT", "O"]
    assert words == ["isoniazid-induced", "hepatitis", "severe"]
    entities = dp.entities_from_tags(text, offsets, tags)
    assert [(e.label, e.text) for e in entities] == [("DRUG", "Isoniazid-induced"),
                                                     ("EFFECT", "hepatitis")]
    assert text[entities[1].start:entities[1].end] == "hepatitis"


def test_a_multiword_entity_spans_its_punctuation():
    text = "It caused ankle, hand, and facial swelling."
    words, offsets = dp.words_with_offsets(text)
    tags = ["O", "O", "B-EFFECT", "I-EFFECT", "I-EFFECT", "I-EFFECT", "I-EFFECT"]
    (entity,) = dp.entities_from_tags(text, offsets, tags)
    assert entity.text == "ankle, hand, and facial swelling"


def test_an_orphan_inside_tag_is_not_an_entity():
    """Strict IOB2, as scored: the demo must not draw what the scorer would not count."""
    text = "Isoniazid hepatitis"
    _, offsets = dp.words_with_offsets(text)
    assert dp.entities_from_tags(text, offsets, ["O", "I-EFFECT"]) == []


def test_a_rejected_sentence_never_reaches_the_tagger():
    gate, tagger = Gate(), Tagger()
    (result,) = dp.Pipeline(gate, tagger).analyse("Hepatitis was treated with steroids.")
    assert not result.is_ade and result.entities == []
    assert tagger.seen == [[]]
    assert result.confidence == pytest.approx(0.8)


def test_sentences_are_judged_separately_and_kept_in_order():
    gate, tagger = Gate(), Tagger()
    text = "Hepatitis was treated with steroids. Isoniazid-induced hepatitis followed."
    first, second = dp.Pipeline(gate, tagger).analyse(text)
    assert gate.seen == [["Hepatitis was treated with steroids.",
                          "Isoniazid-induced hepatitis followed."]]
    assert tagger.seen == [[["isoniazid-induced", "hepatitis", "followed"]]]
    assert (first.is_ade, first.entities) == (False, [])
    assert second.is_ade and second.confidence == pytest.approx(0.9)
    assert [(e.label, e.text) for e in second.entities] == [("DRUG", "Isoniazid-induced"),
                                                            ("EFFECT", "hepatitis")]


def test_empty_text_has_no_sentences():
    assert dp.Pipeline(Gate(), Tagger()).analyse("   \n ") == []


def test_a_sentence_past_the_word_limit_is_flagged():
    tagger = Tagger()
    tagger.max_len = 3
    (result,) = dp.Pipeline(Gate(), tagger).analyse("Isoniazid-induced hepatitis followed soon.")
    assert result.truncated
    (short,) = dp.Pipeline(Gate(), Tagger()).analyse("Isoniazid-induced hepatitis followed soon.")
    assert not short.truncated


def test_examples_are_the_test_sentences_they_claim_to_be():
    pd = pytest.importorskip("pandas")
    test = pd.read_parquet(REPO_ROOT / "data" / "splits" / "stage1_test.parquet")
    assert len(dp.EXAMPLES) == 5
    for example in dp.EXAMPLES:
        assert test.text.iloc[example.test_index] == example.text
        assert bool(test.label.iloc[example.test_index]) == example.ade


def test_logged_scores_read_the_latest_row_per_run(tmp_path):
    path = tmp_path / "runs.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run_id", "macro_f1", "entity_f1_strict"])
        writer.writeheader()
        writer.writerows([
            {"run_id": "6", "macro_f1": "0.5", "entity_f1_strict": ""},
            {"run_id": "6", "macro_f1": "0.8785", "entity_f1_strict": ""},
            {"run_id": "10", "macro_f1": "", "entity_f1_strict": "0.8308"},
        ])
    scores = dp.logged_scores(dp.PAIRS["bilstm"], runs_csv=path)
    assert scores == {"stage1": 0.8785, "stage2": 0.8308, "pipeline": None}


CHECKPOINTS = (REPO_ROOT / "models" / "stage1" / "run6_E3" / "checkpoint.pt",
               REPO_ROOT / "models" / "stage2" / "run10_crf" / "checkpoint.pt")


@pytest.mark.skipif(not all(p.exists() for p in CHECKPOINTS),
                    reason="the run 6 and run 10 checkpoints are fetched, not committed")
def test_app_shows_the_disclaimer_and_follows_the_examples():
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("torch")

    app = testing.AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"),
                                    default_timeout=120)
    app.run()
    assert not app.exception
    assert any("not a clinical or diagnostic tool" in w.value for w in app.warning)

    def verdicts():
        return [m.value for m in app.markdown if "gate confidence" in m.value]

    assert app.text_area[0].value == dp.EXAMPLES[0].text
    assert verdicts()[0].startswith(":red-badge[ADE]")

    negated = next(i for i, e in enumerate(dp.EXAMPLES) if e.button == "Negated")
    app.pills[0].set_value(negated).run()
    assert app.text_area[0].value == dp.EXAMPLES[negated].text
    assert verdicts()[0].startswith(":gray-badge[not ADE]")

    app.text_area[0].set_value("<b>Isoniazid</b> & rifampicin.").run()
    assert not app.exception
    body = next(m.value for m in app.markdown if "line-height: 2" in m.value)
    assert "<b>" not in body and "</b>" not in body
    assert "&lt;" in body and "&amp;" in body
