# ADE-Sentinel

A two-stage NLP pipeline that finds **adverse drug events** in medical case reports:
sentences where a drug caused a harmful side effect, and the exact words naming the drug
and the effect.

> Research demonstration on published literature. Not a clinical, diagnostic or
> pharmacovigilance tool, and not to be used for medical decisions.

```
      "A case of toxic hepatitis caused by methotrexate."
                       |
      Stage 1 — is this an adverse drug event?          ->  yes (98%)
                       |
      Stage 2 — which words are the drug and the effect?
                       |
                EFFECT: toxic hepatitis     DRUG: methotrexate
```

---

## The question the project set out to answer

Not "can we get a high score", but:

> **Do word embeddings trained on medical text beat general-purpose English embeddings?**

The same neural network is trained four times. The only thing that changes is the table of
word vectors it starts from.

| | Starting vectors | Stage 1 macro-F1 |
|---|---|---|
| E0 | random — the floor | 0.744 |
| E1 | GloVe, general English | 0.794 |
| E2 | Word2Vec, trained here on 159,975 PubMed abstracts | 0.861 |
| E3 | FastText, same corpus | **0.879** |

**0.744 → 0.879.** The domain corpus is worth more (+0.084 over GloVe) than having
pretrained vectors at all was (+0.050 over random). That is the headline result;
everything else is supporting evidence.

---

## Run the demo

```
.venv\Scripts\python -m streamlit run app\streamlit_app.py
```

Opens at <http://localhost:8501>. Type any text; each sentence is judged on its own.
[app/HOW_TO_TEST.md](app/HOW_TO_TEST.md) is a walkthrough written for a non-technical
reader.

---

## The notebooks

Read them in order. **Every one has its outputs saved**, so you can read the whole project
without running anything. Nothing retrains: the models in `models/` are already trained,
and the cells that do run load them and score them on the frozen test split in seconds.

| | Notebook | What it covers |
|---|---|---|
| 1 | [Data collection](notebooks/01_data_collection.ipynb) | the labelled ADE corpus, and 159,975 PubMed abstracts |
| 2 | [Preprocessing](notebooks/02_preprocessing.ipynb) | the domain tokenizer, sentence splitting, character spans → BIO tags |
| 3 | [Embeddings](notebooks/03_embeddings.ipynb) | **the headline experiment** — training E2/E3 and building the four matrices |
| 4 | [Stage 1](notebooks/04_stage1_classification.ipynb) | eight classifiers, the embedding ablation, frozen vs fine-tuned |
| 5 | [Stage 2](notebooks/05_stage2_tagging.ipynb) | three taggers, and what a CRF actually buys |
| 6 | [Pipeline and demo](notebooks/06_pipeline_and_demo.ipynb) | chaining the stages, error propagation, negation, the live demo |

Training itself ran on Kaggle GPUs. The training code appears in the notebooks as
annotated code blocks rather than runnable cells — it needs a GPU and a few hours, and the
finished checkpoints are already here.

---

## Results at a glance

| | |
|---|---|
| Embedding ablation, Stage 1 macro-F1 | 0.744 (random) → **0.879** (our FastText) |
| Best Stage 1 model | BiomedBERT, **0.940** macro-F1 |
| Best Stage 2 tagger | BiomedBERT, **0.905** strict entity-F1 |
| CRF effect on malformed tag sequences | 82 → **5** |
| Both stages chained end to end | **0.833** strict entity-F1 |

Every number above is logged in [results/runs.csv](results/runs.csv), one row per
experiment, and the notebooks read it from there rather than quoting it by hand.

---

## Layout

```
notebooks/   the project, in order, with outputs saved
src/         the shared code the notebooks and the app import
app/         the Streamlit demo
data/splits/ the frozen train/dev/test files - generated once, never regenerated
models/      trained embeddings and checkpoints (not in git; large)
results/     runs.csv, figures and measured statistics
```

`src/` is six files:

| File | Contents |
|---|---|
| `tokenizer.py` | word and sentence splitting for medical text |
| `vocab.py` | words → integer ids, against the one frozen vocabulary |
| `bio.py` | character spans ↔ BIO tags, and entity-level scoring |
| `models.py` | the BiLSTM classifier and the BiLSTM(-CRF) tagger |
| `embeddings.py` | loading and inspecting the trained word vectors |
| `pipeline.py` | both stages chained, applied live to any text — what the demo runs |

---

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt ^
    --extra-index-url https://download.pytorch.org/whl/cpu
```

The CPU build of PyTorch is enough for everything here — loading checkpoints, scoring the
test split, and running the demo. Only training needed a GPU.

Two things about `data/splits/` worth knowing: it is **committed to git on purpose**, and
it is never regenerated. Those files are what make every result in `results/runs.csv`
comparable with every other.
