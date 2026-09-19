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
without running anything. Nothing retrains by default: the models in `models/` are already
trained, and the cells that run load them and score them on the frozen test split in
seconds.

The notebooks also **contain the training code that produced those models** — the real
loops, not a summary — behind a `TRAIN = False` switch in notebooks 3, 4 and 5. Flip it to
`True` in a Kaggle GPU session to retrain from scratch.

| | Notebook | What it covers |
|---|---|---|
| 1 | [Data collection](notebooks/01_data_collection.ipynb) | the labelled ADE corpus, and 159,975 PubMed abstracts |
| 2 | [Preprocessing](notebooks/02_preprocessing.ipynb) | the domain tokenizer, sentence splitting, character spans → BIO tags |
| 3 | [Embeddings](notebooks/03_embeddings.ipynb) | **the headline experiment** — training E2/E3 and building the four matrices |
| 4 | [Stage 1](notebooks/04_stage1_classification.ipynb) | **the embedding ablation** — one network, four sets of starting vectors |
| 5 | [Stage 2](notebooks/05_stage2_tagging.ipynb) | the BiLSTM-CRF tagger, and what the CRF actually buys |
| 6 | [Pipeline and demo](notebooks/06_pipeline_and_demo.ipynb) | chaining the stages, error propagation, negation, the live demo |

### Retraining, if anyone asks

Training ran on Kaggle GPUs and takes a few hours. To repeat it: start a GPU session,
attach this repo plus a Dataset holding `models/emb_matrices/*.npy` (57 MB, gitignored),
set `TRAIN = True`, and run all. Kaggle discards `/kaggle/working` when the session ends,
so download `models/` and `results/runs.csv` before it does.

Every run appends its own row to `results/runs.csv` through `log_run`, capturing the score,
the hyperparameters, the seed, the GPU count and the git commit — which is why no number in
this project was ever transcribed by hand.

`set_seed(42)` fixes every RNG and sets `cudnn.deterministic`, so the same code on the same
GPU should land on the same figures. A different GPU or cuDNN version can move the last
decimal place; the ablation *ordering* (E0 < E1 < E2 < E3) is a far larger effect than that
and is what the project actually claims.

---

## Results at a glance

| | |
|---|---|
| Embedding ablation, Stage 1 macro-F1 | 0.744 (random) → **0.879** (our FastText) |
| Stage 2 tagger, strict entity-F1 on gold ADE sentences | **0.831** |
| Structurally impossible tag sequences the CRF still emits | **5** in 640 sentences |
| Both stages chained end to end | **0.689** strict entity-F1 |

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

Five checkpoints are kept, all BiLSTMs: the four Stage 1 gates the ablation compares
(`run3_E0_random`, `run4_E1`, `run5_E2`, `run6_E3`) and the Stage 2 tagger
(`run10_crf`). The demo loads run 6 and run 10.

Earlier versions of this project also trained sparse baselines and BiomedBERT models at
both stages. Their scores are still in `results/runs.csv` as the historical record, but
the checkpoints and the code that trained them are not in this branch — see
[results/README.md](results/README.md).

`src/` is eight files:

| File | Contents |
|---|---|
| `tokenizer.py` | word and sentence splitting for medical text |
| `vocab.py` | words → integer ids, against the one frozen vocabulary |
| `bio.py` | character spans ↔ BIO tags, and entity-level scoring |
| `metrics.py` | Stage 1 scoring — macro-F1, per-class, PR-AUC |
| `models.py` | the BiLSTM classifier and the BiLSTM(-CRF) tagger |
| `embeddings.py` | loading, inspecting and aligning the trained word vectors |
| `utils.py` | `set_seed` and `log_run` — the reproducibility spine |
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
