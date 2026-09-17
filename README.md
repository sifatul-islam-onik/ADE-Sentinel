# ADE-Sentinel

A two-stage NLP pipeline for detecting and extracting adverse drug events from
medical case reports.

> **ADE-Sentinel is a methods study conducted on published biomedical literature.
> It is not a clinical, diagnostic, or pharmacovigilance tool, and must not be
> used to inform medical decisions.**

- `ADE-Sentinel_PRD.md` — the product requirements document.
- `PLAN.md` — the build order, and the list of PRD defects it corrects. **Read this
  before writing code.** Where the two disagree, `PLAN.md` wins.

---

## Execution model

The local machine authors code, prepares the 2.24 MB labeled corpus, and writes the
report. **Every heavy job — the PubMed fetch, embedding training, and all neural
runs — executes on Kaggle (primary) or Colab (fallback).** Nothing here is sized
against local disk, RAM, or GPU.

| | Local | Remote (Kaggle 2×T4) |
|---|---|---|
| Environment | `requirements-local.txt` (no torch) | `requirements-remote.txt` |
| Owns | data prep, splits, BIO conversion, sklearn baselines, analysis, report, demo | PubMed fetch, Word2Vec/FastText, runs 3–11 |

---

## Local setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-local.txt
```

Exact resolved versions are in `requirements-local.lock.txt`.

### Dev extras: CPU torch and transformers

`requirements-local.txt` has no torch on purpose (PLAN F7) — but that also makes
`tests/test_stage1_models.py` and `tests/test_bilstm_tagger.py` skip, so a bug in
them only surfaces inside a Kaggle session. Since Phase 5 the extras are also needed
for local analysis: step 5.9 and every Phase 6 step run the saved Stage 2 taggers on
CPU, and the Phase 7 demo runs both stages on CPU. Installing the CPU build covers all
of it:

```bash
.venv\Scripts\python -m pip install -r requirements-dev.txt ^
    --extra-index-url https://download.pytorch.org/whl/cpu
```

Training still happens on Kaggle. Locally this lets `pytest tests/` run every test
instead of skipping the ones that matter most, and lets inference-only steps load the
checkpoints Kaggle produced.

## Verifying the setup

```bash
.venv\Scripts\python scripts\sanity_check.py   # writes results/dataset_stats.md
.venv\Scripts\python -m src.utils              # writes a dummy row to results/runs.csv
```

---

## Reproducibility contract

Every row in `results/runs.csv` must identify the code and the inputs that produced
it. `log_run()` captures this automatically:

| Column | Why it exists |
|---|---|
| `git_commit`, `git_dirty` | which code ran; `git_dirty=1` means the commit is not the whole story |
| `dataset_version` | which Kaggle Dataset version supplied the corpus and embedding matrices |
| `device_count`, `effective_batch` | `Trainer` multiplies per-device batch by GPU count — see `PLAN.md` F8 |
| `seed` | fixed at 42 |

Two rules that are easy to break and expensive to discover late:

1. **`data/splits/` is committed to git on purpose.** It is small, and committing it
   is what makes "never re-split at runtime" enforceable. Do not add it to
   `.gitignore`.
2. **Runs 3–6 must all see the same `device_count`.** They are supposed to differ only
   in the embedding matrix. If one runs on 1 GPU and another on 2, the effective batch
   size differs too and the headline ablation is contaminated.

---

## Layout

```
data/
  raw/            downloads and per-year PubMed shards (git-ignored)
  interim/        grouped relations (git-ignored)
  splits/         frozen train/dev/test + the run 13 cue subset — COMMITTED, never regenerated
scripts/
  sanity_check.py             step 0.3  dataset statistics
  fetch_pubmed.py             step 1.4  year-windowed PubMed fetch
  push_to_kaggle.py           step 0.5  versioned artefact upload
  build_splits.py             step 1.3  the single global split
  prepare_corpus.py           step 2.3  sentence-split the abstracts
  train_embeddings.py         steps 3.1-3.2  Word2Vec and FastText
  build_embedding_matrices.py step 3.6  E0-E3 aligned to one vocabulary
  train_baselines.py          step 4.1  runs 1-2, local, seconds
  train_bilstm.py             step 4.3  runs 3-6, one call per embedding
  train_bert.py               step 4.4  runs 7-8
  stage1_report.py            step 4.5  results table + four-bar chart
  train_bilstm_tagger.py      steps 5.4-5.5  runs 9-10, --crf is the only diff
  train_bert_tagger.py        step 5.6  run 11, word-level scoring
  check_stage2_inference.py   Phase 6 gate  CPU decode == remote decode
  reference_tagger.py         step 5.9  published SciBERT ADE tagger on our split
  stage2_report.py            steps 5.7-5.9  entity-F1, CRF ablation, sanity check
  document_stage1.py          report section 7, generated from artefacts
  document_stage2.py          report section 8, generated from artefacts
  baseline_predictions.py     Phase 6  refit runs 1/2/2b exactly, save predictions
  pipeline_eval.py            steps 6.1-6.2  runs 12/12b, oracle vs pipeline
  negation_eval.py            step 6.3  run 13, cue vs no-cue sentences, every Stage 1 tier
  error_taxonomy.py           step 6.4  failure kinds, sentence properties, 30-row sample
  document_analysis.py        report sections 9-11, assembled from the Phase 6 results
  check_demo.py               Phase 7 gate  gates reproduce, run 12c, cold start
src/
  utils.py            set_seed, log_run, F8 batch arithmetic
  tokenizer.py        step 2.1  offset-returning domain tokenizer
  bio_convert.py      step 5.1  spans -> BIO, overlap test
  embedding_eval.py   steps 3.4-3.6  coverage, neighbours, matrices
  metrics.py          Stage 1 metrics, shared by every run
  stage2_metrics.py   steps 5.7-5.8  strict/lenient/overlap F1, illegal transitions
  stage2_inference.py Stage 2 taggers on CPU, over the scored word stream
  challenge_set.py    step 6.3  the frozen negation/hedging cue lists
  error_sample.py     step 6.4  the frozen cause codebook; manual readings survive re-runs
  demo_pipeline.py    Phase 7   gate -> tagger on any text, the models the demo loads
  models/
    encoding.py       text -> ids against the frozen vocabulary
    bilstm.py         step 4.2  the BiLSTM trained four times
    bilstm_tagger.py  steps 5.4-5.5  the tagger, with and without a CRF
notebooks/
  kaggle_template.ipynb   step 0.5  starting point for every remote run
  embeddings_remote.ipynb Phase 3  domain embeddings
  stage1_remote.ipynb     Phase 4  runs 3-8
  stage2_remote.ipynb     Phase 5  runs 9-11
results/
  dataset_stats.md    measured corpus statistics
  runs.csv            every run, appended as it happens
  figures/            report figures, all generated - never screenshots
report/
app/
  streamlit_app.py    Phase 7   the demo - layout only, models in src/demo_pipeline.py
```

---

## Status

| Phase | State |
|---|---|
| 0 — foundation | done |
| 1 — data, frozen splits, PubMed corpus | done — `report/data_documentation.md` |
| 2 — tokenizer and sentence splitting | done — `results/figures/tokenizer_table.md` |
| 3 — **embeddings, the headline** | done — `coverage.md`, `neighbours.md`, `embedding_matrices.md` |
| 4 — Stage 1 | done — runs 1–8 in `runs.csv`; `stage1_results.md`, `stage1_embeddings.png`, `report/stage1_documentation.md` |
| 5 — Stage 2 | done — runs 9–11 in `runs.csv`; `stage2_results.md`, `stage2_crf.png`, `report/stage2_documentation.md` |
| 6 — integration & analysis | done — runs 12, 12b, 13 in `runs.csv`; `pipeline_results.md`, `negation_results.md`, `error_taxonomy.md`, `report/analysis_documentation.md`. **The 30 manual causes in `results/error_sample.csv` were drafted by an AI agent and await the authors' review** (`CATEGORISED_BY` in `scripts/error_taxonomy.py`) |
| 7 — demo | built — `app/streamlit_app.py`; run 12c in `runs.csv`; `results/demo_check.json`. **Cold start misses the PRD's 5 s narrowly: median 5.2 s on the development laptop** |
| 8 — report | not started |

### Phase 4 — what runs where

Runs 1–2 are local and take seconds:

```bash
.venv\Scripts\python scripts\train_baselines.py   # runs 1, 2, 2b
.venv\Scripts\python scripts\stage1_report.py     # table + chart from runs.csv
```

Runs 3–8 need a GPU, so they run from `notebooks/stage1_remote.ipynb` on Kaggle
(**Accelerator: GPU T4 ×2, Internet: On**).

**Everything in Phase 4 runs on a single GPU.** Cell 1 sets `CUDA_VISIBLE_DEVICES=0`
before importing torch — it has to be the first cell you run, because the variable is
read once when the CUDA context initialises and is silently ignored afterwards. Both
training scripts pin themselves the same way when invoked directly, and
`train_bilstm.py` refuses to start if two devices are visible. Select T4 ×2 and let the
second card idle rather than P100: T4 has tensor cores, so `fp16` actually helps runs
7–8, and Kaggle bills the session rather than the card.

The notebook clones this repo and calls
the committed training scripts — it holds no training logic of its own, which is
what makes "runs 3–6 differ only in the embedding matrix" a checkable statement
about a command line:

```bash
python scripts/train_bilstm.py --run-id 3 --embedding E0_random
python scripts/train_bilstm.py --run-id 4 --embedding E1
python scripts/train_bilstm.py --run-id 5 --embedding E2
python scripts/train_bilstm.py --run-id 6 --embedding E3
python scripts/train_bert.py   --run-id 7 --model bert-base-uncased
python scripts/train_bert.py   --run-id 8 --model biomedbert
```

Bring back `results/runs.csv` and `results/figures/stage1_*` and commit them.
See "Manual steps" below for the Kaggle and GitHub credentials this needs.

### Phase 5 — what runs where

Runs 9–11 go through `notebooks/stage2_remote.ipynb` on the same Kaggle setup
(single GPU, T4 ×2 with the pin). It installs `pytorch-crf`, which the base image
lacks, then calls:

```bash
python scripts/train_bilstm_tagger.py --run-id 9  --embedding E3
python scripts/train_bilstm_tagger.py --run-id 10 --embedding E3 --crf
python scripts/train_bert_tagger.py   --run-id 11 --model biomedbert
```

`--crf` is the only difference between runs 9 and 10. E3 and BiomedBERT are the
defaults because they won their respective Phase 4 comparisons.

**Scoring happens locally, not remotely.** Bring back `models/stage2/` — the
`test_predictions.json` files, the two BiLSTM checkpoints and run 11's `best/` (not its
`trainer/` directory, which holds a 0.9 GB optimizer state) — and run:

```bash
.venv\Scripts\python scripts\check_stage2_inference.py   # CPU decode == remote decode
.venv\Scripts\python scripts\reference_tagger.py         # step 5.9, several minutes on CPU
.venv\Scripts\python scripts\stage2_report.py            # steps 5.7-5.9
.venv\Scripts\python scripts\document_stage2.py          # report section 8
```

`stage2_report.py` rescores every run from the saved tag sequences — strict IOB2,
lenient and overlap (partial-match) — cross-checks each against `seqeval`, counts
illegal tag transitions for the CRF ablation, checks the BIO conversion against the
published corpus statistics, and compares against a published tagger trained on the
same corpus. The remote session is trusted for the training, not for the numbers.

### Phase 6 — what runs where

All local and inference-only: it needs the dev extras above and the fetched
`models/stage1/` and `models/stage2/`.

```bash
.venv\Scripts\python scripts\baseline_predictions.py   # runs 1/2/2b predictions, refit exactly
.venv\Scripts\python scripts\pipeline_eval.py          # 6.1-6.2, runs 12/12b
.venv\Scripts\python scripts\negation_eval.py          # 6.3, run 13
.venv\Scripts\python scripts\error_taxonomy.py         # 6.4
.venv\Scripts\python scripts\document_analysis.py      # report sections 9-11
```

The first `pipeline_eval.py` run decodes all 3,133 Stage 1 test sentences with
BiomedBERT on CPU (about 3 minutes here) and caches the result under
`models/pipeline/`; later runs take seconds. `pipeline_eval.py` and `negation_eval.py`
append to `results/runs.csv` — pass `--no-log` to regenerate their reports without adding
rows. `negation_eval.py` freezes `data/splits/stage1_test_cues.parquet` on first use and
refuses to run if the cue rule later selects anything different (PLAN F6).

6.4 has a manual half. `results/error_sample.csv` lists 30 sampled failures with
`manual_category` and `notes` columns: the rules flag what a sentence contains, and only
reading it establishes what caused the error. The categories come from the frozen codebook
in `src/error_sample.py`; `error_taxonomy.py` carries filled rows over on a re-run and
refuses to run if one would be lost.

### Phase 7 — what runs where

Local, CPU, dev extras installed, `models/stage1/` and `models/stage2/` fetched:

```bash
.venv\Scripts\python scripts\check_demo.py           # gates reproduce, run 12c, cold start
.venv\Scripts\python -m streamlit run app\streamlit_app.py
```

The demo loads run 6's BiLSTM gate and run 10's BiLSTM-CRF tagger by default (PLAN
7.2). Together they load in well under a second, so a cold start is mostly Streamlit
and torch starting up. The sidebar can switch to the BiomedBERT pair (runs 8 and 11),
which is more accurate and takes several seconds to load. The sidebar shows each
pair's logged test scores, including its end-to-end score. For the default pair that
is run 12c, which `check_demo.py` scores and appends (`--no-log` to skip; an identical
latest row is not appended twice). `--bert` also checks the BiomedBERT gate on CPU,
which takes several minutes.

`check_demo.py` times the PRD's five-second cold start the way a person meets it:
launch `streamlit run`, and stop the clock when the first verdict appears in an
already-open browser. It drives headless Edge or Chrome (`--browser` to point at
one), repeats the launch and passes on the median, and records every launch in
`results/demo_check.json`.

The five preloaded examples are held-out test sentences, each showing one behaviour.
They were picked from sentences the default pair gets right, so they illustrate the
pipeline rather than measure it.

---

## Manual steps

These need an account, a browser, or a credential, so they are yours rather than
scripted.

### 1. NCBI API key — optional, ~2 minutes, 3× faster fetch

Sign in at <https://account.ncbi.nlm.nih.gov/settings/> and create an API key. It
lifts the rate limit from 3 to 10 requests/second, taking the fetch from roughly
3 hours to roughly 1. Then:

```bash
setx NCBI_API_KEY "your_key_here"      # new terminal picks it up
```

The script runs without one; it just sleeps longer.

### 2. GitHub repo — required for the Kaggle notebook to clone the code

Create an empty repo (private is fine), then:

```bash
git remote add origin https://github.com/<you>/ade-sentinel.git
git push -u origin main
```

Put the URL into cell 3 of `notebooks/kaggle_template.ipynb`. For a private repo,
add a token through Kaggle **Add-ons → Secrets** rather than pasting it in.

### 3. Kaggle API token — required for `push_to_kaggle.py`

Kaggle → **Settings → API → Create New Token**, then move the downloaded
`kaggle.json` to `%USERPROFILE%\.kaggle\kaggle.json` and:

```bash
.venv\Scripts\python -m pip install kaggle
```

The artefacts dataset has nothing to upload until Phase 1.4 produces the corpus, so
run `push_to_kaggle.py --init` then.

### 4. Run the fetch on Kaggle — step 0.6

New Kaggle notebook → **Accelerator: None** (the fetch is I/O-bound; a GPU would
burn quota for nothing) → **Internet: On**. Then run the first cells of
`kaggle_template.ipynb` to clone the repo, and:

```python
!cd /kaggle/working/ade-sentinel && python scripts/fetch_pubmed.py \
    --out /kaggle/working/pubmed_corpus.jsonl
```

Expect 1–3 hours. **Download `pubmed_corpus.jsonl` before the session closes**, or
save it straight to a Kaggle Dataset. Record the printed wall-clock time and record
count — they belong in `report/data_documentation.md` (step 1.5).

### 5. Confirm the seed

Fixed at 42 throughout (`src/utils.py:DEFAULT_SEED`). Change it now if you want a
different one; changing it after Phase 4 invalidates every logged run.
