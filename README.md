# ADE-Sentinel

A two-stage NLP pipeline for detecting and extracting adverse drug events from
medical case reports.

> **ADE-Sentinel is a methods study conducted on published biomedical literature.
> It is not a clinical, diagnostic, or pharmacovigilance tool, and must not be
> used to inform medical decisions.**

- `ADE-Sentinel_PRD.md` — the product requirements document.
- `PLAN.md` — the build order, and the list of PRD defects it corrects. Where the
  two disagree, `PLAN.md` wins.

---

## Repository scope — read this first

Phases 0–7 are finished. **This tree has been reduced to what the demo needs to
run, plus the full record of what the experiments produced.**

| Still here | Removed from the working tree |
|---|---|
| `app/` — the Streamlit demo | `scripts/` — all 34 build-step scripts |
| `src/` — the 11 modules the demo imports | `tests/` — the pytest suite |
| `models/`, `data/splits/`, `results/`, `report/`, `notebooks/` | `src/utils.py`, `metrics.py`, `challenge_set.py`, `error_sample.py`, `embedding_eval.py` |

Nothing was lost: every removed file is in git history at commit `b520b51`.
To bring the pipeline back:

```
git checkout b520b51 -- scripts tests src/utils.py src/metrics.py src/challenge_set.py src/error_sample.py src/embedding_eval.py
```

(One line — it works as-is in cmd, PowerShell and bash.)

Until you do, treat every `scripts/...` command in this file and in
`PROJECT_WALKTHROUGH.md` as a **record of how the committed results were
produced**, not as something you can run today. The results themselves —
`results/runs.csv`, `results/figures/`, `report/` — are committed and complete.

The `notebooks/*.ipynb` files were kept deliberately, but the
`scripts/_gen_*_nb.py` generators that produced them were not. The notebooks are
build output; editing them by hand is still the wrong move. Restore the
generators first if a notebook needs to change.

---

## Running the demo

This is the one thing the trimmed tree does.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-local.txt
.venv\Scripts\python -m pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m streamlit run app\streamlit_app.py
```

A browser tab opens at <http://localhost:8501>. Type a sentence, press
**Ctrl+Enter**. See `app/HOW_TO_TEST.md` for a non-technical walkthrough.

`requirements-dev.txt` is not optional any more — it carries the CPU torch build,
`transformers`, `pytorch-crf` and `streamlit`, all of which the demo loads at
runtime. `requirements-local.txt` alone is not enough.

The demo loads run 6's BiLSTM gate and run 10's BiLSTM-CRF tagger by default
(PLAN 7.2). Together they load in well under a second, so a cold start is mostly
Streamlit and torch starting up. The sidebar can switch to the BiomedBERT pair
(runs 8 and 11), which is more accurate and takes several seconds to load. The
sidebar shows each pair's logged test scores, read live from `results/runs.csv`,
including its end-to-end score.

The five preloaded examples are held-out test sentences, each showing one
behaviour. They were picked from sentences the default pair gets right, so they
illustrate the pipeline rather than measure it.

---

## Results

| Phase | State |
|---|---|
| 0 — foundation | done |
| 1 — data, frozen splits, PubMed corpus | done — `report/data_documentation.md` |
| 2 — tokenizer and sentence splitting | done — `results/figures/tokenizer_table.md` |
| 3 — **embeddings, the headline** | done — `coverage.md`, `neighbours.md`, `embedding_matrices.md` |
| 4 — Stage 1 | done — runs 1–8 in `runs.csv`; `stage1_results.md`, `stage1_embeddings.png`, `report/stage1_documentation.md` |
| 5 — Stage 2 | done — runs 9–11 in `runs.csv`; `stage2_results.md`, `stage2_crf.png`, `report/stage2_documentation.md` |
| 6 — integration & analysis | done — runs 12, 12b, 13 in `runs.csv`; `pipeline_results.md`, `negation_results.md`, `error_taxonomy.md`, `report/analysis_documentation.md`. **The 30 manual causes in `results/error_sample.csv` were drafted by an AI agent and await the authors' review** |
| 7 — demo | built — `app/streamlit_app.py`; run 12c in `runs.csv`; `results/demo_check.json`. **Cold start misses the PRD's 5 s narrowly: median 5.2 s on the development laptop** |
| 8 — report | not started |

The headline result is the embedding ablation — the same BiLSTM trained four
times, changing nothing but the embedding matrix:

| | Embedding | Stage 1 macro-F1 |
|---|---|---|
| E0 | random vectors | 0.744 |
| E1 | GloVe (generic English) | 0.794 |
| E2 | Word2Vec on 100k+ PubMed abstracts | 0.861 |
| E3 | FastText on the same abstracts | **0.879** |

---

## Reproducibility contract

Every row in `results/runs.csv` identifies the code and the inputs that produced
it. `log_run()` captured this automatically at the time each run executed; the
function itself lived in `src/utils.py` and is now in git history with the rest of
the pipeline.

| Column | Why it exists |
|---|---|
| `git_commit`, `git_dirty` | which code ran; `git_dirty=1` means the commit is not the whole story |
| `dataset_version` | which Kaggle Dataset version supplied the corpus and embedding matrices |
| `device_count`, `effective_batch` | `Trainer` multiplies per-device batch by GPU count — see `PLAN.md` F8 |
| `seed` | fixed at 42 |

Two rules that were easy to break and expensive to discover late:

1. **`data/splits/` is committed to git on purpose.** It is small, and committing
   it is what makes "never re-split at runtime" enforceable. Do not add it to
   `.gitignore`.
2. **Runs 3–6 all saw the same `device_count`.** They differ only in the embedding
   matrix. If one had run on 1 GPU and another on 2, the effective batch size
   would have differed too and the headline ablation would be contaminated.

---

## Layout

```
app/
  streamlit_app.py    the demo - layout only, models in src/demo_pipeline.py
  HOW_TO_TEST.md      non-technical guide to driving the demo
src/
  demo_pipeline.py    gate -> tagger on any text, the models the demo loads
  tokenizer.py        offset-returning domain tokenizer
  bio_convert.py      spans -> BIO, overlap test
  stage2_metrics.py   strict/lenient/overlap F1, illegal transitions
  stage2_inference.py Stage 2 taggers on CPU, over the scored word stream
  models/
    encoding.py       text -> ids against the frozen vocabulary
    bilstm.py         the BiLSTM trained four times (runs 3-6)
    bilstm_tagger.py  the tagger, with and without a CRF (runs 9-10)
data/
  raw/            downloads and per-year PubMed shards (git-ignored)
  interim/        grouped relations (git-ignored)
  splits/         frozen train/dev/test + the run 13 cue subset — COMMITTED, never regenerated
models/
  emb_matrices/   the shared vocabulary + E0-E3 matrices
  stage1/         run checkpoints and saved test predictions
  stage2/         tagger checkpoints and saved test predictions
  pipeline/       cached Stage 2 decodes over the Stage 1 test output
notebooks/
  kaggle_template.ipynb   starting point for every remote run
  embeddings_remote.ipynb Phase 3  domain embeddings
  stage1_remote.ipynb     Phase 4  runs 3-8
  stage2_remote.ipynb     Phase 5  runs 9-11
results/
  dataset_stats.md    measured corpus statistics
  runs.csv            every run, appended as it happened
  figures/            report figures, all generated - never screenshots
report/               markdown write-ups, generated from results/
```

---

## How the results were produced

**Historical record.** Every command below ran against code that now lives in git
history at `b520b51`; restore it as shown at the top of this file before trying to
re-run anything. The artefacts these commands produced are committed.

### Execution model

The local machine authored code, prepared the 2.24 MB labeled corpus, and wrote
the report. **Every heavy job — the PubMed fetch, embedding training, and all
neural runs — executed on Kaggle (primary) or Colab (fallback).** Nothing was
sized against local disk, RAM, or GPU.

| | Local | Remote (Kaggle 2×T4) |
|---|---|---|
| Environment | `requirements-local.txt` (no torch) | `requirements-remote.txt` |
| Owns | data prep, splits, BIO conversion, sklearn baselines, analysis, report, demo | PubMed fetch, Word2Vec/FastText, runs 3–11 |

### The build steps

```
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
  negation_eval.py            step 6.3  run 13, cue vs no-cue sentences
  error_taxonomy.py           step 6.4  failure kinds, properties, 30-row sample
  document_analysis.py        report sections 9-11, from the Phase 6 results
  check_demo.py               Phase 7 gate  gates reproduce, run 12c, cold start
  _gen_*_nb.py                regenerate the four notebooks/
src/ (removed alongside them)
  utils.py            set_seed, log_run, F8 batch arithmetic
  metrics.py          Stage 1 metrics, shared by every run
  embedding_eval.py   steps 3.4-3.6  coverage, neighbours, matrices
  challenge_set.py    step 6.3  the frozen negation/hedging cue lists
  error_sample.py     step 6.4  the frozen cause codebook
```

### Phase 4 — Stage 1

Runs 1–2 were local and took seconds:

```bash
.venv\Scripts\python scripts\train_baselines.py   # runs 1, 2, 2b
.venv\Scripts\python scripts\stage1_report.py     # table + chart from runs.csv
```

Runs 3–8 needed a GPU, so they ran from `notebooks/stage1_remote.ipynb` on Kaggle
(**Accelerator: GPU T4 ×2, Internet: On**).

**Everything in Phase 4 ran on a single GPU.** Cell 1 sets
`CUDA_VISIBLE_DEVICES=0` before importing torch — it has to be the first cell you
run, because the variable is read once when the CUDA context initialises and is
silently ignored afterwards. Both training scripts pinned themselves the same way
when invoked directly, and `train_bilstm.py` refused to start if two devices were
visible. T4 ×2 was selected over P100 with the second card left idle: T4 has
tensor cores, so `fp16` actually helps runs 7–8, and Kaggle bills the session
rather than the card.

The notebook clones this repo and calls the committed training scripts — it holds
no training logic of its own, which is what makes "runs 3–6 differ only in the
embedding matrix" a checkable statement about a command line:

```bash
python scripts/train_bilstm.py --run-id 3 --embedding E0_random
python scripts/train_bilstm.py --run-id 4 --embedding E1
python scripts/train_bilstm.py --run-id 5 --embedding E2
python scripts/train_bilstm.py --run-id 6 --embedding E3
python scripts/train_bert.py   --run-id 7 --model bert-base-uncased
python scripts/train_bert.py   --run-id 8 --model biomedbert
```

### Phase 5 — Stage 2

Runs 9–11 went through `notebooks/stage2_remote.ipynb` on the same Kaggle setup
(single GPU, T4 ×2 with the pin). It installs `pytorch-crf`, which the base image
lacks, then calls:

```bash
python scripts/train_bilstm_tagger.py --run-id 9  --embedding E3
python scripts/train_bilstm_tagger.py --run-id 10 --embedding E3 --crf
python scripts/train_bert_tagger.py   --run-id 11 --model biomedbert
```

`--crf` is the only difference between runs 9 and 10. E3 and BiomedBERT are the
defaults because they won their respective Phase 4 comparisons.

**Scoring happened locally, not remotely.** `models/stage2/` came back — the
`test_predictions.json` files, the two BiLSTM checkpoints and run 11's `best/`
(not its `trainer/` directory, which holds a 0.9 GB optimizer state) — and then:

```bash
.venv\Scripts\python scripts\check_stage2_inference.py   # CPU decode == remote decode
.venv\Scripts\python scripts\reference_tagger.py         # step 5.9, several minutes on CPU
.venv\Scripts\python scripts\stage2_report.py            # steps 5.7-5.9
.venv\Scripts\python scripts\document_stage2.py          # report section 8
```

`stage2_report.py` rescores every run from the saved tag sequences — strict IOB2,
lenient and overlap (partial-match) — cross-checks each against `seqeval`, counts
illegal tag transitions for the CRF ablation, checks the BIO conversion against
the published corpus statistics, and compares against a published tagger trained
on the same corpus. The remote session is trusted for the training, not for the
numbers.

### Phase 6 — integration and analysis

All local and inference-only, against the fetched `models/stage1/` and
`models/stage2/`:

```bash
.venv\Scripts\python scripts\baseline_predictions.py   # runs 1/2/2b predictions, refit exactly
.venv\Scripts\python scripts\pipeline_eval.py          # 6.1-6.2, runs 12/12b
.venv\Scripts\python scripts\negation_eval.py          # 6.3, run 13
.venv\Scripts\python scripts\error_taxonomy.py         # 6.4
.venv\Scripts\python scripts\document_analysis.py      # report sections 9-11
```

The first `pipeline_eval.py` run decodes all 3,133 Stage 1 test sentences with
BiomedBERT on CPU (about 3 minutes on the development laptop) and caches the
result under `models/pipeline/`; later runs take seconds. `pipeline_eval.py` and
`negation_eval.py` append to `results/runs.csv` — `--no-log` regenerates their
reports without adding rows. `negation_eval.py` froze
`data/splits/stage1_test_cues.parquet` on first use and refuses to run if the cue
rule later selects anything different (PLAN F6).

6.4 has a manual half. `results/error_sample.csv` lists 30 sampled failures with
`manual_category` and `notes` columns: the rules flag what a sentence contains,
and only reading it establishes what caused the error. The categories come from
the frozen codebook that was in `src/error_sample.py`; `error_taxonomy.py` carried
filled rows over on a re-run and refused to run if one would be lost. **The CSV
and its filled rows are committed and intact** — only the code that generated them
was removed.

### Phase 7 — the demo

```bash
.venv\Scripts\python scripts\check_demo.py           # gates reproduce, run 12c, cold start
```

`check_demo.py` scored run 12c and appended it (`--no-log` to skip; an identical
latest row is not appended twice). `--bert` also checks the BiomedBERT gate on
CPU, which takes several minutes. It timed the PRD's five-second cold start the
way a person meets it: launch `streamlit run`, and stop the clock when the first
verdict appears in an already-open browser. It drove headless Edge or Chrome
(`--browser` to point at one), repeated the launch and passed on the median, and
recorded every launch in `results/demo_check.json`.

---

## Manual steps

These need an account, a browser, or a credential, so they were never scripted.
**They apply only if you restore the pipeline** — the demo needs none of them.

### 1. NCBI API key — optional, ~2 minutes, 3× faster fetch

Sign in at <https://account.ncbi.nlm.nih.gov/settings/> and create an API key. It
lifts the rate limit from 3 to 10 requests/second, taking the fetch from roughly
3 hours to roughly 1. Then:

```bash
setx NCBI_API_KEY "your_key_here"      # new terminal picks it up
```

`fetch_pubmed.py` runs without one; it just sleeps longer.

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

### 4. Run the fetch on Kaggle — step 0.6

New Kaggle notebook → **Accelerator: None** (the fetch is I/O-bound; a GPU would
burn quota for nothing) → **Internet: On**. Then run the first cells of
`kaggle_template.ipynb` to clone the repo, and:

```python
!cd /kaggle/working/ade-sentinel && python scripts/fetch_pubmed.py \
    --out /kaggle/working/pubmed_corpus.jsonl
```

Expect 1–3 hours. **Download `pubmed_corpus.jsonl` before the session closes**, or
save it straight to a Kaggle Dataset.

### 5. The seed

Fixed at 42 throughout (it was `DEFAULT_SEED` in `src/utils.py`). Changing it now
would invalidate every logged run.
