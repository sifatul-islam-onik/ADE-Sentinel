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
  splits/         frozen train/dev/test — COMMITTED, never regenerated
scripts/
  sanity_check.py     step 0.3  dataset statistics
  fetch_pubmed.py     step 1.4  year-windowed PubMed fetch
  push_to_kaggle.py   step 0.5  versioned artefact upload
src/
  utils.py            set_seed, log_run
notebooks/
  kaggle_template.ipynb   step 0.5  starting point for every remote run
results/
  dataset_stats.md    measured corpus statistics
  runs.csv            every run, appended as it happens
report/
app/
```

---

## Phase 0 status

| Step | State |
|---|---|
| 0.1 repo tree, `.gitignore` | done |
| 0.2 local + remote requirements | done, local env installed and locked |
| 0.3 day-1 sanity script | done — see `results/dataset_stats.md` |
| 0.4 `set_seed` / `log_run` | done — verified via `python -m src.utils` |
| 0.5 Kaggle data path | **scripted; needs your Kaggle token and a GitHub repo** |
| 0.6 launch PubMed fetch | **needs 0.5; optionally an NCBI API key first** |

See "Manual steps" below for what 0.5 and 0.6 need from you.

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
