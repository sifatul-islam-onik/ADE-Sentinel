# ADE-Sentinel — Implementation Plan

Derived from `ADE-Sentinel_PRD.md`. This document is the build order and the
correction list. Where it disagrees with the PRD, this document wins and the
reason is stated.

**Execution model:** the local machine authors code, prepares the 2.24 MB labeled
corpus, and writes the report. **Every heavy job — PubMed fetch, embedding
training, and all neural runs — executes on Kaggle (primary) or Colab (fallback).**
No step in this plan is sized against local disk, RAM, or GPU.

Status: pre-Phase-0. Nothing built yet.

---

## 0. Verified facts (checked 2026-08-24)

| Check | Result |
|---|---|
| Local Python | 3.10.11 — enough for authoring, data prep, tests, report |
| HF Hub reachable | HTTP 200 on `ade-benchmark-corpus/ade_corpus_v2` |
| NCBI E-utilities reachable | Yes — MeSH query returns **145,177** hits for 2000–2025 |
| Per-year PubMed counts | 2015→4,640 · 2020→5,797 · 2023→5,187 · 2024→4,704 (all under the 9,999 cap) |
| `retstart > 9998` | **Hard error from NCBI** — see F1 |
| HF model IDs resolve (200) | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`, `dmis-lab/biobert-base-cased-v1.2`, `jsylee/scibert_scivocab_uncased-finetuned-ner` |
| GloVe via gensim-data | `glove-wiki-gigaword-300` **is** `glove.6B.300d`, 394 MB, one call — no zip handling (F7) |

---

## 1. Findings that change the plan

Defects and gaps in the PRD, not restatements of it. Each is verified or reasoned,
and each changes a concrete step in §2.

### F1 — `scripts/fetch_pubmed.py` as sketched in PRD §6.2 cannot work. VERIFIED.

The sketch loops `for start in range(0, 100000, 10000)` and passes `start` as
`retstart`. NCBI hard-errors above 9,998:

```
"ERROR":"Search Backend failed: Exception: 'retstart' cannot be larger than 9998.
 For PubMed, ESearch can only retrieve the first 9,999 records matching the query."
```

It also passes `retmax=10000`, itself at the cap. The script dies on its second
iteration and yields at most ~10k abstracts, not 100k.

**Fix: partition the query into date windows, each under the cap.** Verified
per-year counts are all in the 4.6k–5.8k range, so **one esearch call per
publication year** suffices and avoids History-server complexity. (`usehistory=y`
plus WebEnv is the documented alternative; year windows are simpler and trivially
resumable.)

### F2 — The PRD contradicts itself on the PubMed query, and the volume target is tighter than it looks. VERIFIED.

PRD §6.2's *table* specifies a three-clause query; PRD §6.2's *script* uses only the
third clause. They are not the same corpus:

| Query | Hits, 2000–2025 | Verdict |
|---|---|---|
| `"drug-related side effects and adverse reactions"[MeSH]` (the script) | **145,177** | Right size; year windows work |
| 3-clause OR adding `"adverse effects"[Subheading]`, `"drug therapy"[Subheading]` (the table) | **3,591,463** | 25× oversized; every year blows the cap, forcing month or week windows |

**Decision: MeSH-only query, year-windowed, range extended to 1995–2025** for
headroom. 145k PMIDs minus records with no abstract (typically 15–25%) lands around
110–125k — over the 100k floor but not by much, hence the extra five years. Record
the *actual* final count; never quote the PRD's estimate.

### F3 — Deduplicating the relation config by sentence text destroys labels.

PRD §6.1 fact 2 says "deduplicate by exact sentence text before splitting **both**
subsets." Correct for the *classification* config; wrong for the *relation* config,
which is one row **per relation** — a sentence with three drug–effect pairs appears
three times *with different spans*. Dropping duplicate rows keeps one pair and
discards the rest, permanently capping Stage 2 recall on every multi-relation
sentence.

**Fix: for Stage 2, dedup means GROUP BY sentence text and UNION the spans**, then
emit one BIO sequence per unique sentence carrying all its entities. Assert that
total spans before grouping equals total entities after BIO conversion.

### F4 — Stage 1 and Stage 2 need ONE shared split, not two independent ones.

The two configs overlap on sentence text — the PRD's own day-1 script measures the
overlap but never acts on it. Split each config independently and a sentence can land
in Stage-1 *train* and Stage-2 *test*. Run 12, the end-to-end pipeline, then evaluates
a Stage 1 model on sentences it memorised, and the headline error-propagation number
in G4 is invalid.

**Fix:** build a single global `text → split` assignment over the **union** of both
configs' unique sentences, stratified on the Stage 1 label, then project it onto each
config. `data/splits/assignment.parquet` is the source of truth. Assert zero
cross-split text overlap *across configs*, not merely within one.

### F5 — The `to_bio` converter in PRD §8.2 has two silent-failure bugs.

```python
if ts >= s and te <= e:      # containment test
```

1. **Containment, not overlap.** Any token straddling a span boundary is dropped. A
   span ending mid-token produces *fewer* tagged tokens than it should — or none.
2. **Zero-match spans fail silently.** If offsets and tokenisation disagree, the
   function returns all-`O` and training proceeds happily on empty labels. This is
   precisely the failure PRD §12 warns about, and its own reference implementation
   exhibits it.

**Fix:** use the overlap test `ts < e and te > s`, and `assert` that every input span
produced at least one non-`O` token, raising with the offending text. Log how many
spans needed boundary snapping. Unit-test both bugs explicitly.

### F6 — The negation subset must be selected programmatically, before seeing predictions.

PRD §8.3 says "hand-pick ~50–80 sentences." Hand-picking after the models exist is
cherry-picking, and a viva examiner will say so. Freeze a regex cue list first, select
by rule, save the subset to a file, and report its size and class balance. It is
*selection*, not annotation — say so explicitly in the report.

### F7 — Remote execution makes the DATA PATH the thing to design, not the disk.

With every heavy job on Kaggle/Colab, the risk shifts from "will it fit" to "can I
prove which inputs produced which number." Three consequences:

- **Frozen splits are small enough to commit to git** (~3 MB parquet for ~21k
  sentences). Do that. It permanently settles the PRD's "never regenerate at runtime"
  rule and makes every notebook reproducible from a clone.
- **Large artefacts live in a versioned private Kaggle Dataset** — PubMed corpus,
  trained W2V/FastText vectors, embedding matrices. Version number goes in `runs.csv`.
- **GloVe needs no download step:** `gensim.downloader.load("glove-wiki-gigaword-300")`
  *is* `glove.6B.300d` (394 MB, verified). Skip the 822 MB zip entirely.
  Now that size is a non-issue, `glove.840B.300d` is worth adding as an **optional
  stronger-baseline robustness check** — beating a weak baseline proves less than
  beating a strong one.
- Two requirement files, not one: `requirements-local.txt` (datasets, pandas,
  scikit-learn, pytest — no torch) and `requirements-remote.txt` (torch, transformers,
  gensim, seqeval, pytorch-crf, **pinned**, because Kaggle base images change under you).

### F8 — 2× T4 silently doubles your batch size and can contaminate the embedding ablation.

HF `Trainer` auto-wraps the model in `DataParallel` when two GPUs are visible, and
`per_device_train_batch_size` is multiplied by the device count. Two distinct hazards:

1. The PRD specifies **batch 16 at lr 2e-5**. On 2× T4, `per_device=16` trains at an
   effective 32 and your BERT numbers no longer match the stated recipe. Set
   `per_device_train_batch_size=8`, or raise it deliberately and **log the effective
   batch in `runs.csv`**.
2. **Runs 3–6 must differ *only* in the embedding matrix.** If one run lands on 1 GPU
   and another on 2, effective batch differs and the headline ablation is contaminated.
   Pin the device count for all four — either `CUDA_VISIBLE_DEVICES=0` throughout, or 2
   GPUs throughout — and record which.

Also: `DataParallel` and CRF do not mix cleanly — the CRF computes loss inside
`forward`, so `DataParallel` returns a per-GPU loss vector needing `.mean()`. Run the
BiLSTM-CRF on a **single** GPU; it is small enough that this costs nothing.

### F9 — Sequencing: start the fetch early, but it is hours, not days.

PRD schedules the fetch for Days 3–6. Realistic estimate: ~725 efetch calls of 200
records plus ~31 esearch calls; at 3 req/s without a key (10 with one) and a few
seconds per 200-abstract XML payload, expect **roughly 1–3 hours**, not days. It fits
inside one Kaggle or Colab session. Still launch it on Day 1 — it is the
longest-latency single step and it gates embeddings → runs 3–6 — but do not
schedule four days around it. Measure the real duration and record it.

Member B's chain (dedup, split, BIO conversion, baselines) is fully independent of the
fetch and runs in parallel from Day 1.

### F10 — Phase-0 hygiene the PRD defers too long.

- `results/runs.csv` and its `log_run()` helper are specified in Phase 4 but must exist
  in **Phase 0**, or the first eight runs go unlogged and get reconstructed from memory
  at report time.
- **Pin `gensim`, `numpy`, `transformers`, `torch` in the remote notebook.** Kaggle and
  Colab base images update without warning, and gensim 4.3.x against numpy 2.x is a
  known compatibility edge. A run you cannot reproduce next week is not a result.
- Log the git commit hash **and** the Kaggle Dataset version with every run. Seeds alone
  do not make a run reproducible if the code or the inputs changed underneath them.

---

## 2. Build order

Each step names its artefact and its done-when condition. **[A]** / **[B]** map to the
PRD §14 division of labour; **[both]** means do not proceed until both members agree the
output is right. **LOCAL** / **REMOTE** says where it executes.

### Phase 0 — Foundation (Days 1–2)

| # | Where | Step | Done when |
|---|---|---|---|
| 0.1 | LOCAL | `git init`; create the PRD §11 tree; `.gitignore` for `data/raw/`, `*.jsonl`, `*.bin` — but **not** `data/splits/`, which is committed (F7) | `git log` has one commit |
| 0.2 | LOCAL | `requirements-local.txt` (datasets, pandas, scikit-learn, pytest) and `requirements-remote.txt` (torch, transformers, gensim, seqeval, pytorch-crf — **pinned**) (F7, F10) | Local venv imports `datasets`; remote file version-pinned |
| 0.3 | LOCAL **[B]** | Run the PRD §6.1 day-1 sanity script verbatim | `results/dataset_stats.md` holds exact row counts, label distribution, cross-config overlap — numbers, not estimates |
| 0.4 | LOCAL | `src/utils.py`: `set_seed()`, `log_run()` appending to `results/runs.csv`, git-hash + dataset-version capture | A dummy run appears as a row in `runs.csv` (F10) |
| 0.5 | **[both]** | Stand up the remote path: private versioned Kaggle Dataset + a notebook template that clones the repo and attaches the dataset | A Kaggle notebook prints split row counts and `torch.cuda.device_count()` (F7) |
| 0.6 | REMOTE **[A]** | Launch the PubMed fetch (step 1.4) | First year-shard written (F9) |

**Exit:** both configs load, exact statistics recorded, `runs.csv` works, remote path proven, fetch running.

### Phase 1 — Data (Days 1–5, overlaps Phase 0)

| # | Where | Step | Done when |
|---|---|---|---|
| 1.1 | LOCAL **[B]** | Load both configs; dedup classification by exact text | Row counts logged before/after |
| 1.2 | LOCAL **[B]** | Group relation config by text, **union the spans** (F3) | `sum(len(spans))` equals the pre-group row count |
| 1.3 | LOCAL **[B]** | Build the **single global split** over the union of both configs, stratified on the Stage 1 label, 70/15/15, seed recorded (F4). **Commit the result to git** (F7) | Assertions pass: no text in two splits *across both configs*; class ratio within 1pp per split |
| 1.4 | REMOTE **[A]** | `scripts/fetch_pubmed.py` — **year-windowed** esearch 1995–2025, efetch in 200s, per-year shards, retry/backoff, `NCBI_API_KEY` support (F1, F2) | ≥100k records with non-empty abstracts; PMIDs deduped; **wall-clock time recorded** (F9) |
| 1.5 | LOCAL **[A]** | Corpus documentation table: exact query string, date range, fetch date, record count, drop rate | `report/data_documentation.md` — an explicit instructor requirement |

**Exit:** frozen splits committed with cross-config leakage assertions passing, ≥100k abstracts in the Kaggle Dataset, documentation table written.

### Phase 2 — Preprocessing (Days 6–9)

| # | Where | Step | Done when |
|---|---|---|---|
| 2.1 | LOCAL **[A]** | `src/tokenizer.py` — offset-returning tokenizer; keeps `5-fluorouracil`, `TNF-alpha`, `20 mg/kg`, `P<0.05`, Greek letters; protected-caps list so `ALL` ≠ `all` | Returns `(tokens, char_offsets)`; unit-tested. Offsets are required by Phase 5 |
| 2.2 | LOCAL **[A]** | Before/after evidence table vs naive whitespace/regex splitting | `results/figures/tokenizer_table.md` — vocab size, OOV rate, 10 concrete destroyed-token examples. A report figure |
| 2.3 | REMOTE **[A]** | Sentence-split the PubMed abstracts under the same conventions | `pubmed_sentences.txt` in the Kaggle Dataset; line count recorded |

**Exit:** the before/after table exists and the tokenizer emits correct char offsets.

### Phase 3 — Embeddings — *the headline* (Days 10–14)

All REMOTE — the corpus and vectors live in the Kaggle Dataset, and every artefact
produced here feeds runs 3–6. Note gensim is **CPU-only**; use a Kaggle CPU session
(no GPU quota burned) or the CPU of a GPU session.

| # | Where | Step | Done when |
|---|---|---|---|
| 3.1 | REMOTE **[A]** | Word2Vec skip-gram: 300d, window 5, min_count 5, negative 10, 5 epochs, `workers=os.cpu_count()` | `w2v.kv` saved as KeyedVectors |
| 3.2 | REMOTE **[A]** | FastText, identical hyperparameters | `ft.kv` saved |
| 3.3 | REMOTE **[A]** | `gensim.downloader.load("glove-wiki-gigaword-300")` (F7). Optionally also `glove.840B.300d` as the stronger baseline | Both load as KeyedVectors |
| 3.4 | REMOTE **[A]** | Coverage: token-weighted **and** type coverage for E1/E2/E3 over the ADE vocabulary; the 30 most frequent GloVe-OOV terms | `results/figures/coverage.md` — report §6.2 draftable from it |
| 3.5 | REMOTE **[A]** | Nearest-neighbour table: 15 probes × top-5 × 3 embeddings side by side, ≥1 probe absent from GloVe | `results/figures/neighbours.md` — report §6.3 draftable from it |
| 3.6 | **[both]** | Build embedding matrices aligned to the task vocabulary, one per E0–E3; publish as a new Kaggle Dataset version | Same row order as the shared vocab index; version number recorded |

**Exit — the project's most important gate.** Two of the three evidence types for the
main claim are complete and written up by end of Week 2. Everything afterwards
corroborates a result you already hold.

### Phase 4 — Stage 1 (Days 15–21)

| # | Where | Step | Runs |
|---|---|---|---|
| 4.1 | LOCAL **[B]** | Naive Bayes (count 1–2gram), LogReg / LinearSVM (TF-IDF) — sklearn, seconds | 1, 2 |
| 4.2 | LOCAL **[A]** | `src/models/bilstm.py`, one class, embedding matrix injected | — |
| 4.3 | REMOTE **[A]** | BiLSTM × E0/E1/E2/E3 — *only* the embedding layer varies. **Pin the device count across all four** (F8) | 3–6 |
| 4.4 | REMOTE **[B]** | Fine-tune `bert-base-uncased` and `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`; 3 epochs, lr 2e-5, **effective batch 16** (`per_device=8` on 2 GPUs — F8), max_len 128, `fp16=True` | 7, 8 |
| 4.5 | LOCAL **[A]** | Four-bar embedding chart + metrics table (macro-F1, per-class P/R, PR-AUC, confusion matrix) | — |

**Exit:** runs 1–8 in `runs.csv` with effective batch and device count logged; four-bar chart generated; best checkpoint saved.

### Phase 5 — Stage 2 (Days 22–28)

| # | Where | Step | Runs |
|---|---|---|---|
| 5.1 | LOCAL **[B]** | `src/bio_convert.py` with the **overlap** test and the zero-match assertion (F5) | — |
| 5.2 | LOCAL **[B]** | Unit tests: exact-boundary span, mid-token span, overlapping DRUG/EFFECT, multi-relation sentence, zero-match must raise. **Write these before the converter** | — |
| 5.3 | LOCAL **[both]** | Eyeball 20 converted examples, token/tag aligned → `results/bio_spotcheck.txt` | — |
| 5.4 | REMOTE **[A]** | BiLSTM softmax tagger | 9 |
| 5.5 | REMOTE **[A]** | BiLSTM-CRF (`pytorch-crf`) — **single GPU** (F8) | 10 |
| 5.6 | REMOTE **[B]** | BERT token classification | 11 |
| 5.7 | LOCAL **[both]** | `seqeval` at entity level: strict (`mode='strict', scheme=IOB2`) **and** default lenient — this delivers the PRD's "partial vs strict" stretch goal nearly free | — |
| 5.8 | LOCAL **[A]** | Count illegal tag sequences (`I-X` with no preceding `B-X`) with and without CRF | — |
| 5.9 | LOCAL | Sanity-check entity-F1 against `jsylee/scibert_scivocab_uncased-finetuned-ner` and the published 4,272-sentence / 12,264-ADE-tag statistics | — |

**Exit:** entity-F1 table complete; CRF illegal-sequence count obtained.

### Phase 6 — Integration & analysis (Days 29–33)

| # | Where | Step | Produces |
|---|---|---|---|
| 6.1 | LOCAL **[B]** | Run 12: best Stage 1 → best Stage 2 on the shared test split (valid only because of F4) | end-to-end entity-F1 |
| 6.2 | LOCAL **[B]** | Oracle vs pipeline entity-F1, **decomposed** into loss from Stage 1 false negatives (never reached Stage 2) vs false positives (Stage 2 handed an entity-free sentence) | the G4 number |
| 6.3 | LOCAL **[B]** | Run 13: frozen regex-selected negation/hedging subset (F6), all Stage 1 tiers evaluated on it vs full test | negation table |
| 6.4 | LOCAL **[B]** | Sample 30 pipeline failures; categorise (negation, hedging, multi-drug, abbreviation, boundary) | error taxonomy |

Inference-only, so local is fine; pull the checkpoints down once.

**Exit:** the two distinctive results — error propagation and the negation ladder — are in hand.

### Phase 7 — Demo (Days 34–36)

| # | Step |
|---|---|
| 7.1 | **[A]** `app/streamlit_app.py`: one text area → verdict + confidence → highlighted DRUG/EFFECT spans |
| 7.2 | **[A]** Ship the **BiLSTM-CRF** checkpoint rather than BERT — CPU inference is fast enough for the 5-second cold-start target and the demo then needs no network during the viva. HF Spaces is the alternative if you prefer a hosted link |
| 7.3 | **[A]** 5 preloaded examples — one negated, one multi-drug |
| 7.4 | **[A]** Persistent disclaimer: *research demonstration on published literature; not a clinical or diagnostic tool* |

### Phase 8 — Report & viva (Days 37–42)

| # | Step |
|---|---|
| 8.1 | Write to the PRD §13 outline; §6 (embeddings) is the longest section |
| 8.2 | Regenerate every figure at final quality from `results/` — no screenshots |
| 8.3 | README with exact reproduction commands, seed, fetch date, Kaggle Dataset version, and pinned package versions |
| 8.4 | **[both]** Rehearse on three artefacts: nearest-neighbour table, four-bar chart, negation table. Both members must explain every figure |

---

## 3. Kaggle vs Colab

**Kaggle is the primary.** 2× T4 (32 GB total), ~30 GPU-hours/week, and — the reason
that actually matters — **versioned Datasets**, which give every run a citable input
version. Colab's Drive folder has no version identity, so a result becomes hard to
reproduce once the folder changes.

**Use Colab for:** the PubMed fetch if Kaggle's session limit is inconvenient, and as
overflow when the weekly GPU quota runs out.

Operational notes:

- **Internet is off by default in Kaggle notebooks.** Enable it for the fetch and for
  HF model downloads, or attach the BERT checkpoints as a Dataset.
- **Save results out of the session as you go.** Write checkpoints and `runs.csv` rows
  to `/kaggle/working/` every epoch and download them before the session expires. Never
  leave the only copy of a result inside a session that is about to close.
- **Pin your package versions in the first notebook cell** (F10). Base images move.
- **Record `torch.cuda.device_count()` in every run row** — it is the variable that
  silently changes your effective batch size (F8).

---

## 4. Critical path

```
fetch_pubmed (~1-3h, start Day 1) ─► sentence-split ─► W2V/FastText ─► emb matrices ─► runs 3-6 ─┐
                                                                                                  ├─► run 12 ─► report
load + dedup + GLOBAL split ─► BIO convert ─► runs 9-11 ─────────────────────────────────────────┘
```

The top chain is longer, but the fetch is hours rather than days (F9). Member B's
entire chain is independent of it and should run in parallel from Day 1.

---

## 5. Immediate next three actions

1. `git init`, create the tree, `requirements-local.txt` + pinned
   `requirements-remote.txt` (F7).
2. Write the **corrected** year-windowed `scripts/fetch_pubmed.py` (F1, F2) and launch
   it in a Kaggle notebook with internet enabled (F9).
3. Run the day-1 sanity script locally and record exact numbers into
   `results/dataset_stats.md` — PRD §6.1 requires real figures, not estimates.
