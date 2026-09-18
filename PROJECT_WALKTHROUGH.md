# ADE-Sentinel — How This Project Was Built (and How to Read It)

A plain-language guide for a teammate joining late. No prior knowledge of the repo assumed.

---

## 1. What the project does, in three sentences

Doctors write case reports. Some sentences in them describe a **drug causing a bad side effect** (an "adverse drug event", ADE), and some don't.

This project builds a **two-stage pipeline**:

- **Stage 1 — Is this sentence an ADE?** A yes/no classifier on a whole sentence.
- **Stage 2 — Which words are the drug and which are the effect?** A tagger that highlights the exact words, but only on sentences Stage 1 said "yes" to.

The **main question the project is trying to answer** is not "can we get a high score". It is: **do word embeddings trained on medical text beat generic English embeddings?** Everything else exists to support that one claim.

---

## 2. The big idea you must understand first: E0 / E1 / E2 / E3

The project trains the **exact same model four times**. The only thing that changes is the table of word vectors ("embedding matrix") it starts from:

| Name | What it is | Result (macro-F1) |
|---|---|---|
| **E0** | Random vectors — the floor, a model that knows nothing about words | 0.744 |
| **E1** | GloVe — generic English, trained on Wikipedia/news | 0.794 |
| **E2** | Word2Vec trained by this project on 100k+ PubMed medical abstracts | 0.861 |
| **E3** | FastText trained by this project on the same medical abstracts | **0.879** |

That climb from 0.744 → 0.879 **is the project's headline result.** When you see `E0`, `E1`, `E2`, `E3` anywhere in the code, this is what it means.

Everything after Phase 3 is basically corroborating evidence for this table.

---

## 3. Where the work actually ran

This matters a lot for understanding the file layout:

- **Local laptop** — writes code, prepares data, runs the small sklearn models, does analysis, writes the report.
- **Kaggle (2 free GPUs)** — does every heavy job: downloading 100k PubMed abstracts, training the embeddings, training all the neural networks.

This is why the repo has `scripts/` (real code) **and** `notebooks/` (glue that runs that code on Kaggle). The notebooks are mostly **auto-generated** — that's what the `scripts/_gen_*_nb.py` files do. You never edit the `.ipynb` by hand; you edit the generator.

---

## 4. Timeline — what he built, in order

Reconstructed from the git history. Dates are real commit dates.

### Aug 24 — Day 1: Foundation (Phase 0)

He started with **planning documents, not code**:

1. `ADE-Sentinel_PRD.md` — the original project requirements.
2. `PLAN.md` — he read the PRD, **found bugs in it**, and wrote a corrected build order. (This is the single most important file in the repo. See section 6 below.)
3. `README.md` — how to set the project up.

Then the skeleton:

4. `.gitignore`, `requirements-local.txt` (no GPU stuff), `requirements-remote.txt` (GPU stuff, version-pinned).
5. `src/utils.py` — **the reproducibility spine.** Contains `set_seed()` and `log_run()`. Every single experiment in the project appends one row to `results/runs.csv` through this file, recording which git commit produced it. Nothing is ever typed by hand.
6. `scripts/sanity_check.py` — counts rows in the public ADE dataset, writes `results/dataset_stats.md`.
7. `scripts/push_to_kaggle.py`, `scripts/kaggle_output.py`, `notebooks/kaggle_template.ipynb` — the plumbing to get code up to Kaggle and results back down.

### Aug 24 — Data (Phase 1)

8. `scripts/fetch_pubmed.py` — downloads 100k+ medical abstracts from PubMed. He had to rewrite the PRD's version because it was broken (PubMed refuses to give you more than ~10k results per query, so he splits the query **one year at a time**).
9. `scripts/build_splits.py` → produces `data/splits/*.parquet`.

**Important, and unusual:** the train/dev/test split files are **committed to git on purpose**. They are generated once, frozen forever, and never regenerated. That's what makes every later result comparable.

10. `scripts/document_corpus.py` → `report/data_documentation.md`.

### Aug 24 — Preprocessing (Phase 2)

11. `src/tokenizer.py` — a custom word-splitter for medical text. A normal tokenizer destroys `5-fluorouracil`, `TNF-alpha`, `20 mg/kg`, `P<0.05`. This one keeps them. It also returns **character offsets** (where each word started in the original string) — needed later by Stage 2 to highlight spans.
12. `tests/test_tokenizer.py` + `scripts/tokenizer_report.py` → `results/figures/tokenizer_table.md` (proof it's better than naive splitting).
13. `scripts/prepare_corpus.py` — splits the PubMed abstracts into sentences, ready for embedding training.

### Aug 24–25 — Embeddings, the headline (Phase 3)

14. `scripts/train_embeddings.py` — trains Word2Vec (E2) and FastText (E3) on the PubMed sentences. Downloads GloVe (E1).
15. `src/embedding_eval.py` + `scripts/embedding_report.py` → `results/figures/coverage.md` and `results/figures/neighbours.md`. These answer "how many of our medical words does GloVe even know?" (answer: not enough) and "what words does each embedding think are similar to *nausea*?"
16. `scripts/build_embedding_matrices.py` → aligns all four embeddings to **one shared vocabulary** (`models/emb_matrices/vocab.json`), so the four models later differ **only** in their numbers, never in their word list.
17. `scripts/document_embeddings.py` → `report/embedding_documentation.md`.

### Sep 10 — Stage 1: is this sentence an ADE? (Phase 4)

18. `scripts/train_baselines.py` — Naive Bayes, Logistic Regression, Linear SVM (runs 1, 2, 2b). Fast, runs locally.
19. `src/models/bilstm.py` + `src/models/encoding.py` + `scripts/train_bilstm.py` — the BiLSTM, run four times as E0/E1/E2/E3 (runs 3–6). **This is the headline experiment.**
20. `scripts/train_bert.py` — fine-tunes BERT and BiomedBERT (runs 7, 8). BiomedBERT wins overall at 0.940.
21. `src/metrics.py`, `scripts/stage1_report.py` → `results/figures/stage1_results.md` + `stage1_embeddings.png`.

### Sep 11 — Stage 2: which words are the drug and the effect? (Phase 5)

22. `src/bio_convert.py` — converts "characters 12–20 are a DRUG" into per-word tags like `B-DRUG I-DRUG O B-EFFECT`. Tests were written **before** the converter.
23. `src/models/bilstm_tagger.py`, `scripts/train_bilstm_tagger.py` — runs 9 (plain) and 10 (with a CRF layer on top, which stops the model producing impossible tag sequences).
24. `scripts/train_bert_tagger.py` — run 11.
25. `src/stage2_metrics.py`, `scripts/stage2_report.py` → `results/figures/stage2_results.md`.
26. `scripts/reference_tagger.py` — sanity check against a published model, to prove the numbers are in a believable range.

### Sep 11 — Gluing the two stages together + analysis (Phase 6)

This is where the two most interesting findings come from:

27. `src/stage2_inference.py` + `scripts/check_stage2_inference.py` — proves a model loaded on a laptop CPU gives the *identical* answer it gave on the Kaggle GPU. A trust gate.
28. `scripts/pipeline_eval.py` → **error propagation**: how much accuracy is lost because Stage 1 wrongly filtered out a sentence before Stage 2 ever saw it. (`results/figures/pipeline_results.md`, `pipeline_loss.png`)
29. `src/challenge_set.py` + `scripts/negation_eval.py` → **the negation ladder**: sentences saying "the drug did **not** cause nausea" are hard. This measures every model on a negation-heavy subset. (`negation_ladder.png`)
30. `scripts/error_taxonomy.py` + `src/error_sample.py` → 30 real failures, categorised. (`results/error_taxonomy.md`)
31. `scripts/document_stage1.py`, `document_stage2.py`, `document_analysis.py` → the `report/*.md` files. **These are generated from the results, not typed** — so the report can never drift from the numbers.

### Sep 17 — The demo (Phase 7)

32. `src/demo_pipeline.py` — loads both stages and runs them on one sentence, on CPU.
33. `app/streamlit_app.py` — the web demo: type a sentence → get a verdict + highlighted drug/effect words.
34. `scripts/check_demo.py` → `results/demo_check.json` — checks the demo starts fast enough and gives the same answers as the experiments.

### Not done: Phase 8

The final written report. `PLAN.md` says Phases 0–7 are built, Phase 8 is not started.

---

## 5. The repo's folder logic (one line each)

| Folder | What lives there |
|---|---|
| `src/` | Reusable logic. Imported by scripts and tests. **The real code.** |
| `scripts/` | One file per build step. Each one runs, writes an artefact, exits. |
| `notebooks/` | Auto-generated Kaggle glue. Generated by `scripts/_gen_*_nb.py`. |
| `tests/` | pytest. Mostly guards the fiddly bits (tokenizer, BIO conversion). |
| `data/splits/` | The frozen train/dev/test files. Committed on purpose. Never regenerate. |
| `models/` | The shared vocabulary + embedding matrices. |
| `results/` | **Every number the project produced.** `runs.csv` is the master log. |
| `report/` | Markdown write-ups, auto-generated from `results/`. |
| `app/` | The Streamlit demo. |

---

## 6. Suggested reading order

Follow this top to bottom. Roughly half a day to get comfortable.

### Step 1 — Understand the goal (don't open any code yet)

1. **[README.md](README.md)** — skim the top half. Gets you the two-stage idea and the local-vs-Kaggle split.
2. **[PLAN.md](PLAN.md)** — read **section 1 "Findings that change the plan" (F1–F10)** properly. This is the best file in the repo. It explains *why* the code looks the way it does, including several places where the original spec was simply wrong. Then skim **section 2 "Build order"**.
3. Skip `ADE-Sentinel_PRD.md` for now. It's the original spec, and `PLAN.md` overrides it where they disagree.

### Step 2 — See the results before the code

4. **[results/figures/stage1_results.md](results/figures/stage1_results.md)** — the headline table. The E0→E3 climb.
5. **[results/figures/stage2_results.md](results/figures/stage2_results.md)** — the tagging scores.
6. **[results/figures/pipeline_results.md](results/figures/pipeline_results.md)** and **[results/figures/negation_results.md](results/figures/negation_results.md)** — the two "interesting" findings.
7. **[results/runs.csv](results/runs.csv)** — open it in Excel. One row per experiment. Every number in the report traces back here.

### Step 3 — The three foundational code files

Read these carefully; everything else assumes them.

8. **[src/utils.py](src/utils.py)** — start at `log_run()`. Understand what gets recorded and why (`git_commit`, `device_count`, `effective_batch`).
9. **[src/tokenizer.py](src/tokenizer.py)** — start at `tokenize()`. Note it returns offsets, not just words.
10. **[src/models/encoding.py](src/models/encoding.py)** — how a sentence becomes a list of numbers the model can eat.

### Step 4 — Follow the data

11. **[scripts/build_splits.py](scripts/build_splits.py)** — how train/dev/test were frozen, and the leakage assertions.
12. **[scripts/fetch_pubmed.py](scripts/fetch_pubmed.py)** — skim only. Just note the year-window trick.
13. **[scripts/prepare_corpus.py](scripts/prepare_corpus.py)** — skim.

### Step 5 — The headline experiment

14. **[scripts/train_embeddings.py](scripts/train_embeddings.py)** — how E2 and E3 were made.
15. **[scripts/build_embedding_matrices.py](scripts/build_embedding_matrices.py)** — **read this one properly.** The "only the embedding matrix changes" guarantee lives here.
16. **[src/models/bilstm.py](src/models/bilstm.py)** — the model itself.
17. **[scripts/train_bilstm.py](scripts/train_bilstm.py)** — runs 3–6. This is the experiment.
18. **[src/embedding_eval.py](src/embedding_eval.py)** — the coverage and nearest-neighbour evidence.

### Step 6 — Stage 2

19. **[src/bio_convert.py](src/bio_convert.py)** — read `to_bio()` and `bio_to_entities()`. Read [tests/test_bio_convert.py](tests/test_bio_convert.py) alongside it; the tests explain the tricky cases better than the code does.
20. **[src/models/bilstm_tagger.py](src/models/bilstm_tagger.py)** — note the optional CRF.
21. **[src/stage2_metrics.py](src/stage2_metrics.py)** — strict vs lenient entity scoring.

### Step 7 — Putting it together

22. **[scripts/pipeline_eval.py](scripts/pipeline_eval.py)** — the error-propagation analysis.
23. **[src/challenge_set.py](src/challenge_set.py)** + **[scripts/negation_eval.py](scripts/negation_eval.py)** — the negation ladder.
24. **[src/demo_pipeline.py](src/demo_pipeline.py)** — both stages end to end on one sentence.
25. **[app/streamlit_app.py](app/streamlit_app.py)** — the UI. Short and easy; a nice place to finish.

### Step 8 — Run it yourself

26. **[app/HOW_TO_TEST.md](app/HOW_TO_TEST.md)** — follow it, launch the demo, paste in a sentence. Nothing makes the pipeline click faster.
27. `pytest tests/` — confirms your environment is right.

---

## 7. Things that will confuse you if nobody warns you

- **"Run 5", "run 12b", "run 13-4u"** — every experiment has an ID. Look it up in `results/runs.csv`. The `u` suffix means "embeddings were fine-tuned during training" instead of frozen.
- **Stage 1 and Stage 2 numbers aren't comparable.** Stage 1 is scored on whole sentences (macro-F1). Stage 2 is scored on entities (entity-F1). Different units.
- **The `notebooks/*.ipynb` files are build output.** Don't edit them. Edit `scripts/_gen_*_nb.py` and regenerate.
- **The `report/*.md` files are also generated**, by `scripts/document_*.py`. Don't hand-edit those either.
- **`data/splits/` being in git is deliberate**, not a mistake. `README.md` calls this out explicitly.
- **`requirements-local.txt` has no PyTorch on purpose.** If tests are skipping, you need `requirements-dev.txt` (the CPU torch build) — see the README's "Dev extras" section.
- **Two known open items** (from `PLAN.md`): the 30 manual error causes in step 6.4 still need human review, and the demo's cold start is 5.2s against a 5s target.
