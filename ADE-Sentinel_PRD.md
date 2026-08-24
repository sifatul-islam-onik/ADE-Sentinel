# ADE-Sentinel

**A Two-Stage NLP Pipeline for Detecting and Extracting Adverse Drug Events from Medical Case Reports**

Product Requirements Document & Implementation Plan
NLP Laboratory Project | CSE, KUET | 4th Year

---

## 0. One-line summary

Given a sentence from a medical case report, ADE-Sentinel first decides **whether it reports an adverse drug event**, and if it does, extracts **which drug** caused **which effect** — using a domain-trained embedding space that we show empirically outperforms general-purpose pretrained vectors on biomedical text.

---

## 1. Problem statement

Adverse drug events (ADEs) — harmful reactions caused by a medication — are a major source of hospitalisation worldwide. A large share of the evidence about them is not in structured databases but buried in free text: published case reports, discharge summaries, and pharmacovigilance narratives. Pharmacovigilance teams currently find them by reading.

Two things make this hard for NLP, and both are the actual research content of this project:

**Problem 1 — Vocabulary.** Biomedical text is written in a vocabulary that general-purpose embeddings have essentially never seen. Words like *doxorubicin*, *hepatotoxicity*, *thrombocytopenia*, and *rhabdomyolysis* are either absent from GloVe/Word2Vec-Google-News entirely, or present with meaningless neighbours. Any model built on those vectors is starting from a broken representation of the input.

**Problem 2 — Assertion, not mention.** A sentence containing a drug and a symptom does **not** mean the drug caused the symptom. Consider:

| Sentence | Contains drug + effect? | Is it an ADE report? |
|---|---|---|
| Hepatotoxicity developed after 3 weeks of methotrexate. | Yes | **Yes** |
| Methotrexate was administered; no evidence of hepatotoxicity was observed. | Yes | **No** |
| The patient's hepatotoxicity resolved when methotrexate was withdrawn. | Yes | Yes |
| Methotrexate is indicated for rheumatoid arthritis. | Yes | No |

A bag-of-words model sees nearly identical feature vectors for all four. Distinguishing them requires modelling **negation, scope, and word order** — which is exactly the capability that separates BoW/TF-IDF from LSTM from BERT. This gives the project a built-in, principled reason to climb the model ladder rather than climbing it just because the syllabus says so.

---

## 2. Goals and non-goals

### Goals

- **G1** — Build a working two-stage pipeline: sentence classification → span extraction.
- **G2** — Empirically demonstrate that embeddings trained on our own biomedical corpus outperform pretrained general-purpose embeddings, measured three independent ways (coverage, nearest-neighbour quality, downstream F1).
- **G3** — Quantify the performance progression across classical → recurrent → transformer models, and explain *why* each jump happens with reference to negation and word order.
- **G4** — Measure error propagation: how much does Stage 2 degrade when it runs on Stage 1's predictions instead of gold labels?
- **G5** — Deliver a simple input → output demo.

### Non-goals

- Not a clinical decision-support tool. **This is a methods study on published literature. It must not be presented as diagnostic.** State this explicitly in the report and on the demo interface.
- Not a real-time pharmacovigilance system.
- Not a multilingual system.
- No manual corpus annotation — all labels are pre-existing or derived programmatically.

---

## 3. Instructor requirement compliance

| Requirement | How ADE-Sentinel satisfies it |
|---|---|
| Embeddings explained concretely; own-trained shown to beat pretrained | Section 7 is a dedicated three-part experiment (coverage %, nearest-neighbour tables, downstream F1 with embedding layer as the only variable). This is the project's headline result, not a footnote. |
| Custom corpus should be moderately sized | Labeled corpus is pre-existing (~21k sentences). Unlabeled domain corpus for embedding training is 100k–200k PubMed abstracts, fetched via documented API. |
| Mainly supervised; unsupervised as bonus only | Both tasks are fully supervised. Word2Vec/FastText training is the only unsupervised component, and it is a *means* to the supervised end, not the contribution. |
| Simple input → output demo; GUI optional | Streamlit single-textbox demo. |
| Classification not mandatory | We do classification **and** span extraction, so we are not relying on classification alone. |
| Scraped data must be documented | Section 6.2 specifies source, method, query, date range, and volume, with a reproducible fetch script. |

---

## 4. Non-overlap justification

Checked against all 50 entries in the class submission sheet. Nearest neighbours and why we differ:

| Submitted project | Overlap risk | Why ADE-Sentinel is distinct |
|---|---|---|
| Fine-Grained Classification of Play Store Complaints (2107055/59) | Multi-label classification + BERT | Different domain, different task type (we add span extraction), and their embedding story is generic — ours is the core contribution |
| NLP-Based Research Paper Intelligence System (2107117/110) | Also uses academic/PubMed-adjacent text | They do recommendation + classification of *whole papers*; we do sentence-level assertion detection + relation extraction |
| Semantic Relation Extraction on Bangla-REX (2107090/88) | Also relation extraction | Different language, different relation type; theirs is closed-set relation *classification* between given entity pairs, ours is open span *extraction* plus an upstream gating classifier |
| Astha Bangla Clinical Chatbot (2107109/121) | Also medical | Theirs is retrieval/RAG over a knowledge base in Bangla; ours is supervised extraction from English literature. No methodological or data overlap |
| English Suicide Risk Detection (2107057/31) | Also English + BERT + health | Different task (risk classification from social media), no extraction component, no embedding-comparison contribution |

**Verdict:** no substantive overlap. The combination of *domain-embedding evidence + assertion detection + two-stage pipeline with error-propagation analysis* does not appear anywhere in the sheet.

---

## 5. System architecture

```
                    ┌──────────────────────────────┐
   PubMed abstracts │  UNLABELED DOMAIN CORPUS     │
   (100k–200k)      │  → Word2Vec / FastText       │
                    └──────────────┬───────────────┘
                                   │ embedding matrix
                                   ▼
  ┌─────────────┐   ┌──────────────────────────┐   ┌────────────────────────┐
  │  Raw        │   │  STAGE 1                 │   │  STAGE 2               │
  │  sentence   ├──►│  ADE Classifier          ├──►│  Drug/Effect Extractor │
  │             │   │  (binary: ADE / not ADE) │   │  (BIO sequence tagger) │
  └─────────────┘   └──────────────────────────┘   └───────────┬────────────┘
                          │ negative                            │
                          ▼                                     ▼
                    "No ADE reported"            { drug: "methotrexate",
                                                   effect: "hepatotoxicity" }
```

Stage 1 is a gate. Stage 2 only runs on sentences Stage 1 accepts. This is what creates the error-propagation measurement in G4.

---

## 6. Data

### 6.1 Labeled corpus — ADE Corpus V2

**Source:** Gurulingappa et al. (2012), *Journal of Biomedical Informatics* — a benchmark corpus built from MEDLINE case reports.
**Access:** Hugging Face Hub — `ade-benchmark-corpus/ade_corpus_v2` (the bare `ade_corpus_v2` path redirects here). Stored natively as **parquet**; no loader script, no version pinning required.
**Raw backup:** https://github.com/trunghlt/AdverseDrugReaction/tree/master/ADE-Corpus-V2

Three configurations; we use the first two:

| Config | Rows | Fields | Used for |
|---|---|---|---|
| `Ade_corpus_v2_classification` | 23,516 | `text`, `label` (0/1) | Stage 1 |
| `Ade_corpus_v2_drug_ade_relation` | 6,821 | `text`, `drug`, `effect`, char offsets | Stage 2 |
| `Ade_corpus_v2_drug_dosage_relation` | 279 | drug + dosage | Not used (too small) |

Total 30,616 rows, 2.24 MB. Published NER benchmark statistics for the positive portion: 4,272 sentences, 86,865 tokens, 12,264 ADE tags, 5,544 Drug tags — use these to sanity-check your Stage 2 numbers against the literature.

**Four data facts you must handle — do not skip these:**

1. **Only a `train` split exists.** There is no official train/dev/test partition. You must create your own stratified 70/15/15 split with a fixed seed, and report the seed.
2. **BOTH configs contain duplicate sentences.** The corpus was built one row per annotated relation, so a sentence mentioning several drug–effect pairs is repeated — the dataset preview shows single sentences appearing up to six times, in the *classification* config as well as the relation config. Deduplicate by exact sentence text before splitting **both** subsets, then assert zero overlap across splits. Skip this and the same sentence lands in train and test, your F1 looks excellent, and the entire result is void.
3. **Class imbalance.** Roughly 1 in 5 sentences is positive. Use macro-F1 and per-class F1, never raw accuracy. Report the confusion matrix.
4. **Sanity reference.** A published SciBERT fine-tuned on this corpus exists at `jsylee/scibert_scivocab_uncased-finetuned-ner`. Do **not** use it as your model, but do use it to confirm your entity-F1 lands in a plausible range rather than being silently broken by a BIO conversion bug.

```python
# Day-1 sanity script — run this before anything else
from datasets import load_dataset
from collections import Counter

REPO = "ade-benchmark-corpus/ade_corpus_v2"
cls = load_dataset(REPO, "Ade_corpus_v2_classification")["train"]
rel = load_dataset(REPO, "Ade_corpus_v2_drug_ade_relation")["train"]

print("classification rows:", len(cls), "| unique:", len(set(cls["text"])))
print("label distribution:", Counter(cls["label"]))
print("relation rows:", len(rel), "| unique:", len(set(rel["text"])))
print("overlap between configs:", len(set(cls["text"]) & set(rel["text"])))
print(cls[0])
print(rel[0])
```

Record the exact numbers this prints in your report. Do not quote the approximations in this document.

### 6.2 Unlabeled domain corpus — PubMed abstracts (documented as required)

| Field | Value |
|---|---|
| Source | PubMed / MEDLINE, NCBI E-utilities API |
| Method | `esearch` to retrieve PMIDs → `efetch` in batches of 200 to retrieve abstracts (XML) |
| Query | `("adverse effects"[Subheading] OR "drug therapy"[Subheading] OR "drug-related side effects and adverse reactions"[MeSH])` |
| Date range | To be fixed at collection time, e.g. 2000/01/01 – 2025/12/31 — **record the exact range** |
| Target volume | 100,000–200,000 abstracts (~1.5M–3M sentences) |
| Rate limit | 3 requests/sec without API key, 10/sec with a free key. Sleep accordingly. |
| Storage | One JSONL file: `{"pmid": ..., "title": ..., "abstract": ...}` |

```python
# scripts/fetch_pubmed.py  (sketch)
import time, requests, json
from xml.etree import ElementTree as ET

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
QUERY = '"drug-related side effects and adverse reactions"[MeSH]'

def get_pmids(retmax=10000, retstart=0):
    r = requests.get(f"{BASE}/esearch.fcgi", params={
        "db": "pubmed", "term": QUERY, "retmax": retmax,
        "retstart": retstart, "retmode": "json",
        "mindate": "2000/01/01", "maxdate": "2025/12/31", "datetype": "pdat"})
    return r.json()["esearchresult"]["idlist"]

def fetch_abstracts(pmids):
    r = requests.get(f"{BASE}/efetch.fcgi", params={
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
    root = ET.fromstring(r.text)
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext(".//PMID")
        title = art.findtext(".//ArticleTitle") or ""
        abstract = " ".join(t.text or "" for t in art.iter("AbstractText"))
        if abstract.strip():
            yield {"pmid": pmid, "title": title, "abstract": abstract}

if __name__ == "__main__":
    with open("data/pubmed_corpus.jsonl", "w") as f:
        for start in range(0, 100000, 10000):
            pmids = get_pmids(retstart=start)
            for i in range(0, len(pmids), 200):
                for rec in fetch_abstracts(pmids[i:i+200]):
                    f.write(json.dumps(rec) + "\n")
                time.sleep(0.4)          # respect rate limit
```

**Fallback if the API is slow or blocked:** use the `pubmed_qa` unlabeled context split or a PubMed abstract dump from the Hub. Same corpus, fewer moving parts. Decide this in Week 1, not Week 3.

---

## 7. The embedding experiment (the project's headline contribution)

This is the section the instructor explicitly asked for, so it gets its own experimental design rather than being folded into the model section.

**Independent variable:** the word representation.
**Controlled:** everything else — same tokenizer, same splits, same BiLSTM architecture, same hyperparameters, same seed.

### 7.1 Representations compared

| ID | Representation | Trained on |
|---|---|---|
| E0 | Randomly initialised, learned end-to-end | — |
| E1 | GloVe 300d (`glove.6B` / `glove.840B`) | General web + Wikipedia |
| E2 | Word2Vec Skip-gram 300d, **ours** | Our PubMed corpus |
| E3 | FastText Skip-gram 300d, **ours** | Our PubMed corpus |

FastText is included deliberately: biomedical morphology is highly regular (*-emia*, *-itis*, *-osis*, *-toxicity*), so subword information should help on rare drug names. If E3 > E2, you have a clean linguistic explanation for why.

### 7.2 Evidence type 1 — Vocabulary coverage

```python
# What fraction of ADE-corpus tokens does each embedding actually know?
def coverage(tokens, vocab):
    known = sum(1 for t in tokens if t in vocab)
    return known / len(tokens)
```

Report both **token coverage** (weighted by frequency) and **type coverage** (unique words). Type coverage for GloVe on biomedical text is typically far worse than token coverage, because the common English scaffolding is covered while the domain terms are not — and the domain terms are precisely the informative ones. Make that point explicitly.

**Deliverable:** a table, plus a list of the 30 most frequent ADE-corpus terms that are OOV in GloVe.

### 7.3 Evidence type 2 — Nearest-neighbour quality

For a fixed probe list of ~15 terms (`doxorubicin`, `hepatotoxicity`, `thrombocytopenia`, `methotrexate`, `rash`, `renal`, `carcinoma`, `induced`, `withdrawal`, ...), print the top-5 cosine neighbours under E1, E2, E3 side by side.

**Deliverable:** a three-column qualitative table. This is the figure that communicates the result in three seconds during the viva. Include at least one probe where GloVe simply has no entry.

### 7.4 Evidence type 3 — Downstream F1

Train the identical BiLSTM four times, changing only the embedding layer. Plot macro-F1 as a four-bar chart for Stage 1, and entity-F1 for Stage 2.

**Deliverable:** the bar chart plus a paragraph interpreting the gaps. Expected ordering is E0 < E1 < E2 ≤ E3 — but **report what you actually observe.** If E1 beats E2, that is a legitimate finding about corpus size, and explaining it honestly is worth more than a result that matches expectations.

---

## 8. Modelling plan

### 8.1 Stage 1 — ADE sentence classification

| Tier | Model | Features | Syllabus topic |
|---|---|---|---|
| T1 | Multinomial Naive Bayes | unigram + bigram counts | BoW, n-grams, generative baseline |
| T2 | Logistic Regression / Linear SVM | TF-IDF | TF-IDF, discriminative baseline |
| T3 | BiLSTM (+ attention) | E0–E3 embeddings | Word2Vec, RNN/LSTM |
| T4 | Fine-tuned `bert-base-uncased` | WordPiece | Transformers/BERT |
| T5 | Fine-tuned `BiomedBERT` / `BioBERT` | domain WordPiece | Transformers + the domain argument again |

T4 vs T5 is the domain-vs-general comparison replayed at the transformer level. That symmetry — domain wins at the static embedding level *and* at the contextual level — is a tidy narrative for the report.

**Metrics:** macro-F1 (primary), per-class precision/recall, PR-AUC, confusion matrix. Accuracy reported but explicitly deprecated as misleading under 1:4 imbalance.

### 8.2 Stage 2 — Drug/effect span extraction

The relation config gives character offsets, not tags. Convert to BIO:

```python
def to_bio(text, spans, tokenizer):
    """spans: [(start, end, 'DRUG'), (start, end, 'EFFECT')]"""
    tokens, offsets = tokenizer(text)          # must return char offsets
    tags = ["O"] * len(tokens)
    for s, e, label in spans:
        first = True
        for i, (ts, te) in enumerate(offsets):
            if ts >= s and te <= e:
                tags[i] = ("B-" if first else "I-") + label
                first = False
    return tokens, tags
```

Write unit tests for this. Off-by-one offset bugs are the single most common way this kind of project silently produces garbage, and they are invisible in the loss curve.

| Tier | Model |
|---|---|
| S1 | BiLSTM, softmax per token |
| S2 | BiLSTM-**CRF** |
| S3 | Fine-tuned BERT token classification |

**Why the CRF earns its place in the report:** a per-token softmax decides each tag independently, so it can emit `I-EFFECT` with no preceding `B-EFFECT` — a structurally impossible sequence. The CRF learns transition scores between tags and decodes with Viterbi, making illegal transitions vanishingly unlikely. Report the count of illegal sequences produced with and without the CRF; it is a concrete, easily-explained ablation.

**Metrics:** entity-level precision/recall/F1 via `seqeval` — **not** token accuracy. Explain in the report that token accuracy is inflated because the `O` class dominates, and that predicting 4 of 5 tokens in *chronic obstructive pulmonary disease* still yields a wrong entity.

### 8.3 The negation & hedging challenge set (differentiator)

Hand-pick ~50–80 sentences from the test set that contain negation (*no evidence of*, *ruled out*, *without*) or hedging (*may be associated with*, *possible*, *suspected*). No new annotation is needed — the gold labels already exist; you are only *selecting* a hard subset.

Report accuracy of T1–T5 on this subset versus the full test set.

**Expected story:** the BoW models collapse (they cannot represent scope), the LSTM partially recovers, BERT handles it best. This is a compact, memorable result that directly justifies the whole model ladder — and it turns "we tried five models because the syllabus listed five models" into "we tried five models and here is the linguistic phenomenon that separates them."

### 8.4 Error propagation (G4)

Run Stage 2 twice:

- **Oracle setting:** on gold positive sentences.
- **Pipeline setting:** on sentences Stage 1 *predicted* positive.

Report the entity-F1 drop. Decompose it: how much loss comes from Stage 1 false negatives (sentences never reaching Stage 2) versus false positives (Stage 2 asked to extract entities from a sentence with none)?

This is the analysis that most distinguishes a real engineering report from a tutorial writeup. Almost nobody in the class will have it.

---

## 9. Experiment matrix

| # | Stage | Model | Embedding | Purpose |
|---|---|---|---|---|
| 1 | 1 | Naive Bayes | BoW | Generative baseline |
| 2 | 1 | LogReg | TF-IDF | Discriminative baseline |
| 3 | 1 | BiLSTM | E0 random | Embedding ablation floor |
| 4 | 1 | BiLSTM | E1 GloVe | Pretrained general |
| 5 | 1 | BiLSTM | E2 our W2V | **Own embeddings** |
| 6 | 1 | BiLSTM | E3 our FastText | **Own + subword** |
| 7 | 1 | BERT-base | — | General transformer |
| 8 | 1 | BiomedBERT | — | Domain transformer |
| 9 | 2 | BiLSTM softmax | best of E1–E3 | No-CRF ablation |
| 10 | 2 | BiLSTM-CRF | best of E1–E3 | Structured prediction |
| 11 | 2 | BERT token-cls | — | Transformer tagger |
| 12 | 1+2 | Best + best | — | End-to-end pipeline |
| 13 | 1 | All of 1,2,6,7,8 | — | Negation challenge subset |

Thirteen runs. Log every run to a CSV (`results/runs.csv`) with seed, hyperparameters, and metrics as you go. Do not plan to reconstruct this at report-writing time.

---

## 10. Step-by-step work plan

### Phase 0 — Setup (Days 1–2)

- [ ] Create repo, `requirements.txt`, fix `datasets` version, verify both dataset configs load
- [ ] Run the day-1 sanity script; record exact row counts and label distribution
- [ ] Set seeds globally (`random`, `numpy`, `torch`); document the seed
- [ ] Confirm GPU availability (Colab free tier is sufficient for everything here)

**Exit criterion:** both configs load and print correct-looking examples.

### Phase 1 — Data acquisition (Days 3–6)

- [ ] Deduplicate the relation config by sentence text
- [ ] Build stratified 70/15/15 splits; save as fixed files, never re-split at runtime
- [ ] Verify no sentence appears in more than one split
- [ ] Run `fetch_pubmed.py`; target ≥100k abstracts
- [ ] Write the corpus documentation table (source, method, query, date range, volume)

**Exit criterion:** `data/` contains frozen splits plus `pubmed_corpus.jsonl`, and the documentation table is written.

### Phase 2 — Preprocessing (Days 7–9)

- [ ] Sentence-split the PubMed abstracts
- [ ] Build the domain tokenizer:
  - keep hyphenated chemical names intact (`5-fluorouracil`, `TNF-alpha`)
  - preserve dosage patterns (`20 mg/kg`)
  - handle Greek letters and parenthesised statistics (`P<0.05`)
  - **careful lowercasing** — `ALL` (acute lymphoblastic leukemia) must not become `all`. Protect all-caps tokens of length ≤5 that appear in a medical abbreviation list.
- [ ] Produce a before/after table: vocabulary size, OOV rate, and 10 concrete examples of tokens standard whitespace/regex tokenisation destroys

**Exit criterion:** the before/after table exists. It is a report figure, not throwaway work.

### Phase 3 — Embeddings (Days 10–14)

- [ ] Train Word2Vec Skip-gram (`gensim`, 300d, window 5, min_count 5, negative 10, 5 epochs)
- [ ] Train FastText with identical hyperparameters
- [ ] Download GloVe 300d
- [ ] Compute coverage table (token + type) for all three
- [ ] Generate nearest-neighbour comparison table for the probe list
- [ ] Save embedding matrices aligned to the task vocabulary

**Exit criterion:** Sections 7.2 and 7.3 of the report are fully drafted with real numbers. **The project's main claim is already evidenced at the end of Week 2.** Everything after this is corroboration.

### Phase 4 — Stage 1 models (Days 15–21)

- [ ] Runs 1–2: Naive Bayes, Logistic Regression (`scikit-learn`, an afternoon)
- [ ] Runs 3–6: BiLSTM × 4 embeddings — identical code, embedding matrix swapped
- [ ] Runs 7–8: BERT-base and BiomedBERT fine-tuning (`transformers.Trainer`, 3 epochs, lr 2e-5, batch 16)
- [ ] Produce the four-bar embedding chart and the full metrics table
- [ ] Save the best Stage 1 checkpoint

**Exit criterion:** Runs 1–8 logged in `results/runs.csv`; the embedding chart is generated.

### Phase 5 — Stage 2 models (Days 22–28)

- [ ] Implement and **unit-test** the char-offset → BIO converter
- [ ] Manually eyeball 20 converted examples before training anything
- [ ] Runs 9–11: BiLSTM softmax, BiLSTM-CRF, BERT token classification
- [ ] Evaluate with `seqeval` at entity level
- [ ] Count illegal tag sequences with and without CRF

**Exit criterion:** entity-F1 table complete; CRF ablation number obtained.

### Phase 6 — Integration & analysis (Days 29–33)

- [ ] Run 12: chain best Stage 1 → best Stage 2
- [ ] Compute oracle vs pipeline entity-F1; decompose the drop into FN-driven and FP-driven loss
- [ ] Run 13: build the negation/hedging subset, evaluate all Stage 1 tiers on it
- [ ] Error analysis: sample 30 pipeline failures, categorise them (negation, hedging, multi-drug sentences, abbreviations, boundary errors), tabulate

**Exit criterion:** error-propagation number and negation table obtained. These are the two results that make the report distinctive.

### Phase 7 — Demo (Days 34–36)

- [ ] Streamlit app, single text area
- [ ] Output: ADE / not-ADE verdict + confidence, then highlighted drug and effect spans
- [ ] Include 5 preloaded example sentences, one of them negated, to show the model handling scope
- [ ] Add a visible disclaimer: *research demonstration on published literature; not a clinical or diagnostic tool*

**Exit criterion:** app runs locally, cold start to result under 5 seconds.

### Phase 8 — Report & submission (Days 37–42)

- [ ] Write to the outline in Section 13
- [ ] Regenerate all figures at final quality
- [ ] Clean the repo, write README with reproduction instructions
- [ ] Rehearse the viva around three artefacts: the nearest-neighbour table, the four-bar chart, the negation table

---

## 11. Repository structure

```
ade-sentinel/
├── data/
│   ├── raw/                    # untouched downloads
│   ├── splits/                 # frozen train/dev/test — never regenerate
│   └── pubmed_corpus.jsonl
├── scripts/
│   ├── fetch_pubmed.py
│   ├── build_splits.py
│   └── train_embeddings.py
├── src/
│   ├── tokenizer.py            # domain tokenizer
│   ├── bio_convert.py          # + tests
│   ├── models/
│   │   ├── baselines.py        # NB, LogReg
│   │   ├── bilstm.py
│   │   ├── bilstm_crf.py
│   │   └── bert_finetune.py
│   └── evaluate.py
├── notebooks/                  # exploration only, not the source of truth
├── results/
│   ├── runs.csv                # every run logged here
│   └── figures/
├── app/
│   └── streamlit_app.py
├── report/
└── README.md
```

---

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Train/test leakage from duplicate sentences (present in **both** configs) | High if ignored | Deduplicate by sentence text before splitting; assert zero overlap between splits. This is the single most likely way to invalidate the whole project |
| PubMed fetch is slow or rate-limited | Medium | Get a free NCBI API key (10 req/s); fall back to a Hub abstract dump |
| Our embeddings do **not** beat GloVe | Medium | This is still a publishable finding. Diagnose it: corpus too small, min_count too high, or task too easy. Report honestly with the diagnosis — a well-explained negative result scores better than a suspicious positive one |
| Char-offset → BIO conversion has off-by-one bugs | High | Unit tests + manual inspection of 20 examples before training |
| Entity-level F1 much lower than expected | Medium | Check whether failures are boundary errors (partial overlap) vs complete misses; report partial-match F1 alongside strict F1 |
| BERT fine-tuning exceeds Colab session limits | Low | Save checkpoints every epoch; batch 16 with max_len 128 is small |

---

## 13. Report outline

1. Introduction — ADEs, pharmacovigilance, why free text
2. Problem formulation — the two problems from Section 1, with the four-sentence table
3. Related work — brief
4. Data
   4.1 ADE Corpus V2 (with exact statistics)
   4.2 PubMed domain corpus (full documentation table)
   4.3 Splitting and deduplication protocol
5. Preprocessing — domain tokenisation, before/after evidence
6. **Embeddings** ← the longest section
   6.1 Training setup
   6.2 Coverage analysis
   6.3 Nearest-neighbour analysis
   6.4 Downstream impact
7. Stage 1 — models and results
8. Stage 2 — models and results, CRF ablation
9. Pipeline integration and error propagation
10. Negation and hedging analysis
11. Error analysis
12. Demo
13. Limitations — English only, case-report genre only, not clinically validated
14. Conclusion

---

## 14. Division of labour (2 members)

| Member A | Member B |
|---|---|
| PubMed fetching + domain corpus | ADE corpus loading, dedup, splits |
| Domain tokenizer | BIO conversion + tests |
| Embedding training (W2V, FastText) | Baseline models (NB, LogReg) |
| BiLSTM + BiLSTM-CRF | BERT fine-tuning (both variants) |
| Coverage + neighbour analysis | Negation subset + error analysis |
| Streamlit demo | Pipeline integration + error propagation |
| Report §§4–6 | Report §§7–11 |

Both review the full report. Both must be able to explain every figure — vivas do not respect the division of labour.

---

## 15. Success criteria

**Minimum viable:** Stage 1 working across all five tiers, embedding comparison complete with all three evidence types, demo running.

**Target:** the above plus Stage 2 with CRF ablation, pipeline error propagation, and the negation analysis.

**Stretch:** partial-match vs strict entity-F1 breakdown; attention-weight visualisation showing the LSTM attending to negation cues; a per-drug-class error breakdown.

---

## Appendix A — Verified download links

*Checked August 2026.*

### Labeled dataset

| Resource | Link |
|---|---|
| Canonical dataset page | https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2 |
| Parquet files (direct download) | https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2/tree/refs%2Fconvert%2Fparquet |
| Raw corpus mirror (folder) | https://github.com/trunghlt/AdverseDrugReaction/tree/master/ADE-Corpus-V2 |
| `DRUG-AE.rel` | https://raw.githubusercontent.com/trunghlt/AdverseDrugReaction/master/ADE-Corpus-V2/DRUG-AE.rel |
| `ADE-NEG.txt` | https://raw.githubusercontent.com/trunghlt/AdverseDrugReaction/master/ADE-Corpus-V2/ADE-NEG.txt |
| `DRUG-DOSE.rel` | https://raw.githubusercontent.com/trunghlt/AdverseDrugReaction/master/ADE-Corpus-V2/DRUG-DOSE.rel |
| Source paper (cite this) | https://doi.org/10.1016/j.jbi.2012.04.008 |

**Raw file formats**, if you parse the mirror instead of the Hub version:

```
DRUG-AE.rel   (pipe-delimited)
  PMID | sentence | effect | effect_start | effect_end | drug | drug_start | drug_end

ADE-NEG.txt   (space-delimited)
  PMID NEG sentence
```

**Critical:** offsets in the raw `.rel` files are **document-level**, not sentence-level. You must rebase them before building BIO tags. The HuggingFace version has already done this — one more reason to prefer it and keep the mirror only as a backup.

### Embeddings and models

| Resource | Link / model ID |
|---|---|
| GloVe (E1 baseline, 822 MB, 50–300d) | https://nlp.stanford.edu/data/glove.6B.zip |
| BioBERT | `dmis-lab/biobert-base-cased-v1.2` |
| SciBERT | `allenai/scibert_scivocab_uncased` |
| Reference model fine-tuned on this corpus (sanity check only) | `jsylee/scibert_scivocab_uncased-finetuned-ner` |

### PubMed corpus collection

| Resource | Link |
|---|---|
| E-utilities base | https://eutils.ncbi.nlm.nih.gov/entrez/eutils/ |
| Free API key (raises 3 → 10 req/sec) | https://account.ncbi.nlm.nih.gov/settings/ |
| Usage guidelines | https://www.ncbi.nlm.nih.gov/books/NBK25497/ |

---

*ADE-Sentinel is a methods study conducted on published biomedical literature. It is not a clinical, diagnostic, or pharmacovigilance tool, and must not be used to inform medical decisions.*
