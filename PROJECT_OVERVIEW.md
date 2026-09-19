# ADE-Sentinel: Comprehensive Project Overview

**ADE-Sentinel** is a two-stage Natural Language Processing (NLP) pipeline designed to identify **Adverse Drug Events (ADEs)** in medical text, such as clinical case reports and PubMed literature. It determines whether a sentence reports a drug-induced adverse effect (Stage 1), and if so, extracts the exact character and word spans identifying the **drug** and the **adverse effect** (Stage 2).

---

## 1. Executive Summary: What & Why

### The Problem
Adverse Drug Events (ADEs) are a leading cause of patient morbidity and healthcare costs. Identifying ADE mentions in vast biomedical literature manually is unsustainable. Automated extraction requires solving two distinct NLP challenges:
1. **Sentence Classification (Stage 1):** Distinguishing sentences that describe actual adverse reactions from sentences that describe treatments, disease etiology, or negated events.
2. **Named Entity Recognition (Stage 2):** Marking the exact word boundaries of both the causal agent (`DRUG`) and the clinical symptom/condition (`EFFECT`).

### The Core Research Question
Beyond developing an end-to-end extraction pipeline, ADE-Sentinel answers an empirical representation question:

> **Do word embeddings trained from scratch on specialized in-domain biomedical text outperform general-purpose English embeddings?**

To ensure a strict, fair comparison, the exact same Bidirectional LSTM (BiLSTM) network with additive attention was trained four times under identical conditions (same random seed, hyperparameters, and frozen vocabulary), varying only the starting word vectors:

| Matrix | Embedding Type | Source Corpus | Stage 1 Macro-F1 | Delta vs Floor |
| :--- | :--- | :--- | :--- | :--- |
| **E0** | Random Vectors | None (Baseline floor) | **0.744** | Baseline |
| **E1** | GloVe 300d | Wikipedia & Gigaword (General English) | **0.794** | +0.050 |
| **E2** | Word2Vec (Skip-gram) | 159,975 PubMed abstracts (Medical domain) | **0.861** | +0.117 |
| **E3** | FastText (Subword n-grams)| 159,975 PubMed abstracts (Medical domain) | **0.879** | **+0.135** |

### Headline Takeaway
The in-domain medical corpus added **+0.084 F1 over general GloVe**, whereas general GloVe only added **+0.050 F1 over random vectors**. Domain adaptation via custom PubMed training provided substantially more value than broad-domain pre-training.

---

## 2. HOW: System Architecture & Workflow

The pipeline operates in two sequential stages preceded by specialized domain tokenization:

```
[ Clinical Sentence / Medical Literature ]
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ 1. Biomedical Regex Tokenizer & Sentence Splitter   │
│    - Protects chemical names (e.g. 5-fluorouracil)  │
│    - Protects doses & units (e.g. 20 mg/kg)         │
│    - Protects uppercase acronyms (ALL, HIV, TNF)    │
└─────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ 2. Stage 1: Binary Classifier ("The Gate")          │
│    - Evaluates: Is this sentence an ADE?            │
│    - Models: BiLSTM+Attention or BiomedBERT         │
└─────────────────────────────────────────────────────┘
         │                                   │
      [No ADE]                             [ADE]
         │                                   │
         ▼                                   ▼
┌─────────────────────────┐     ┌─────────────────────────────────────────────────┐
│ Reject Sentence         │     │ 3. Stage 2: Sequence Tagger                     │
│ (Stage 2 bypassed to    │     │    - Identifies token labels:                   │
│ prevent false positives)│     │      B-DRUG, I-DRUG, B-EFFECT, I-EFFECT, O      │
└─────────────────────────┘     │    - Models: BiLSTM-CRF or BiomedBERT           │
                                └─────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                                ┌─────────────────────────────────────────────────┐
                                │ 4. Strict Entity Span Reconstruction & UI       │
                                │    - Highlights: DRUG (Blue), EFFECT (Orange)   │
                                └─────────────────────────────────────────────────┘
```

---

### Step-by-Step Technical Details

### A. Preprocessing & Domain Tokenization (`src/tokenizer.py`)
Standard tokenizers (such as `re.findall(r'\w+')` or standard whitespace/punctuation tokenizers) damage critical biomedical terms:
- `5-fluorouracil` splits into `"5"` and `"fluorouracil"`, destroying the drug entity.
- `TNF-alpha` splits into `"TNF"` and `"alpha"`.
- `20 mg/kg` breaks into four disjoint tokens.
- `ALL` (Acute Lymphoblastic Leukemia) lowercases into `"all"` (an English stopword).
- `i.v.`, `b.i.d.`, `Fig. 2`, `et al.` trigger false sentence splits.

**Solution:** ADE-Sentinel implements a custom deterministic regex tokenizer with:
- Multi-token pattern matching for numerical doses (`20 mg/kg`, `5mg`), statistical notation (`P<0.05`), and hyphenated drug compounds.
- A protected list of all-caps medical abbreviations (`ALL`, `AML`, `MS`, `HIV`, `NSAID`, `ACE`, etc.) that are preserved through `smart_lower()`.
- Lookbehind-guarded sentence boundaries preventing premature sentence fragmentation.

### B. Span-to-BIO Alignment (`src/bio.py`)
- The source dataset annotations provide character offsets (e.g., characters 12 to 24 are `DRUG`).
- `to_bio()` converts these offsets to per-token tags using overlap semantics (not strict containment, preventing dropped boundary tokens).
- Overlapping and nested entities (e.g., drug names embedded within adverse effect phrases like *"theophylline intoxication"*) are resolved using a **longest-span** policy.
- Tags follow the strict BIO standard: `O`, `B-DRUG`, `I-DRUG`, `B-EFFECT`, `I-EFFECT`.

### C. Embedding Ablation (`src/embeddings.py`, `src/vocab.py`)
- A single frozen vocabulary of **12,220 words** was extracted across the corpus.
- Four distinct embedding matrices (`E0`, `E1`, `E2`, `E3`) were constructed using this fixed vocabulary:
  - **E0 (Random):** Uniform random initialization (`[-0.05, 0.05]`).
  - **E1 (GloVe):** 300-dimensional vectors from Wikipedia 2014 + Gigaword 5.
  - **E2 (Word2Vec):** Skip-gram with negative sampling, trained on 159,975 PubMed abstracts (~270MB text).
  - **E3 (FastText):** Subword n-gram representations trained on the same PubMed corpus, capable of embedding morphological variations of rare drug names.

### D. Stage 1 Classification (`src/models.py`)
- **BiLSTM with Additive Attention:** Words pass through an embedding layer, a Bidirectional LSTM (256 hidden units per direction), followed by Bahdanau-style additive attention pooling and a classification head.
  - *Why attention?* ADE cues are localized in short phrases (e.g., *"developed acute hepatitis"*) within long clinical descriptions. Attention pools the relevant hidden states far more effectively than mean pooling.
- **Transformer Benchmarks:** Evaluated against `bert-base-uncased` (0.916 macro-F1) and `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` (**0.940 macro-F1**).

### E. Stage 2 Sequence Tagging & The CRF Advantage (`src/models.py`)
- Evaluated two token architectures using the E3 FastText embeddings:
  1. **BiLSTM + Independent Softmax:** Predicts each token's tag independently.
  2. **BiLSTM + Linear-chain CRF:** Learns label transition probabilities and decodes sequences globally using the Viterbi algorithm.
- **Impact of CRF:** A per-word softmax emits invalid transitions (e.g., jumping from `O` directly to `I-DRUG` without a preceding `B-DRUG`). The CRF layer reduced illegal transitions from **82 down to 5**, boosting strict entity-F1 from **0.812 to 0.831**.
- **BiomedBERT Token Classifier:** Serves as the high-accuracy benchmark, reaching **0.905 strict entity-F1**.

### F. Chained Pipeline & Error Propagation (`src/pipeline.py`)
- In production, Stage 2 only annotates sentences approved by Stage 1.
- If Stage 1 produces a false negative, all entities in that sentence are lost.
- Tested end-to-end performance on the complete test split:
  - **BiomedBERT Gate + BiomedBERT Tagger:** **0.833** strict entity-F1.
  - **BiLSTM Gate (E3) + BiLSTM-CRF Tagger:** **0.689** strict entity-F1.

### G. Handling Negation
- Medical records frequently state what did *not* happen (e.g., *"The patient tolerated the therapy without hepatotoxicity"*).
- The pipeline was evaluated against a dedicated challenge test set containing negation cues (`without`, `no evidence of`, `denies`).
- Because Stage 1 acts as a gate, it filters out negated statements before Stage 2 can falsely tag non-events as active ADEs.

---

## 3. WHERE: Complete Codebase Map

```
ADE-Sentinel/
├── app/                  # Web user interface & non-technical testing documentation
│   ├── streamlit_app.py  # Interactive Streamlit application
│   └── HOW_TO_TEST.md    # Non-technical manual with test cases & drug/effect tables
│
├── data/                 # Datasets and frozen splits
│   ├── splits/           # Fixed train/dev/test splits (Parquet format)
│   │   ├── stage1_train.parquet, stage1_dev.parquet, stage1_test.parquet
│   │   └── stage2_train.parquet, stage2_dev.parquet, stage2_test.parquet
│   ├── pubmed_corpus.jsonl    # Scraped PubMed abstracts (~160,000 articles)
│   └── pubmed_sentences.txt   # Sentence-split PubMed corpus for embedding training
│
├── models/               # Model weights and vector representations
│   ├── emb_matrices/     # Pre-built numpy matrices (E0, E1, E2, E3) & vocab.json
│   ├── ft.kv, w2v.kv     # FastText and Word2Vec KeyedVectors
│   ├── glove.kv          # GloVe vectors
│   ├── stage1/           # Checkpoints for Naive Bayes, SVM, BiLSTMs, BERT, BiomedBERT
│   └── stage2/           # Checkpoints for BiLSTM-Softmax, BiLSTM-CRF, BiomedBERT
│
├── notebooks/            # Documented, sequential experiments with preserved outputs
│   ├── 01_data_collection.ipynb         # Corpus loading, dedup, and PubMed scraping
│   ├── 02_preprocessing.ipynb           # Tokenizer design & BIO tag validation
│   ├── 03_embeddings.ipynb              # Training E2/E3 & embedding space analysis
│   ├── 04_stage1_classification.ipynb   # 8 classifiers & embedding ablation
│   ├── 05_stage2_tagging.ipynb          # Tagging models & CRF impact analysis
│   └── 06_pipeline_and_demo.ipynb       # Chained pipeline, error taxonomy & negation
│
├── results/              # Quantitative findings, figures, and experiment logs
│   ├── runs.csv          # Master log of all 32 experimental runs and metrics
│   ├── dataset_stats.md  # Class balances and deduplication metrics
│   ├── split_report.md   # Split verification (70/15/15 stratified)
│   ├── error_taxonomy.md # Deep-dive analysis into 30 representative failure cases
│   └── figures/          # Loss plots, ablation charts, and performance tables
│
├── src/                  # Core reusable library modules
│   ├── tokenizer.py      # Regex tokenizer, smart_lower, sentence_split
│   ├── vocab.py          # Frozen vocabulary management and token encoders
│   ├── bio.py            # Span-to-BIO converters, strict entity decoding, PRF metrics
│   ├── models.py         # PyTorch BiLSTMClassifier and BiLSTMTagger architectures
│   ├── embeddings.py     # Memory-mapped vector loading and cosine nearest neighbors
│   └── pipeline.py       # End-to-end inference engine chaining Stage 1 and Stage 2
│
├── README.md             # Project summary and quickstart
└── requirements.txt      # Python dependencies
```

---

## 4. Key Experimental Results Summary

All experimental runs were logged under [`results/runs.csv`](results/runs.csv):

### Stage 1: Sentence Classification (Metric: Macro-F1)
| Run ID | Model Architecture | Embedding | Macro-F1 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `run1` | Multinomial Naive Bayes | Count 1-2 gram | 0.768 | 0.769 | 0.767 |
| `run2` | Logistic Regression | TF-IDF 1-2 gram | 0.793 | 0.812 | 0.779 |
| `run2b`| Linear SVM | TF-IDF 1-2 gram | 0.798 | 0.803 | 0.793 |
| `run3` | BiLSTM + Additive Attention | E0 (Random, frozen) | 0.744 | 0.738 | 0.751 |
| `run4` | BiLSTM + Additive Attention | E1 (GloVe, frozen) | 0.794 | 0.796 | 0.792 |
| `run5` | BiLSTM + Additive Attention | E2 (PubMed Word2Vec, frozen)| 0.861 | 0.867 | 0.855 |
| `run6` | BiLSTM + Additive Attention | E3 (PubMed FastText, frozen)| **0.879** | 0.887 | 0.871 |
| `run7` | `bert-base-uncased` | Transformer fine-tuned | 0.916 | 0.916 | 0.916 |
| `run8` | `BiomedBERT-base-uncased` | In-domain Transformer | **0.940** | 0.941 | 0.940 |

### Stage 2: Sequence Tagging (Metric: Strict Entity-F1 on Gold ADEs)
| Run ID | Architecture | Decoding Strategy | Strict Entity-F1 | Invalid Transitions |
| :--- | :--- | :--- | :--- | :--- |
| `run9` | BiLSTM + E3 FastText | Per-token Softmax (argmax) | 0.812 | 82 |
| `run10`| BiLSTM + E3 FastText | Linear-chain CRF (Viterbi) | **0.831** | **5** |
| `run11`| `BiomedBERT` | Word-piece classification | **0.905** | N/A (subword) |

### End-to-End Chained Pipeline (Stage 1 + Stage 2 on All Sentences)
| Run ID | Stage 1 Gate | Stage 2 Tagger | Strict Entity-F1 |
| :--- | :--- | :--- | :--- |
| `run12c`| BiLSTM (Run 6) | BiLSTM-CRF (Run 10) | **0.689** |
| `run12b`| BiLSTM (Run 6) | BiomedBERT (Run 11) | **0.764** |
| `run12` | BiomedBERT (Run 8) | BiomedBERT (Run 11) | **0.833** |

---

## 5. How to Run and Test

### Setup
Ensure dependencies are installed in your virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### Launch the Streamlit Web Application
```powershell
.venv\Scripts\python -m streamlit run app\streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### Suggested Test Sentences
1. **Classic Adverse Drug Event (True Positive):**
   > *"A 55-year-old woman developed acute urticaria and labial angioedema 60 minutes after ingesting cloxacillin."*
   - **Expected Output:** Red **ADE** verdict (~98% confidence); `cloxacillin` highlighted in **blue (DRUG)**; `acute urticaria` and `labial angioedema` highlighted in **orange (EFFECT)**.
2. **Medical Treatment Without Adverse Event (True Negative):**
   > *"The patient was prescribed oral amoxicillin for a bacterial throat infection."*
   - **Expected Output:** Gray **not ADE** verdict; Stage 2 does not execute.
3. **Negated Adverse Event (Gate Filter Test):**
   > *"The patient was treated with sertraline without experiencing any incontinence episodes."*
   - **Expected Output:** Gray **not ADE** verdict (negation prevents false positive entity tagging).

