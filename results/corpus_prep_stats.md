# Corpus preparation for embedding training (step 2.3)

Produced by `scripts/prepare_corpus.py` from `data/pubmed_corpus.jsonl`,
using the same domain tokenizer the supervised stages use.

| Measure | Value |
|---|---|
| Abstracts in | 159,975 |
| Sentences out | 1,883,683 |
| Sentences per abstract | 11.8 |
| Running tokens | 40,519,109 |
| Mean sentence length | 21.5 tokens |
| Vocabulary (all) | 398,123 |
| Vocabulary at `min_count=5` | 113,103 |
| Hapax legomena | 177,202 (44.5%) |
| Output size | 270.3 MB |
| Wall clock | 0.8 min |

## Filtering rules

| Rule | Rationale |
|---|---|
| Same tokenizer as Stage 1/2 | embedding lookups must match the tokens the supervised models see, or every domain term misses |
| Punctuation-only tokens dropped | no lexical content, and they consume context-window slots that should hold the drug or the effect |
| Sentences with < 3 tokens dropped | too short to contribute a useful skip-gram context |

`min_count=5` retains 113,103 of 398,123 types (28.4%). The discarded tail is dominated by hapax legomena, which for biomedical text are largely author-specific compounds, mis-OCR'd tokens and one-off identifiers - exactly the forms Word2Vec cannot learn a useful vector for anyway. FastText (E3) is the counter-case: its subword units still reach these terms, which is the reason the PRD includes it.

## Token count sanity

40,519,109 running tokens over 159,975 abstracts is 253 tokens per abstract, consistent with the typical structured PubMed abstract. This is the corpus size that the Section 7 embedding comparison rests on: if E2/E3 fail to beat GloVe, corpus size is the first hypothesis to test, and this is the number to quote.
