"""Steps 3.4-3.5 - evidence that domain embeddings beat general-purpose ones.

This is the project's headline contribution (PRD section 7), and it is argued
three independent ways. Two of them live here:

  3.4 Vocabulary coverage - what fraction of the task's tokens does each
      embedding actually know?
  3.5 Nearest neighbours  - for a fixed probe list, are the neighbours
      clinically sensible or noise?

The third (downstream F1) needs the BiLSTM and therefore a GPU.

The coverage distinction that matters: report BOTH token coverage (frequency
weighted) and type coverage (unique words). General-purpose vectors score well
on tokens and badly on types, because the common English scaffolding is covered
while the domain terms are not - and the domain terms are the informative ones.
A single blended number hides exactly the effect this section exists to show.
"""

from __future__ import annotations

from collections import Counter

# PRD section 7.3 asks for ~15 probes. Chosen to span four families, so the
# table shows more than one kind of failure:
#   drugs           - rare, morphologically distinctive proper nouns
#   toxicity terms  - regular biomedical morphology (-toxicity, -emia, -osis)
#   clinical signs  - ordinary English words used in a specialised sense
#   relation cues   - the words that carry causation, which Stage 1 depends on
PROBE_TERMS: list[str] = [
    # drugs
    "doxorubicin", "methotrexate", "cisplatin", "vancomycin",
    # toxicity / adverse-event vocabulary
    "hepatotoxicity", "thrombocytopenia", "rhabdomyolysis", "nephrotoxicity",
    "agranulocytosis",
    # clinical signs with an everyday surface form
    "rash", "renal", "carcinoma", "lesions",
    # causation and temporality cues
    "induced", "withdrawal", "discontinued",
]


def lookup_key(term: str, vocab, case_fallback: bool = True) -> str | None:
    """Resolve `term` against `vocab`, optionally falling back to lowercase.

    WHY THE FALLBACK IS NOT OPTIONAL IN PRACTICE. Our tokenizer deliberately
    protects medical abbreviations from lowercasing, so `HIV`, `CT`, `AML` and
    `RA` stay uppercase. `glove.6B` is an UNCASED release: its vocabulary holds
    `hiv`, `ct`, `aml`. Comparing the two without a case-folded retry counts
    those as GloVe failures when they are really artefacts of OUR preprocessing,
    inflating GloVe's OOV rate and flattering the domain vectors.

    The headline claim of PRD section 7 has to survive a hostile reading, so the
    comparison is made against the strongest fair version of the baseline. Both
    numbers are reported; the difference between them is itself informative.
    """
    if term in vocab:
        return term
    if case_fallback:
        lowered = term.lower()
        if lowered != term and lowered in vocab:
            return lowered
    return None


def coverage(
    counts: Counter[str],
    vocab,
    case_fallback: bool = True,
) -> dict[str, float | int]:
    """Token- and type-level coverage of `counts` by an embedding vocabulary.

    Args:
        counts: token -> frequency over the task corpus.
        vocab: anything supporting `in` (gensim KeyedVectors, set, dict).
        case_fallback: retry a miss in lowercase; see `lookup_key`.

    Returns a dict with both rates plus the raw numerators, because the report
    needs to state the counts, not only the percentages.
    """
    resolved = {term: lookup_key(term, vocab, case_fallback) for term in counts}
    types_known = sum(1 for key in resolved.values() if key is not None)
    tokens_known = sum(freq for term, freq in counts.items()
                       if resolved[term] is not None)
    total_types = len(counts)
    total_tokens = sum(counts.values())

    return {
        "types_known": types_known,
        "types_total": total_types,
        "type_coverage": types_known / total_types if total_types else 0.0,
        "tokens_known": tokens_known,
        "tokens_total": total_tokens,
        "token_coverage": tokens_known / total_tokens if total_tokens else 0.0,
    }


def oov_terms(
    counts: Counter[str],
    vocab,
    limit: int = 30,
    case_fallback: bool = True,
) -> list[tuple[str, int]]:
    """The most frequent task terms this embedding has never seen.

    PRD 7.2 asks for the top 30 GloVe-OOV terms specifically. Frequency order
    matters: a rare OOV term costs little, whereas a frequent one means the model
    is blind to a word it meets constantly.
    """
    return [(term, freq) for term, freq in counts.most_common()
            if lookup_key(term, vocab, case_fallback) is None][:limit]


def neighbours(kv, term: str, topn: int = 5,
               case_fallback: bool = True) -> list[tuple[str, float]] | None:
    """Top-n cosine neighbours, or None when the term is absent.

    None is a result, not an error: PRD 7.3 explicitly asks for at least one
    probe that GloVe has no entry for, and "no entry" is the strongest possible
    statement about coverage.
    """
    key = lookup_key(term, kv, case_fallback)
    if key is None:
        return None
    return [(w, float(s)) for w, s in kv.most_similar(key, topn=topn)]


def neighbour_table(
    models: dict[str, object],
    probes: list[str] | None = None,
    topn: int = 5,
) -> str:
    """Side-by-side markdown table of neighbours across embeddings (PRD 7.3).

    This is the figure that communicates the result in three seconds during a
    viva, so it is generated rather than transcribed.
    """
    probes = probes or PROBE_TERMS
    names = list(models)

    lines = ["| Probe | " + " | ".join(names) + " |",
             "|---" * (len(names) + 1) + "|"]

    for probe in probes:
        cells = []
        for name in names:
            got = neighbours(models[name], probe, topn)
            if got is None:
                cells.append("**not in vocabulary**")
            else:
                cells.append("<br>".join(f"{w} ({s:.2f})" for w, s in got))
        lines.append(f"| `{probe}` | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def coverage_table(models: dict[str, object], counts: Counter[str]) -> str:
    """Markdown coverage table across embeddings (PRD 7.2).

    Reports the case-folded numbers as the headline and the exact-case numbers
    alongside. The gap between the two is the share of vocabulary an uncased
    baseline only *appears* to be missing because of our abbreviation
    protection - a confound of our own making, not a property of the baseline.
    See `lookup_key`.
    """
    lines = [
        "| Embedding | Type coverage | Token coverage | Types known | Types OOV "
        "| Type coverage, exact case |",
        "|---|---|---|---|---|---|",
    ]
    for name, kv in models.items():
        c = coverage(counts, kv, case_fallback=True)
        strict = coverage(counts, kv, case_fallback=False)
        lines.append(
            f"| {name} | {c['type_coverage'] * 100:.2f}% | "
            f"{c['token_coverage'] * 100:.2f}% | "
            f"{c['types_known']:,} | {c['types_total'] - c['types_known']:,} | "
            f"{strict['type_coverage'] * 100:.2f}% |"
        )
    return "\n".join(lines)


def task_token_counts(texts, tokenizer) -> Counter[str]:
    """Token frequencies over the task corpus, using the project's tokenizer.

    Must be the same tokenizer the supervised models use, or coverage is
    measured against tokens that never actually reach an embedding lookup.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        tokens, _ = tokenizer(text)
        counts.update(t for t in tokens if any(ch.isalnum() for ch in t))
    return counts
