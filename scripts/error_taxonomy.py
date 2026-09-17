"""Step 6.4 - an error taxonomy for the end-to-end pipeline (run 12).

    python scripts/error_taxonomy.py

PLAN 6.4 asks for 30 sampled pipeline failures, categorised as negation,
hedging, multi-drug, abbreviation or boundary. Those are two different kinds of
category, and this keeps them apart:

* **What went wrong** - the gate missed an ADE sentence, the gate passed a
  non-ADE one, or Stage 2 got an entity's boundaries, label or existence wrong.
  These are properties of the *error* and are decided exactly by the tags.
* **What the sentence contains** - a negation or hedging cue, several drugs, an
  abbreviation inside an entity. These are properties of the *sentence*, decided
  by frozen surface rules, and a sentence having one does not mean it caused the
  error.

Every failure is classified, and the sentence properties are compared between
failures and successes **within the same population** - gate misses against
gate hits among ADE sentences, gate false alarms against correct rejections
among non-ADE sentences, Stage 2 errors against Stage 2 successes among the ADE
sentences the gate passed - so a rate never reflects the gate's class mix.

Thirty failures are then sampled for reading (seeded, listed in test-set order)
and written to results/error_sample.csv with `manual_category` and `notes`
columns. The rules say what a sentence contains; only a reader can say what
caused the error. Whatever is already in those columns is carried over by test
index and checked against the frozen codebook in `src/error_sample.py` before
anything is written, and the report shows shares of the causes only once all
thirty are read.

A sentence is a pipeline failure when the pipeline's entity set for it differs
from the gold set. A machine-readable summary goes to results/error_taxonomy.json
for the report generator.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pipeline_eval as pe  # noqa: E402
from src import challenge_set, error_sample  # noqa: E402
from src.models.encoding import is_indexable  # noqa: E402
from src.stage2_metrics import strict_entities  # noqa: E402
from src.tokenizer import tokenize  # noqa: E402

OUT_MD = REPO_ROOT / "results" / "error_taxonomy.md"
OUT_CSV = REPO_ROOT / "results" / "error_sample.csv"
OUT_JSON = REPO_ROOT / "results" / "error_taxonomy.json"
RUN, GATE, TAGGER = "12", "8", "11"
SAMPLE, SEED = 30, 42
CATEGORISED_BY = ("drafted by an AI coding agent from the sentence text; not yet reviewed by the "
                  "authors")

# Two or more capitals and nothing lowercase: INH, AZT, 5-FU, IL-2. Fixed with
# the taxonomy, before any failure was read.
ABBREVIATION = re.compile(r"^(?=(?:[^A-Z]*[A-Z]){2})[A-Z0-9][A-Z0-9-]*$")

KINDS = ("gate miss", "gate false alarm", "boundary", "wrong label", "missed entity",
         "spurious entity")
PROPERTIES = ("negation", "hedging", "multi_drug", "abbreviation")
PROPERTY_NAMES = {"negation": "negation cue", "hedging": "hedging cue",
                  "multi_drug": "two or more drugs", "abbreviation": "abbreviation in an entity"}
POPULATIONS = ("gate miss", "gate false alarm", "Stage 2 error")


def surfaces(text: str) -> list[str]:
    """Original-case surface of each scored word, aligned with the pipeline's tags."""
    tokens, offsets = tokenize(text, lower=True)
    return [text[s:e] for token, (s, e) in zip(tokens, offsets) if is_indexable(token)]


def overlaps(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def render(entities, words, quote="`") -> str:
    return "; ".join(f"{label} {quote}{' '.join(words[s:e])}{quote}"
                     for s, e, label in entities) or "none"


def classify(text, label, decision, gold, pred, words) -> dict:
    g, p = strict_entities(gold), strict_entities(pred)
    kinds = set()

    if label and not decision:
        kinds.add("gate miss")
    if not label and decision and p:
        kinds.add("gate false alarm")
    if label and decision:
        for entity in g:
            if entity in p:
                continue
            if any(overlaps(x, entity) and x[2] == entity[2] for x in p):
                kinds.add("boundary")
            elif any(overlaps(x, entity) for x in p):
                kinds.add("wrong label")
            else:
                kinds.add("missed entity")
        for entity in p:
            if entity in g:
                continue
            if any(overlaps(entity, x) and x[2] == entity[2] for x in g):
                kinds.add("boundary")
            elif any(overlaps(entity, x) for x in g):
                kinds.add("wrong label")
            else:
                kinds.add("spurious entity")

    # Sentence properties are read off the true entities wherever there are any, so a
    # tagger's own mistakes cannot create the property they are then compared on. A
    # non-ADE sentence has none, so its predicted entities stand in - and those
    # sentences are compared on the cue properties only.
    cues = challenge_set.cue_matches(text)
    basis = g if label else p
    drugs = sum(e[2] == "DRUG" for e in basis)
    in_entities = {k for s, e, _ in basis for k in range(s, e)}

    return {
        "failure": set(g) != set(p),
        "kinds": [k for k in KINDS if k in kinds],
        "negation": bool(cues["negation"]),
        "hedging": bool(cues["hedging"]),
        "negation_cues": cues["negation"],
        "hedging_cues": cues["hedging"],
        "multi_drug": drugs >= 2,
        "abbreviation": any(ABBREVIATION.match(words[k]) for k in in_entities),
        "gold": g,
        "pred": p,
    }


def compare(failures, successes, prop) -> dict | None:
    """Fisher's exact test of a sentence property between failures and successes."""
    from scipy.stats import fisher_exact

    if not failures or not successes:
        return None
    a = sum(r[prop] for r in failures)
    c = sum(r[prop] for r in successes)
    odds, p = fisher_exact([[a, len(failures) - a], [c, len(successes) - c]])
    return {"fail_rate": a / len(failures), "ok_rate": c / len(successes),
            "odds": float(odds), "p": float(p),
            "n_fail": len(failures), "n_ok": len(successes)}


def population(r) -> str:
    if r["label"] and not r["decision"]:
        return "gate miss"
    return "Stage 2 error" if r["label"] else "gate false alarm"


def read_causes(sample, rows, manual) -> dict:
    """What the manual columns say, counted. Intervals only once every sampled row is read."""
    read = {i: error_sample.validate(manual[i][0]) for i in sample if manual[i][0].strip()}
    primary = Counter(cats[0] for cats in read.values())
    secondary = Counter(cats[1] for cats in read.values() if len(cats) == 2)
    by_population = Counter((cats[0], population(rows[i])) for i, cats in read.items())
    complete = bool(sample) and len(read) == len(sample)
    categories = error_sample.CATEGORIES
    return {
        "categorised_by": CATEGORISED_BY,
        "sampled": len(sample), "read": len(read), "complete": complete,
        "primary": {c: primary[c] for c in categories},
        "secondary": {c: secondary[c] for c in categories},
        "ci95_primary": ({c: list(error_sample.wilson(primary[c], len(sample))) for c in categories}
                         if complete else None),
        "primary_by_population": {c: {p: by_population[(c, p)] for p in POPULATIONS}
                                  for c in categories if primary[c]},
        "population_sizes": dict(Counter(population(rows[i]) for i in sample)),
        "low_confidence": sum(error_sample.is_low_confidence(manual[i][1]) for i in read),
        "gold_questionable": primary["gold questionable"],
        "hedging_cue_rows": sum(rows[i]["hedging"] for i in sample),
        "hedging_cue_rows_read_as_hedging": sum(rows[i]["hedging"] and "hedging" in read[i]
                                                for i in read),
    }


def share(k, n, ci) -> str:
    return f"{k} of {n}, {k / n:.0%} [{ci[0]:.0%}, {ci[1]:.0%}]"


def causes_section(causes, n_failures) -> list[str]:
    n, k = causes["sampled"], causes["read"]
    L = ["## Causes, from reading the sample", ""]
    if not causes["complete"]:
        L += [f"**{k} of {n} read.** Shares of the causes appear here only once every sampled "
              "failure has been read, so a partly read sample is never quoted as a distribution.", ""]
        if k:
            L += [f"The categories read so far were {CATEGORISED_BY}.", ""]
        return L

    primary, secondary, ci = causes["primary"], causes["secondary"], causes["ci95_primary"]
    categories = error_sample.CATEGORIES
    ranked = sorted(categories, key=lambda c: (-primary[c], categories.index(c)))
    L += [
        f"**Provenance.** The categories below were {CATEGORISED_BY}.",
        "",
        f"Each of the {n} sampled failures was read - the sentence first, then gold against the "
        "pipeline, then the rule flags - and given one primary cause from the frozen codebook in "
        "`src/error_sample.py`, with a second only for a separate error in the same sentence. "
        "*Gold questionable* means that, read plainly under the corpus's own convention, the "
        "sentence contradicts its gold; it is the primary cause whenever it applies, because it "
        "puts in doubt whether the row is an error at all. Shares are of the "
        f"{n} sampled failures, which were drawn uniformly from all {n_failures:,}, with Wilson "
        "95% intervals.",
        "",
        "| Cause | Primary cause of | Share | 95% interval (Wilson) | Also a secondary cause of |",
        "|---|---|---|---|---|",
    ]
    for c in ranked:
        L.append(f"| {c} | {primary[c]} | {primary[c] / n:.0%} | [{ci[c][0]:.0%}, {ci[c][1]:.0%}] "
                 f"| {secondary[c]} |")
    L += [f"| all | {n} | 100% | | {sum(secondary.values())} |", ""]

    sizes = causes["population_sizes"]
    L += ["| Primary cause | Gate misses | Gate false alarms | Stage 2 errors | All |",
          "|---|---|---|---|---|"]
    for c in ranked:
        if primary[c]:
            cells = " | ".join(str(causes["primary_by_population"][c][p]) for p in POPULATIONS)
            L.append(f"| {c} | {cells} | {primary[c]} |")
    L += ["| all | " + " | ".join(str(sizes.get(p, 0)) for p in POPULATIONS) + f" | {n} |", ""]

    model_side = [c for c in ranked if c != "gold questionable" and primary[c]]
    never = [c for c in ranked if not primary[c]]
    parts = []
    if model_side:
        top = model_side[0]
        first = f"{top} ({share(primary[top], n, ci[top])})"
        if len(model_side) > 1:
            nxt = model_side[1]
            order = ("their 95% intervals overlap, so the sample shows which causes recur but not "
                     "their order" if ci[top][0] <= ci[nxt][1] else
                     "their 95% intervals do not overlap")
            parts.append(f"Setting gold-questionable rows aside, the largest primary causes are {first} "
                         f"and {nxt} ({share(primary[nxt], n, ci[nxt])}); {order}")
        else:
            parts.append(f"Setting gold-questionable rows aside, the only primary cause is {first}")
    if never:
        names = ", ".join(never[:-1]) + (" and " if len(never) > 1 else "") + never[-1]
        parts.append(f"{names} {'was' if len(never) == 1 else 'were'} never the primary cause, "
                     f"which puts each below about {error_sample.wilson(0, n)[1]:.0%} of all "
                     f"{n_failures:,} failures (the 95% upper bound)")
    hedged = causes["hedging_cue_rows"]
    if hedged:
        caused = causes["hedging_cue_rows_read_as_hedging"]
        parts.append(f"{'none' if not caused else caused} of the {hedged} sampled failures with a "
                     f"hedging cue {'was' if caused <= 1 else 'were'} read as caused by hedging")
    sentence = "; ".join(parts) + "."
    gq = primary["gold questionable"]
    if gq:
        lo, hi = ci["gold questionable"]
        gold = (f"{gq} of the {n} sampled failures ({gq / n:.0%}, 95% interval [{lo:.0%}, {hi:.0%}]) "
                "are gold questionable: read plainly, the sentence contradicts its gold label or "
                f"entities, so somewhere between about {lo:.0%} and {hi:.0%} of what is scored as "
                "pipeline failure may be annotation error rather than model error.")
    else:
        gold = (f"No sampled failure is gold questionable, which puts annotation error at no more "
                f"than about {ci['gold questionable'][1]:.0%} of what is scored as pipeline failure.")
    L += [sentence[0].upper() + sentence[1:] + " " + gold, ""]
    return L


def cell(result) -> str:
    if result is None:
        return "-"
    odds = "inf" if result["odds"] == float("inf") else f"{result['odds']:.2f}"
    return (f"{result['fail_rate']:.1%} vs {result['ok_rate']:.1%} "
            f"(OR {odds}, p = {result['p']:.3f})")


def main() -> int:
    texts, labels, _, gold = pe.load_test()
    decisions = [int(d) for d in pe.stage1_decisions()[GATE]["y_pred"]]
    if not (pe.CACHE / f"stage2_run{TAGGER}_on_stage1_test.json").exists():
        raise SystemExit("no cached Stage 2 decode - run scripts/pipeline_eval.py first")
    decode = pe.stage2_decode(TAGGER, texts, labels, [None] * len(texts), refresh=False)
    pipeline = pe.build_settings(gold, labels, decisions, decode["pred"])["pipeline"]

    rows = []
    for i, text in enumerate(texts):
        words = surfaces(text)
        if len(words) != len(gold[i]):
            raise SystemExit(f"sentence {i}: {len(words)} surface words for {len(gold[i])} tags")
        info = classify(text, labels[i], decisions[i], gold[i], pipeline[i], words)
        info.update({"index": i, "text": text, "label": labels[i],
                     "decision": decisions[i], "words": words})
        rows.append(info)

    failures = [r for r in rows if r["failure"]]
    if any(not r["kinds"] for r in failures):
        raise SystemExit("a failing sentence received no error kind - the classifier is incomplete")

    # ---- populations in which a comparison means something ---------------------------
    ade = [r for r in rows if r["label"]]
    gate_miss, gate_hit = [r for r in ade if not r["decision"]], [r for r in ade if r["decision"]]
    non_ade = [r for r in rows if not r["label"]]
    false_alarm = [r for r in non_ade if r["decision"] and r["failure"]]
    rejected = [r for r in non_ade if not r["decision"]]
    stage2_error = [r for r in gate_hit if r["failure"]]
    stage2_ok = [r for r in gate_hit if not r["failure"]]

    comparisons = {
        prop: {
            "gate": compare(gate_miss, gate_hit, prop),
            "alarm": compare(false_alarm, rejected, prop) if prop in ("negation", "hedging") else None,
            "stage2": compare(stage2_error, stage2_ok, prop),
        }
        for prop in PROPERTIES
    }
    tests_run = sum(v is not None for c in comparisons.values() for v in c.values())
    alpha = 0.05 / max(tests_run, 1)

    kind_counts = Counter(k for r in failures for k in r["kinds"])
    only = Counter(r["kinds"][0] for r in failures if len(r["kinds"]) == 1)

    sample = sorted(random.Random(SEED).sample([r["index"] for r in failures],
                                               min(SAMPLE, len(failures))))

    # ---- the manual columns survive a re-run --------------------------------------------
    # Read before anything is written: a reading that no longer fits the sample stops the
    # run with every output untouched.
    existing = []
    if OUT_CSV.exists():
        with OUT_CSV.open(encoding="utf-8", newline="") as fh:
            existing = list(csv.DictReader(fh))
    try:
        carried = error_sample.carry_over(existing, sample, texts={i: rows[i]["text"] for i in sample})
        for i, (category, _) in carried.items():
            if category.strip():
                error_sample.validate(category)
    except ValueError as err:
        raise SystemExit(f"{OUT_CSV.relative_to(REPO_ROOT)}: {err}")
    manual = {i: carried.get(i, ("", "")) for i in sample}
    causes = read_causes(sample, rows, manual)

    # ---- the sample, for the reviewers ------------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["test_index", "gold_label", "gate_decision", "error_kinds", "negation_cues",
                         "hedging_cues", "multi_drug", "abbreviation", "gold_entities",
                         "pipeline_entities", "text", "manual_category", "notes"])
        for i in sample:
            r = rows[i]
            writer.writerow([i, r["label"], r["decision"], "; ".join(r["kinds"]),
                             "|".join(r["negation_cues"]), "|".join(r["hedging_cues"]),
                             int(r["multi_drug"]), int(r["abbreviation"]),
                             render(r["gold"], r["words"], quote='"'),
                             render(r["pred"], r["words"], quote='"'), r["text"], *manual[i]])

    # ---- report -----------------------------------------------------------------------
    L = [
        f"# Pipeline error taxonomy (step 6.4, run {RUN})",
        "",
        f"Generated by `scripts/error_taxonomy.py` from run {RUN}'s pipeline output - Stage 1 "
        f"run {GATE} as the gate, Stage 2 run {TAGGER} as the tagger. A sentence is a "
        "**failure** when the pipeline's entity set for it differs from the gold set.",
        "",
        f"**{len(failures):,} of {len(rows):,} test sentences are failures**: "
        f"{len(gate_miss)} ADE sentences the gate rejected, {len(false_alarm)} non-ADE sentences "
        f"it passed and Stage 2 tagged, and {len(stage2_error)} of the {len(gate_hit)} passed ADE "
        "sentences with at least one entity error.",
        "",
        "PLAN 6.4 lists negation, hedging, multi-drug, abbreviation and boundary as the "
        "categories. They are two different things, so they are reported separately: "
        "*boundary* is a property of the error, decided exactly by the tags; the other four "
        "are properties of the sentence, decided by frozen surface rules, and a sentence having "
        "one does not show that it caused the error.",
        "",
        "## What went wrong",
        "",
        "| Error kind | Failing sentences | Share of failures | Only kind in the sentence |",
        "|---|---|---|---|",
    ]
    for kind in KINDS:
        n = kind_counts[kind]
        L.append(f"| {kind} | {n:,} | {n / max(len(failures), 1):.1%} | {only[kind]:,} |")
    L += [
        "",
        "A sentence can carry several kinds, so the shares sum past 100%. *Boundary* means an "
        "entity overlaps a gold entity of the same label with different extent; *wrong label* "
        "means it overlaps only one of the other label.",
        "",
        "## What the failing sentences contain",
        "",
        "Each cell compares the rate of a sentence property among failures with its rate among "
        "successes **in the same population**, with Fisher's exact test. For ADE sentences the "
        "drug count and abbreviation flag are read off the gold entities, so a tagger's own "
        f"mistakes cannot create them. {tests_run} tests are run, so a Bonferroni threshold of "
        f"p < {alpha:.4f} is the bar for calling any single difference real.",
        "",
        "| Property | Gate misses vs gate hits (ADE sentences) | Gate false alarms vs correct rejections (non-ADE sentences) | Stage 2 errors vs successes (passed ADE sentences) |",
        "|---|---|---|---|",
    ]
    for prop in PROPERTIES:
        c = comparisons[prop]
        L.append(f"| {PROPERTY_NAMES[prop]} | {cell(c['gate'])} | {cell(c['alarm'])} | "
                 f"{cell(c['stage2'])} |")
    L.append("")

    significant = [(prop, where) for prop, c in comparisons.items() for where, v in c.items()
                   if v is not None and v["p"] < alpha]
    names = {"gate": "gate misses", "alarm": "gate false alarms", "stage2": "Stage 2 errors"}
    if significant:
        L += ["Differences that clear the Bonferroni threshold: "
              + "; ".join(f"**{PROPERTY_NAMES[p]}** among {names[w]} "
                          f"({comparisons[p][w]['fail_rate']:.1%} vs {comparisons[p][w]['ok_rate']:.1%}, "
                          f"p = {comparisons[p][w]['p']:.3f})"
                          for p, w in significant) + ".", ""]
    else:
        L += ["**No difference clears the Bonferroni threshold.** On this test set, no sentence "
              "property in PLAN 6.4's list is shown to be over-represented among any kind of "
              "pipeline failure; the error kinds above describe the failures better than the "
              "sentence properties do.", ""]
    nominal = [(p, w) for p, c in comparisons.items() for w, v in c.items()
               if v is not None and alpha <= v["p"] < 0.05]
    if nominal:
        L += ["Nominally significant before correction only: "
              + "; ".join(f"{PROPERTY_NAMES[p]} among {names[w]} (p = {comparisons[p][w]['p']:.3f})"
                          for p, w in nominal)
              + ". With this many tests, one or two of these are expected by chance.", ""]

    L += causes_section(causes, len(failures))

    L += [
        f"## {len(sample)} failures to read",
        "",
        f"Sampled with seed {SEED} from all {len(failures):,} failures and listed in test-set order. "
        "The flags are the rules' output, not a diagnosis. The same rows are in "
        "`results/error_sample.csv` with empty `manual_category` and `notes` columns for the "
        "reviewers - the report should quote a cause for these errors only once those are "
        "filled in.",
        "",
    ]
    for n, i in enumerate(sample, start=1):
        r = rows[i]
        flags = [f"negation ({', '.join(r['negation_cues'])})" if r["negation"] else "",
                 f"hedging ({', '.join(r['hedging_cues'])})" if r["hedging"] else "",
                 "two or more drugs" if r["multi_drug"] else "",
                 "abbreviation in an entity" if r["abbreviation"] else ""]
        flags = [f for f in flags if f]
        truth = "ADE" if r["label"] else "not ADE"
        gate = "passed" if r["decision"] else "rejected"
        category, notes = manual[i]
        reading = (" - ".join(x for x in (category.strip(), notes.strip()) if x)
                   if category.strip() else "not read yet")
        L += [
            f"**{n}. Test sentence {i}** - {truth}, gate {gate} - {', '.join(r['kinds'])}",
            "",
            f"> {r['text']}",
            "",
            f"- Gold: {render(r['gold'], r['words'])}",
            f"- Pipeline: {render(r['pred'], r['words'])}",
            f"- Sentence flags: {', '.join(flags) or 'none'}",
            f"- Reader's category: {reading}",
            "",
        ]

    L += [
        "## Threats to validity",
        "",
        "| Threat | Status |",
        "|---|---|",
        "| Sentence properties read as causes | controlled in the text - reported as "
        "co-occurrence with a significance test; causes come only from reading the sampled "
        "sentences, in *Causes, from reading the sample* |",
    ]
    if causes["read"]:
        L += [
            f"| Causes come from a single reader | not controlled - {CATEGORISED_BY}; there is no "
            "second reader and no agreement check |",
            f"| Ambiguous readings | not controlled - {causes['low_confidence']} of the "
            f"{causes['read']} readings are marked [low confidence], where two categories were close "
            "or the sentence reads two ways |",
        ]
    L += [
        "| Cue and abbreviation rules are surface patterns | not controlled - a cue need not scope "
        "over the relation; an all-caps token need not be an abbreviation |",
        "| Comparisons confounded by the gate's class mix | controlled - each comparison stays "
        "inside one population |",
        f"| Many comparisons | controlled - Bonferroni over {tests_run} tests |",
        (f"| Thirty is a small sample | by design - the sample is for reading; every rate outside "
         f"*Causes, from reading the sample* uses all {len(failures):,} failures, and the shares "
         "inside it carry Wilson intervals |" if causes["complete"] else
         f"| Thirty is a small sample | by design - the sample is for reading; every rate above uses "
         f"all {len(failures):,} failures |"),
        "",
    ]

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "run": RUN, "gate": GATE, "tagger": TAGGER,
        "sentences": len(rows), "failures": len(failures),
        "gate_miss": len(gate_miss), "gate_hit": len(gate_hit),
        "gate_false_alarm": len(false_alarm), "stage2_error": len(stage2_error),
        "kinds": dict(kind_counts), "tests": tests_run, "bonferroni_alpha": alpha,
        "significant": [{"property": p, "population": w, **comparisons[p][w]}
                        for p, w in significant],
        "sample": sample,
        "manual": causes,
    }, indent=2), encoding="utf-8")

    print(f"{len(failures):,} failures | kinds {dict(kind_counts)}")
    print(f"manual readings: {causes['read']} of {causes['sampled']} carried over and validated")
    print(f"wrote {OUT_MD.relative_to(REPO_ROOT)}, {OUT_JSON.relative_to(REPO_ROOT)} and "
          f"{OUT_CSV.relative_to(REPO_ROOT)} ({len(sample)} sampled rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
