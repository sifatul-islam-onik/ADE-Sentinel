"""Steps 6.1-6.2 - run 12: the end-to-end pipeline, and where its loss comes from.

    python scripts/pipeline_eval.py

Stage 1 decides which sentences reach Stage 2; Stage 2 tags the ones that do.
Everything in `report/stage2_documentation.md` is the oracle setting - Stage 2
on the gold-positive sentences, as if Stage 1 were perfect. This measures the
pipeline setting on the same test split, which is valid only because the split
is global (PLAN F4), and decomposes the drop as PRD 8.4 asks:

* **Stage 1 false negatives** - ADE sentences the gate rejects. Their entities
  never reach Stage 2, so every one of them is lost recall.
* **Stage 1 false positives** - non-ADE sentences the gate lets through. They
  have no gold entities, so anything Stage 2 extracts from them is a false
  positive.

The two interact through F1's shared denominator, so "remove the false negatives,
then the false positives" gives a different split from the reverse order. Both
orders are computed and the loss is divided by their average - the two-player
Shapley value, which sums exactly to the total - with sentence-level bootstrap
intervals.

Nothing is trained. Stage 1 decisions are the saved test predictions of the gate
run. Stage 2 is decoded on CPU over every Stage 1 test sentence through the path
`scripts/check_stage2_inference.py` verifies against the remote decode, cached
under models/pipeline/, and checked again here against the saved oracle
predictions before anything is scored.

Writes results/figures/pipeline_results.md and pipeline_loss.png, and appends
runs 12 and 12b to results/runs.csv (skip with --no-log).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.stage2_metrics import entity_metrics, error_breakdown, strict_entities  # noqa: E402
from src.utils import RUNS_CSV  # noqa: E402

SPLITS = REPO_ROOT / "data" / "splits"
STAGE1 = REPO_ROOT / "models" / "stage1"
STAGE2 = REPO_ROOT / "models" / "stage2"
MATRICES = REPO_ROOT / "models" / "emb_matrices"
CACHE = REPO_ROOT / "models" / "pipeline"
FIGURES = REPO_ROOT / "results" / "figures"
TABLE = FIGURES / "pipeline_results.md"
CHART = FIGURES / "pipeline_loss.png"

# run id, Stage 1 gate, Stage 2 tagger, why this pairing exists
PIPELINES = (
    ("12", "8", "11", "best Stage 1 into best Stage 2 (PLAN 6.1)"),
    ("12b", "8", "10", "best Stage 1 into the BiLSTM-CRF the demo ships (PLAN 7.2)"),
)
SETTINGS = ("oracle", "fn_only", "fp_only", "pipeline")
SEED = 42


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bootstrap", type=int, default=2000, help="resamples for the intervals")
    p.add_argument("--refresh-cache", action="store_true",
                   help="re-decode Stage 2 even if a cached decode exists")
    p.add_argument("--no-log", action="store_true", help="do not append to results/runs.csv")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_test():
    """Stage 1 test sentences, with word-level gold tags (all `O` for non-ADE sentences).

    Gold for an ADE sentence comes from its Stage 2 spans through the same
    converter and punctuation filter runs 9-11 were scored with; a non-ADE
    sentence has no entities by the task's definition.
    """
    import pandas as pd

    from src.bio_convert import to_bio
    from src.models.encoding import is_indexable
    from src.stage2_inference import words_of

    stage1 = pd.read_parquet(SPLITS / "stage1_test.parquet")
    stage2 = pd.read_parquet(SPLITS / "stage2_test.parquet")
    spans_by_text = dict(zip(stage2.text, stage2.spans))

    if set(stage1.text[stage1.label == 1]) != set(spans_by_text):
        raise SystemExit("the Stage 2 test sentences are not exactly the Stage 1 test "
                         "positives (PLAN F4) - one split cannot score the pipeline")

    texts, labels, words, gold = [], [], [], []
    for text, label in zip(stage1.text, stage1.label):
        if label == 1:
            spans = [(int(s), int(e), str(lab)) for s, e, lab in json.loads(spans_by_text[text])]
            tokens, tags = to_bio(text, spans, strict=False)
            kept = [(t, g) for t, g in zip(tokens, tags) if is_indexable(t)]
            sentence_words, sentence_gold = [t for t, _ in kept], [g for _, g in kept]
        else:
            sentence_words = words_of(text)
            sentence_gold = ["O"] * len(sentence_words)
        texts.append(text)
        labels.append(int(label))
        words.append(sentence_words)
        gold.append(sentence_gold)
    return texts, labels, words, gold


def stage1_decisions() -> dict:
    """Saved Stage 1 test predictions per run, aligned row-for-row with the test split."""
    import numpy as np

    out = {}
    for path in sorted(STAGE1.glob("run*/test_predictions.npz")):
        run_id = path.parent.name.split("_")[0].removeprefix("run")
        saved = np.load(path)
        out[run_id] = {"y_true": saved["y_true"].astype(int), "y_pred": saved["y_pred"].astype(int)}
    return out


def stage1_rows() -> dict:
    import pandas as pd

    df = pd.read_csv(RUNS_CSV, dtype={"run_id": str, "stage": str})
    df = df[df.stage == "1"].drop_duplicates(subset="run_id", keep="last")
    return {r.run_id: (r.model, json.loads(r.metrics_json)) for r in df.itertuples()}


def stage2_logged_f1(run_id: str):
    import pandas as pd

    df = pd.read_csv(RUNS_CSV, dtype={"run_id": str, "stage": str})
    rows = df[(df.stage == "2") & (df.run_id == run_id)]
    return json.loads(rows.iloc[-1].metrics_json)["entity_f1_strict"] if len(rows) else None


def stage2_decode(run_id: str, texts, labels, words, refresh: bool) -> dict:
    """Stage 2 tags for every Stage 1 test sentence, decoded on CPU and cached.

    Before it is used, the decode is compared with the tags the remote session
    saved for the gold-positive sentences. Those are the sentences both settings
    share, so any disagreement would make the oracle row here a different
    measurement from the Stage 2 report's.
    """
    from src.stage2_inference import BertTagger, BiLSTMTaggerRunner

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"stage2_run{run_id}_on_stage1_test.json"
    result = None
    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("text") == texts:
            result = cached

    if result is None:
        builders = {
            "9": lambda: BiLSTMTaggerRunner(STAGE2 / "run9_softmax" / "checkpoint.pt", MATRICES),
            "10": lambda: BiLSTMTaggerRunner(STAGE2 / "run10_crf" / "checkpoint.pt", MATRICES),
            "11": lambda: BertTagger(STAGE2 / "run11_biomedbert" / "best"),
        }
        tagger = builders[run_id]()
        t0 = time.perf_counter()
        pred = tagger.tag(words)
        seconds = time.perf_counter() - t0

        if hasattr(tagger, "tokenizer"):
            nonempty = [w for w in words if w]
            pieces = tagger.tokenizer(nonempty, is_split_into_words=True, truncation=False)
            truncated = sum(len(ids) > tagger.max_len for ids in pieces["input_ids"])
            limit = f"{tagger.max_len} subwords"
        else:
            truncated = sum(len(w) > tagger.max_len for w in words)
            limit = f"{tagger.max_len} tokens"

        result = {"text": texts, "pred": pred, "decode_seconds": round(seconds, 1),
                  "truncated_sentences": truncated, "limit": limit}
        path.write_text(json.dumps(result), encoding="utf-8")

    saved_dir = next(STAGE2.glob(f"run{run_id}_*"))
    saved = json.loads((saved_dir / "test_predictions.json").read_text(encoding="utf-8"))
    by_text = dict(zip(saved["text"], saved["pred"]))
    differing = sum(result["pred"][i] != by_text[t]
                    for i, t in enumerate(texts) if labels[i] == 1)
    if differing:
        raise SystemExit(f"run {run_id}: the CPU decode differs from the saved remote decode "
                         f"on {differing} gold-positive sentences - rerun "
                         "scripts/check_stage2_inference.py and do not score the pipeline")
    return result


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def build_settings(gold, labels, decisions, pred) -> dict:
    """The four tag sets whose F1s define the oracle, the pipeline and the two mixes.

    oracle:   Stage 2 sees exactly the gold-positive sentences.
    fn_only:  only the gate's misses are applied - positives it rejects get nothing.
    fp_only:  only the gate's false alarms are applied - they are tagged too.
    pipeline: both, i.e. Stage 2 sees exactly what the gate passes.
    """
    blank = [["O"] * len(g) for g in gold]
    rows = range(len(gold))
    return {
        "oracle": [pred[i] if labels[i] else blank[i] for i in rows],
        "fn_only": [pred[i] if labels[i] and decisions[i] else blank[i] for i in rows],
        "fp_only": [pred[i] if labels[i] or decisions[i] else blank[i] for i in rows],
        "pipeline": [pred[i] if decisions[i] else blank[i] for i in rows],
    }


def per_sentence_counts(gold, pred):
    """(true positives, predicted, gold) per sentence, exact match - F1 = 2TP / (P + G)."""
    import numpy as np

    tp = np.zeros(len(gold), dtype=np.int64)
    n_pred = np.zeros(len(gold), dtype=np.int64)
    n_gold = np.zeros(len(gold), dtype=np.int64)
    for i, (g, p) in enumerate(zip(gold, pred)):
        gold_entities = set(strict_entities(g))
        predicted = strict_entities(p)
        tp[i] = sum(e in gold_entities for e in predicted)
        n_pred[i], n_gold[i] = len(predicted), len(gold_entities)
    return tp, n_pred, n_gold


def decompose(f1: dict) -> dict:
    """Split oracle - pipeline into a false-negative and a false-positive share."""
    fn_first = (f1["oracle"] - f1["fn_only"], f1["fn_only"] - f1["pipeline"])   # (FN, FP)
    fp_first = (f1["fp_only"] - f1["pipeline"], f1["oracle"] - f1["fp_only"])   # (FN, FP)
    return {
        "total": f1["oracle"] - f1["pipeline"],
        "fn": (fn_first[0] + fp_first[0]) / 2,
        "fp": (fn_first[1] + fp_first[1]) / 2,
        "fn_first": fn_first,
        "fp_first": fp_first,
    }


def bootstrap(counts: dict, reps: int, seed: int = SEED) -> dict:
    """95% sentence-level bootstrap intervals for the total loss and both shares."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(counts["oracle"][0])
    draws = {"total": [], "fn": [], "fp": []}

    for start in range(0, reps, 250):
        index = rng.integers(0, n, size=(min(250, reps - start), n))
        f1 = {}
        for name, (tp, n_pred, n_gold) in counts.items():
            hits = tp[index].sum(axis=1)
            denominator = n_pred[index].sum(axis=1) + n_gold[index].sum(axis=1)
            f1[name] = np.where(denominator > 0, 2 * hits / np.maximum(denominator, 1), 0.0)
        split = decompose(f1)
        for key in draws:
            draws[key].append(split[key])

    return {key: (float(np.percentile(np.concatenate(v), 2.5)),
                  float(np.percentile(np.concatenate(v), 97.5)))
            for key, v in draws.items()}


def gate_counts(gold, labels, decisions, pred) -> dict:
    tp = [i for i, (l, d) in enumerate(zip(labels, decisions)) if l and d]
    fn = [i for i, (l, d) in enumerate(zip(labels, decisions)) if l and not d]
    fp = [i for i, (l, d) in enumerate(zip(labels, decisions)) if not l and d]
    tn = len(labels) - len(tp) - len(fn) - len(fp)

    fn_gold = sum(len(strict_entities(gold[i])) for i in fn)
    fn_found = sum(len(set(strict_entities(gold[i])) & set(strict_entities(pred[i]))) for i in fn)
    fp_entities = Counter(e[2] for i in fp for e in strict_entities(pred[i]))
    fp_with = sum(1 for i in fp if strict_entities(pred[i]))

    return {"tp": len(tp), "fn": len(fn), "fp": len(fp), "tn": tn,
            "fn_gold_entities": fn_gold, "fn_oracle_found": fn_found,
            "fp_sentences_with_entities": fp_with,
            "fp_predicted_entities": sum(fp_entities.values()),
            "fp_predicted_by_label": dict(fp_entities)}


def score_pipeline(gold, labels, decisions, pred, reps: int) -> dict:
    tag_sets = build_settings(gold, labels, decisions, pred)
    scores = {}
    for name, tags in tag_sets.items():
        metrics = entity_metrics(gold, tags)
        breakdown = error_breakdown(gold, tags)
        metrics["overlap_f1"] = breakdown["overlap_f1"]
        metrics["pred_entities_strict"] = sum(
            breakdown[k] for k in ("pred_exact", "pred_boundary", "pred_type", "pred_spurious"))
        scores[name] = metrics

    f1 = {name: scores[name]["entity_f1_strict"] for name in SETTINGS}
    counts = {name: per_sentence_counts(gold, tag_sets[name]) for name in SETTINGS}
    return {
        "scores": scores,
        "decomposition": decompose(f1),
        "ci": bootstrap(counts, reps) if reps else None,
        "counts": gate_counts(gold, labels, decisions, pred),
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def share(part: float, total: float) -> str:
    return f"{part / total:.0%}" if abs(total) > 1e-12 else "-"


def interval(ci, key) -> str:
    return f"[{ci[key][0]:.4f}, {ci[key][1]:.4f}]" if ci else ""


def make_chart(results) -> bool:
    """Left: oracle vs pipeline entity-F1. Right: the loss split by its source, with intervals.

    The split gets its own panel on its own scale. Drawn as slivers stacked on an
    F1 bar it is either too thin to read or needs a truncated axis that
    exaggerates it - and the claim the section makes is about the split, with
    its intervals, so that is what the second panel shows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [f"run {r['run_id']}\n(Stage 2 run {r['s2']})" for r in results]
    x = np.arange(len(results))
    width = 0.36
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=200)

    oracle = [r["scores"]["oracle"]["entity_f1_strict"] for r in results]
    pipeline = [r["scores"]["pipeline"]["entity_f1_strict"] for r in results]
    left.bar(x - width / 2, oracle, width, color="#90a4ae", label="oracle", edgecolor="white")
    left.bar(x + width / 2, pipeline, width, color="#2e7d32", label="pipeline", edgecolor="white")
    for xs, values in ((x - width / 2, oracle), (x + width / 2, pipeline)):
        for xi, v in zip(xs, values):
            left.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    left.set_ylim(0, 1.08)
    left.set_xticks(x)
    left.set_xticklabels(names, fontsize=8.5)
    left.set_ylabel("strict entity-F1 (test)")
    left.set_title("Oracle vs pipeline", fontsize=10.5)
    left.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)

    has_ci = all(r["ci"] for r in results)
    tops = []
    for offset, key, colour, label in ((-width / 2, "fn", "#c62828", "from Stage 1 false negatives"),
                                       (width / 2, "fp", "#ef6c00", "from Stage 1 false positives")):
        values = [r["decomposition"][key] for r in results]
        errors = None
        if has_ci:
            errors = [[v - r["ci"][key][0] for v, r in zip(values, results)],
                      [r["ci"][key][1] - v for v, r in zip(values, results)]]
        right.bar(x + offset, values, width, yerr=errors, capsize=4, color=colour,
                  label=label, edgecolor="white", error_kw={"lw": 1})
        for i, v in enumerate(values):
            top = results[i]["ci"][key][1] if has_ci else v
            tops.append(top)
            right.text(x[i] + offset, top + 0.002, f"{v:.4f}", ha="center", fontsize=8)
    right.set_ylim(0, max(tops) * 1.45)
    right.set_xticks(x)
    right.set_xticklabels(names, fontsize=8.5)
    right.set_ylabel("strict entity-F1 lost")
    right.set_title("Where the loss comes from (Shapley split"
                    + (", 95% intervals)" if has_ci else ")"), fontsize=10.5)
    right.legend(frameon=False, fontsize=8, loc="upper right")

    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(CHART)
    plt.close(fig)
    return True


def write_report(results, ladder, stage1, reps, decodes) -> None:
    head = results[0]
    gold_total = head["scores"]["oracle"]["gold_entities"]

    L = [
        "# End-to-end pipeline (steps 6.1-6.2, runs 12 and 12b)",
        "",
        "Generated by `scripts/pipeline_eval.py`. Nothing is trained: Stage 1 decisions are "
        "the saved test predictions of the gate run, and Stage 2 is decoded on CPU over "
        "every Stage 1 test sentence through the path `scripts/check_stage2_inference.py` "
        "verifies against the remote decode.",
        "",
        "Stage 2's own report scores it in the **oracle** setting - on the gold-positive "
        "sentences, as if Stage 1 were perfect. The **pipeline** setting lets Stage 1 "
        "decide: sentences it rejects get no entities, and sentences it wrongly accepts "
        "are tagged like any other. Both are scored over the same "
        f"{head['n_sentences']:,}-sentence test split, containing all {gold_total:,} gold "
        "entities, which is valid only because Stage 1 and Stage 2 share one global split "
        "(PLAN F4).",
        "",
        "## The gate",
        "",
    ]

    for s1 in sorted({r["s1"] for r in results}):
        c = next(r["counts"] for r in results if r["s1"] == s1)
        model, metrics = stage1.get(s1, ("?", {}))
        positives, passed = c["tp"] + c["fn"], c["tp"] + c["fp"]
        L += [
            f"Stage 1 run {s1} (`{model}`, test macro-F1 {metrics.get('macro_f1', float('nan')):.4f}):",
            "",
            "| | Gate says ADE | Gate says not-ADE |",
            "|---|---|---|",
            f"| ADE sentence | {c['tp']:,} | {c['fn']:,} |",
            f"| not-ADE sentence | {c['fp']:,} | {c['tn']:,} |",
            "",
            f"It passes {passed:,} sentences to Stage 2: {c['tp']:,} of the {positives:,} ADE "
            f"sentences ({c['tp'] / positives:.1%} recall) and {c['fp']:,} that are not "
            f"({c['tp'] / max(passed, 1):.1%} precision).",
            "",
        ]

    L += [
        "## Oracle vs pipeline",
        "",
        "| Run | Stage 2 | Setting | Strict F1 | Precision | Recall | Lenient F1 | Overlap F1 | Predicted entities |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        for setting in ("oracle", "pipeline"):
            m = r["scores"][setting]
            label = "**pipeline**" if setting == "pipeline" else "oracle"
            L.append(
                f"| {r['run_id']} | run {r['s2']} | {label} | {m['entity_f1_strict']:.4f} | "
                f"{m['entity_precision_strict']:.3f} | {m['entity_recall_strict']:.3f} | "
                f"{m['entity_f1_lenient']:.4f} | {m['overlap_f1']:.4f} | "
                f"{m['pred_entities_strict']:,} |")
    L.append("")

    L += [
        "Each oracle row reproduces its Stage 2 run's logged test F1 exactly ("
        + ", ".join(f"run {r['s2']} {r['logged_stage2_f1']:.4f}" for r in results)
        + "), so the pipeline rows are measured with the same scorer, over the same entities, "
        "as `report/stage2_documentation.md`.",
        "",
    ]
    if CHART.exists():
        L += [f"![Oracle vs pipeline]({CHART.name})", ""]

    so, sp = head["scores"]["oracle"], head["scores"]["pipeline"]
    d, ci = head["decomposition"], head["ci"]
    L += [
        f"**Chaining the stages costs run {head['run_id']} {d['total']:.4f} strict entity-F1** "
        f"({so['entity_f1_strict']:.4f} oracle to {sp['entity_f1_strict']:.4f} pipeline). "
        f"Recall falls from {so['entity_recall_strict']:.3f} to {sp['entity_recall_strict']:.3f} "
        f"and precision from {so['entity_precision_strict']:.3f} to "
        f"{sp['entity_precision_strict']:.3f}.",
        "",
        "## Where the loss comes from (PRD 8.4)",
        "",
        "The loss has two sources, and they interact. Stage 1 false negatives take gold "
        "entities out of reach; Stage 1 false positives add predictions with no gold behind "
        "them. Both act on F1's shared denominator, so whichever is removed first changes "
        "how much the other appears to cost. Both orders are computed below, and the loss "
        "is divided by their average - the two-player Shapley value, which sums exactly to "
        "the total."
        + (f" Brackets are 95% sentence-level bootstrap intervals ({reps:,} resamples)."
           if reps else ""),
        "",
        "| Run | Total loss | From Stage 1 false negatives | From Stage 1 false positives | FN removed first (FN / FP) | FP added first (FN / FP) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        rd, rci = r["decomposition"], r["ci"]
        L.append(
            f"| {r['run_id']} | {rd['total']:.4f} {interval(rci, 'total')} | "
            f"{rd['fn']:.4f} ({share(rd['fn'], rd['total'])}) {interval(rci, 'fn')} | "
            f"{rd['fp']:.4f} ({share(rd['fp'], rd['total'])}) {interval(rci, 'fp')} | "
            f"{rd['fn_first'][0]:.4f} / {rd['fn_first'][1]:.4f} | "
            f"{rd['fp_first'][0]:.4f} / {rd['fp_first'][1]:.4f} |")
    L.append("")

    c = head["counts"]
    by_label = ", ".join(f"{n} {label}" for label, n in sorted(c["fp_predicted_by_label"].items()))
    L += [
        f"**{c['fn']} ADE sentences never reached Stage 2.** They hold {c['fn_gold_entities']} "
        f"of the {gold_total:,} gold entities, and Stage 2 finds {c['fn_oracle_found']} of those "
        "exactly when it is given the sentences - so the gate's misses cost "
        f"{c['fn_oracle_found']} true positives outright.",
        "",
        f"**{c['fp']} non-ADE sentences did reach it.** Stage 2 extracted "
        f"{c['fp_predicted_entities']} entities from "
        + ("every one of them" if c["fp_sentences_with_entities"] == c["fp"]
           else f"{c['fp_sentences_with_entities']} of them")
        + (f" ({by_label}; {c['fp_predicted_entities'] / max(c['fp'], 1):.1f} per sentence)"
           if by_label else "")
        + ", and all of them count as false positives. Not all of them are tagging "
        "mistakes: ADE Corpus v2 annotates drugs and effects only inside adverse-event "
        "relations, so a drug correctly named in a sentence that reports no adverse event "
        "is still not an entity of this task. The sentences a gate wrongly passes are, "
        "almost by definition, the ones that read like ADE reports - which is why "
        + ("every one of them hands" if c["fp_sentences_with_entities"] == c["fp"]
           else "most of them hand")
        + " Stage 2 something to extract.",
        "",
    ]

    if ci:
        fn_low, fn_high = ci["fn"]
        fp_low, fp_high = ci["fp"]
        untested = ("It is not tested here: moving the threshold would need Stage 1 dev "
                    "scores, which were not saved, and choosing it on test would contaminate "
                    "every number above.")
        if fn_low > fp_high:
            L += ["**The gate's false negatives dominate the loss**, and the bootstrap "
                  "intervals for the two shares do not overlap. That points at the gate's "
                  "recall as the lever for the pipeline. " + untested, ""]
        elif fp_low > fn_high:
            L += ["**The gate's false positives dominate the loss**, and the bootstrap "
                  "intervals for the two shares do not overlap. That points at the gate's "
                  "precision, not its recall, as the lever for the pipeline - the opposite of "
                  "the usual instinct to tune a screening stage for recall. " + untested, ""]
        else:
            larger = "false negatives" if d["fn"] > d["fp"] else "false positives"
            L += [f"The {larger} account for the larger share, but the bootstrap intervals "
                  "for the two shares overlap, so this test set cannot say which source "
                  "dominates.", ""]

    if len(results) > 1:
        b = results[1]
        L += [
            f"Run {b['run_id']} puts the BiLSTM-CRF behind the same gate - the configuration "
            f"PLAN 7.2 ships in the demo. It loses {b['decomposition']['total']:.4f} "
            f"({b['scores']['oracle']['entity_f1_strict']:.4f} to "
            f"{b['scores']['pipeline']['entity_f1_strict']:.4f}). The gate is identical, so "
            f"the {c['fn']} missed and {c['fp']} extra sentences are the same; the loss differs "
            "only through what each tagger does with them.",
            "",
        ]

    # ---- the gate ladder ----------------------------------------------------------
    if ladder:
        L += [
            "## Does a better gate matter?",
            "",
            f"Every Stage 1 run with saved test predictions, used as the gate in front of "
            f"Stage 2 run {head['s2']}. Stage 2 is held fixed, so the rows differ only in which "
            "sentences reach it.",
            "",
            "| Gate | Model | Gate macro-F1 | ADE recall | ADE precision | Pipeline strict F1 | Loss | from FN | from FP |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in sorted(ladder, key=lambda x: -x["pipeline_f1"]):
            L.append(
                f"| run {row['s1']} | `{row['model']}` | {row['macro_f1']:.4f} | "
                f"{row['recall']:.3f} | {row['precision']:.3f} | **{row['pipeline_f1']:.4f}** | "
                f"{row['loss']:.4f} | {row['fn']:.4f} | {row['fp']:.4f} |")
        L.append("")

        import numpy as np

        best_gate = max(ladder, key=lambda x: x["macro_f1"])
        best_pipe = max(ladder, key=lambda x: x["pipeline_f1"])
        pipe = np.array([x["pipeline_f1"] for x in ladder])
        corr_recall = float(np.corrcoef([x["recall"] for x in ladder], pipe)[0, 1])
        corr_precision = float(np.corrcoef([x["precision"] for x in ladder], pipe)[0, 1])
        corr_macro = float(np.corrcoef([x["macro_f1"] for x in ladder], pipe)[0, 1])
        L += [
            ("The gate with the best macro-F1 is also the best gate for the pipeline "
             f"(run {best_gate['s1']})."
             if best_gate["s1"] == best_pipe["s1"] else
             f"**The gate with the best macro-F1 (run {best_gate['s1']}) is not the best gate "
             f"for the pipeline (run {best_pipe['s1']}).**")
            + f" Across these {len(ladder)} gates, pipeline F1 correlates with gate macro-F1 "
            f"at r = {corr_macro:.2f}, with ADE recall at r = {corr_recall:.2f} and with ADE "
            f"precision at r = {corr_precision:.2f}"
            + ("; tracking precision more closely than recall is consistent with false "
               "positives being the larger loss" if corr_precision > corr_recall else "")
            + f". With {len(ladder)} points these are descriptive, not a test.",
            "",
        ]

    # ---- threats --------------------------------------------------------------------
    trunc = ", ".join(f"run {rid}: {v['truncated_sentences']} of {len(v['text']):,} sentences "
                      f"over {v['limit']}" for rid, v in decodes.items())
    L += [
        "## Threats to validity",
        "",
        "| Threat | Status |",
        "|---|---|",
        "| Stage 1 threshold not tuned for the pipeline | **not controlled** - the gate is the "
        "saved argmax decision; Stage 1 dev scores were not saved, so no threshold can be "
        "chosen without touching test |",
        "| Stage 2 decoded locally, not remotely | controlled - the CPU decode is checked "
        "against the saved remote decode on every gold-positive sentence before scoring, and "
        "each oracle row reproduces the logged F1 |",
        f"| Long sentences truncated by the tagger | measured - {trunc}; words past the limit "
        "are tagged `O` |",
        "| Drugs named in non-ADE sentences scored as false positives | by task definition - "
        "stated above |",
        "| Single seed | **not controlled** - every model in the chain is seed 42 only |",
        "",
    ]

    TABLE.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)

    texts, labels, words, gold = load_test()
    decisions = stage1_decisions()
    stage1 = stage1_rows()
    print(f"stage1_test: {len(texts):,} sentences, {sum(labels)} ADE-positive")

    for run_id, saved in decisions.items():
        if list(saved["y_true"]) != labels:
            raise SystemExit(f"Stage 1 run {run_id}'s saved y_true is not aligned with "
                             "stage1_test.parquet")

    decodes, results = {}, []
    for run_id, s1, s2, why in PIPELINES:
        if s1 not in decisions:
            raise SystemExit(f"no saved test predictions for Stage 1 run {s1}")
        if s2 not in decodes:
            decodes[s2] = stage2_decode(s2, texts, labels, words, args.refresh_cache)
            print(f"stage 2 run {s2}: decoded {len(texts):,} sentences "
                  f"({decodes[s2]['decode_seconds']} s), matches the remote decode on "
                  f"{sum(labels)} positives")

        scored = score_pipeline(gold, labels, list(decisions[s1]["y_pred"]),
                                decodes[s2]["pred"], args.bootstrap)
        logged = stage2_logged_f1(s2)
        oracle = scored["scores"]["oracle"]["entity_f1_strict"]
        if logged is None or abs(logged - oracle) > 1e-9:
            raise SystemExit(f"run {run_id}: oracle F1 {oracle:.6f} does not reproduce "
                             f"Stage 2 run {s2}'s logged {logged}")

        scored.update({"run_id": run_id, "s1": s1, "s2": s2, "why": why,
                       "logged_stage2_f1": logged, "n_sentences": len(texts)})
        results.append(scored)
        d = scored["decomposition"]
        print(f"run {run_id}: oracle {oracle:.4f} -> pipeline "
              f"{scored['scores']['pipeline']['entity_f1_strict']:.4f} | loss {d['total']:.4f} "
              f"= FN {d['fn']:.4f} + FP {d['fp']:.4f}")

    ladder = []
    s2 = results[0]["s2"]
    for s1, saved in sorted(decisions.items()):
        y_pred = list(saved["y_pred"])
        scored = score_pipeline(gold, labels, y_pred, decodes[s2]["pred"], reps=0)
        tp = sum(l and p for l, p in zip(labels, y_pred))
        model, metrics = stage1.get(s1, ("?", {}))
        ladder.append({
            "s1": s1, "model": model, "macro_f1": metrics.get("macro_f1", float("nan")),
            "recall": tp / max(sum(labels), 1), "precision": tp / max(sum(y_pred), 1),
            "pipeline_f1": scored["scores"]["pipeline"]["entity_f1_strict"],
            "loss": scored["decomposition"]["total"],
            "fn": scored["decomposition"]["fn"], "fp": scored["decomposition"]["fp"],
        })

    FIGURES.mkdir(parents=True, exist_ok=True)
    make_chart(results)
    write_report(results, ladder, stage1, args.bootstrap, decodes)
    print(f"wrote {TABLE.relative_to(REPO_ROOT)} and {CHART.relative_to(REPO_ROOT)}")

    if not args.no_log:
        import torch
        import transformers

        from src.utils import log_run

        for r in results:
            so, sp = r["scores"]["oracle"], r["scores"]["pipeline"]
            d, ci, c = r["decomposition"], r["ci"], r["counts"]
            metrics = {
                "entity_f1_strict": sp["entity_f1_strict"],
                "entity_f1_lenient": sp["entity_f1_lenient"],
                "entity_precision_strict": sp["entity_precision_strict"],
                "entity_recall_strict": sp["entity_recall_strict"],
                "overlap_f1": sp["overlap_f1"],
                "oracle_entity_f1_strict": so["entity_f1_strict"],
                "oracle_precision_strict": so["entity_precision_strict"],
                "oracle_recall_strict": so["entity_recall_strict"],
                "fn_only_entity_f1_strict": r["scores"]["fn_only"]["entity_f1_strict"],
                "fp_only_entity_f1_strict": r["scores"]["fp_only"]["entity_f1_strict"],
                "loss_total": d["total"], "loss_from_stage1_fn": d["fn"],
                "loss_from_stage1_fp": d["fp"],
                **({"ci95_loss_total": ci["total"], "ci95_loss_fn": ci["fn"],
                    "ci95_loss_fp": ci["fp"]} if ci else {}),
                "gold_entities": so["gold_entities"],
                "pipeline_pred_entities": sp["pred_entities_strict"],
                **{f"gate_{k}": v for k, v in c.items()},
                "stage2_decode_seconds": decodes[r["s2"]]["decode_seconds"],
            }
            log_run(
                run_id=r["run_id"], stage="1+2", model=f"run{r['s1']}+run{r['s2']}",
                metrics=metrics,
                params={"stage1_run": r["s1"], "stage2_run": r["s2"],
                        "gate": "saved Stage 1 test argmax decisions",
                        "stage2_inference": "local CPU, verified against remote decode",
                        "decomposition": "two-order average (Shapley)",
                        "bootstrap_resamples": args.bootstrap, "bootstrap_seed": SEED,
                        "torch": torch.__version__, "transformers": transformers.__version__},
                seed=SEED, device_count=0,
                notes=f"Step 6.1-6.2 run {r['run_id']}; {r['why']}; inference only.",
            )
        print(f"appended runs {', '.join(r['run_id'] for r in results)} to results/runs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
