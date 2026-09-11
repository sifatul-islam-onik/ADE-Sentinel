"""Steps 5.7-5.9 - the Stage 2 entity-F1 table, CRF ablation, and sanity check.

Reads `results/runs.csv`, the saved tag sequences and the step 5.9 reference
output, and writes:

    results/figures/stage2_results.md   entity-F1 strict + lenient, per label
    results/figures/stage2_crf.png      the CRF ablation figure

**5.7 strict and lenient, side by side.** Rescored here, locally, from the saved
tag sequences rather than trusted from the remote session - and locally seqeval
is installed, so every number is cross-checked against a second implementation.

**5.8 illegal tag sequences with and without the CRF.** Counted, then taken
apart: which transitions they are, and what the CRF changed about the entities
it emits. Entity-F1 alone says the CRF helped; this says how.

**5.9 the external sanity check.** The BIO conversion against the published
corpus statistics, and entity-F1 against a published tagger fine-tuned on the
same corpus (`scripts/reference_tagger.py`).

Every interpretive sentence is conditioned on the numbers it describes, so a
re-run that changes a result changes the prose with it.

Usage:  .venv\\Scripts\\python scripts\\stage2_report.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.stage2_metrics import (  # noqa: E402
    entity_metrics,
    error_breakdown,
    format_entity_table,
    repaired_entities,
    strict_entities,
)
from src.utils import RUNS_CSV  # noqa: E402

FIGURES = REPO_ROOT / "results" / "figures"
TABLE = FIGURES / "stage2_results.md"
CHART = FIGURES / "stage2_crf.png"
PREDICTIONS = REPO_ROOT / "models" / "stage2"
REFERENCE = PREDICTIONS / "ref_scibert_ade" / "metrics.json"

MODEL_NAMES = {
    "9": ("BiLSTM softmax", "per-token independent decisions"),
    "10": ("BiLSTM-CRF", "Viterbi over learned transitions"),
    "11": ("BiomedBERT tagger", "contextual encoder, per-token softmax head"),
}

# PLAN 5.9 / PRD 6.1 fact 4 - the published statistics for the positive portion
# of the ADE corpus. These are TOKEN-level counts under the benchmark's own
# tokenizer, which is why the comparison below is approximate by construction.
PUBLISHED = {
    "sentences": 4272,
    "tokens": 86865,
    "effect_tokens": 12264,     # published as "ADE tags"
    "drug_tokens": 5544,        # published as "Drug tags"
}
# Inside this band a residual is a tokenisation difference; outside it, a bug.
TOLERANCE_PCT = 5.0


def load_runs():
    import pandas as pd

    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} does not exist")

    df = pd.read_csv(RUNS_CSV, dtype={"run_id": str, "stage": str})
    df = df[df.stage == "2"].copy()
    if df.empty:
        raise SystemExit(
            "no Stage 2 runs in results/runs.csv yet - run "
            "notebooks/stage2_remote.ipynb first")

    df = df.drop_duplicates(subset="run_id", keep="last")
    df["metrics"] = df.metrics_json.apply(json.loads)
    df["_order"] = df.run_id.str.extract(r"^(\d+)").astype(float)
    return df.sort_values("_order").reset_index(drop=True)


def load_predictions() -> dict:
    """Saved tag sequences per run, from `models/stage2/run*/test_predictions.json`."""
    saved = {}
    for directory in sorted(PREDICTIONS.glob("run*")):
        path = directory / "test_predictions.json"
        if path.exists():
            run_id = directory.name.split("_")[0].removeprefix("run")
            saved[run_id] = json.loads(path.read_text(encoding="utf-8"))
    return saved


def rescore(saved: dict) -> dict:
    """Every metric recomputed locally from the saved tag sequences.

    Independence is the point: the remote session reported these numbers, and
    this recomputes them from the raw predictions. `entity_metrics` also scores
    with seqeval where it is installed and records whether the two agree.
    """
    return {
        run_id: {**entity_metrics(p["gold"], p["pred"]),
                 **repaired_entities(p["gold"], p["pred"]),
                 **error_breakdown(p["gold"], p["pred"])}
        for run_id, p in saved.items()
    }


def load_reference():
    return json.loads(REFERENCE.read_text(encoding="utf-8")) if REFERENCE.exists() else None


def dev_illegal_curve(run_id: str):
    """The per-epoch dev illegal-transition log saved beside a BiLSTM checkpoint."""
    for directory in PREDICTIONS.glob(f"run{run_id}_*"):
        path = directory / "metrics.json"
        if not path.exists():
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        history = saved.get("history") or []
        if not history or "dev_illegal" not in history[0]:
            continue
        by_epoch = {h["epoch"]: h["dev_illegal"] for h in history}
        best = saved.get("metrics", {}).get("best_epoch")
        return {"first": history[0]["dev_illegal"], "min": min(by_epoch.values()),
                "best_epoch": best, "at_best": by_epoch.get(best),
                "epochs": len(history)}
    return None


def transition_kinds(sequences) -> Counter:
    """Illegal transitions by kind - the same definition `count_illegal_transitions` uses."""
    kinds = Counter()
    for tags in sequences:
        previous = "O"
        for tag in tags:
            if tag.startswith("I-") and previous not in (f"B-{tag[2:]}", f"I-{tag[2:]}"):
                kinds[f"{previous} -> {tag}"] += 1
            previous = tag
    return kinds


def get(row, key, default=None):
    return row["metrics"].get(key, default)


def fmt(value, spec=".4f", missing="-"):
    return missing if value is None else f"{value:{spec}}"


def name_of(run_id: str) -> str:
    return MODEL_NAMES.get(run_id, (run_id, ""))[0]


def make_chart(df) -> bool:
    """Two panels: entity-F1 by run, and illegal transitions by run.

    Side by side because the argument needs both. The F1 panel shows the size of
    the CRF's gain; the illegal-transition panel shows the structural difference
    that explains it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    runs = [r for r in ("9", "10", "11") if r in set(df.run_id)]
    if not runs:
        return False

    by_id = df.set_index("run_id")
    labels = [f"{r}\n{name_of(r)}" for r in runs]
    strict = [get(by_id.loc[r], "entity_f1_strict", 0.0) for r in runs]
    lenient = [get(by_id.loc[r], "entity_f1_lenient", 0.0) for r in runs]
    illegal = [get(by_id.loc[r], "illegal_transitions", 0) for r in runs]

    x = np.arange(len(runs))
    colours = {"9": "#5b8ff9", "10": "#2e7d32", "11": "#6a1b9a"}
    bar_colours = [colours.get(r, "#888888") for r in runs]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)

    width = 0.38
    left.bar(x - width / 2, strict, width, color=bar_colours, edgecolor="white")
    left.bar(x + width / 2, lenient, width, color=bar_colours, alpha=0.45,
             edgecolor="white", hatch="//")
    for xi, v in zip(x - width / 2, strict):
        left.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + width / 2, lenient):
        left.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)

    left.legend(handles=[Patch(facecolor="#6b6b6b", label="strict (IOB2)"),
                         Patch(facecolor="#6b6b6b", alpha=0.45, hatch="//",
                               label="lenient (default)")],
                frameon=False, fontsize=8.5, loc="upper left")
    left.set_xticks(x)
    left.set_xticklabels(labels, fontsize=8.5)
    left.set_ylabel("entity-F1 (test)")
    left.set_title("Entity-F1: strict vs lenient", fontsize=10.5)
    top = max(strict + lenient)
    left.set_ylim(max(0.0, min(strict + lenient) - 0.12), top + 0.09)
    left.spines[["top", "right"]].set_visible(False)
    left.grid(axis="y", alpha=0.25, lw=0.6)

    bars = right.bar(x, illegal, 0.55, color=bar_colours, edgecolor="white")
    for bar, v in zip(bars, illegal):
        right.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + max(illegal + [1]) * 0.02,
                   f"{int(v)}", ha="center", fontsize=9)
    right.set_xticks(x)
    right.set_xticklabels(labels, fontsize=8.5)
    right.set_ylabel("illegal tag transitions (test)")
    right.set_title("Structurally impossible sequences\n(`I-X` not preceded by `B-X` or `I-X`)",
                    fontsize=10.5)
    right.set_ylim(0, max(illegal + [1]) * 1.18)
    right.spines[["top", "right"]].set_visible(False)
    right.grid(axis="y", alpha=0.25, lw=0.6)

    fig.tight_layout()
    fig.savefig(CHART)
    plt.close(fig)
    return True


def write_table(df, saved, rescored, has_chart, reference) -> None:
    by_id = df.set_index("run_id")

    lines = [
        "# Stage 2 results (steps 5.7-5.9, runs 9-11)",
        "",
        "Generated by `scripts/stage2_report.py`. Entity-level scoring is redone "
        "**locally, from the saved tag sequences** - the remote session is trusted "
        "for the training, not for the numbers.",
        "",
        "All three runs share the frozen split, the project tokenizer, the BIO "
        "converter and seed 42, and are scored over an identical word-level token "
        "stream. Run 11 predicts at subword level and is read back to words (the "
        "first piece of each word) before scoring, or its entity counts would not "
        "be comparable with the BiLSTMs'.",
        "",
        "## Entity-F1 (test)",
        "",
        "| Run | Model | Strict F1 | Lenient F1 | Lenient - strict | Overlap F1 | Strict P | Strict R | Illegal transitions |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"| {row.run_id} | {name_of(row.run_id)} | **{fmt(get(row, 'entity_f1_strict'))}** | "
            f"{fmt(get(row, 'entity_f1_lenient'))} | "
            f"{fmt(get(row, 'strict_lenient_gap'), '+.4f')} | "
            f"{fmt(rescored.get(row.run_id, {}).get('overlap_f1'))} | "
            f"{fmt(get(row, 'entity_precision_strict'), '.3f')} | "
            f"{fmt(get(row, 'entity_recall_strict'), '.3f')} | "
            f"{get(row, 'illegal_transitions', '-')} |"
        )
    lines += [
        "",
        "Three scores, three different questions. **Strict** is exact-match IOB2 "
        "entity-F1, the headline. **Lenient** is seqeval's default: boundaries must "
        "still be exact, but a malformed sequence such as `O I-DRUG` is repaired into "
        "an entity first. **Overlap** credits a predicted entity that overlaps a gold "
        "entity of the same label even when its boundaries are wrong - the PRD's "
        "partial-match stretch goal. Lenient is easily mistaken for a partial-match "
        "score, and `PLAN.md` step 5.7 makes that mistake; only the overlap column "
        "measures tolerance to boundary errors.",
        "",
    ]

    if has_chart:
        lines += [f"![Stage 2 entity-F1 and CRF ablation]({CHART.name})", ""]

    lines += lenient_section(df, rescored)
    lines += crf_section(by_id, rescored)
    lines += transitions_section(saved)
    lines += per_label_section(df, by_id)
    lines += sanity_section(by_id, reference)
    lines += verification_section(df, rescored)

    TABLE.write_text("\n".join(lines), encoding="utf-8")


def lenient_section(df, rescored) -> list[str]:
    """Step 5.7 - why the two scoring modes disagree, and in which direction."""
    runs = [r for r in df.run_id if r in rescored]
    if not runs:
        return []

    lines = [
        "## What lenient scoring adds (step 5.7)",
        "",
        "**Strict** requires a well-formed IOB2 entity: `O I-DRUG` scores nothing. "
        "**Lenient**, seqeval's default, reads the same sequence as a DRUG entity. "
        "Both readings open an entity at every `B-X`, so a model's strict entities "
        "are always a subset of its lenient ones, and the two scores differ only by "
        "the *repaired* entities - exactly one per illegal transition.",
        "",
        "With k repairs of which c are correct, lenient F1 beats strict F1 exactly "
        "when c / k > strict F1 / 2 (`repaired_entities` in `src/stage2_metrics.py`; "
        "the test suite checks the criterion in exact arithmetic). The sign of the "
        "gap is therefore not a property of the scoring mode. It measures how often "
        "a model's malformed output was right anyway.",
        "",
        "| Run | Model | Repairs (k) | Correct (c) | c / k | Needed to help | Lenient vs strict |",
        "|---|---|---|---|---|---|---|",
    ]

    effect = {}
    for run_id in runs:
        m = rescored[run_id]
        k, c = m["repaired_entities"], m["repaired_correct"]
        needed = m["entity_f1_strict"] / 2
        gap = m["entity_f1_lenient"] - m["entity_f1_strict"]
        effect[run_id] = (k, c, needed, gap)
        direction = "higher" if gap > 0 else "lower" if gap < 0 else "equal"
        lines.append(
            f"| {run_id} | {name_of(run_id)} | {k} | {c} | "
            f"{f'{c / k:.1%}' if k else '-'} | > {needed:.1%} | {direction} ({gap:+.4f}) |")
    lines.append("")

    if "9" in effect:
        k, c, needed, gap = effect["9"]
        if k and gap < 0:
            lines += [
                "**Lenient scoring penalises the softmax tagger rather than flattering "
                f"it.** Run 9 emitted {k} malformed entities and {c} of them "
                f"({c / k:.0%}) were right - far short of the {needed:.0%} a repair "
                f"needs in order to help - so the lenient reading costs it {gap:+.4f}. "
                "The intuitive expectation is the reverse: that a scorer which forgives "
                "malformed output favours the model producing the most of it. Reporting "
                "both columns is what shows which of the two holds here.",
                "",
            ]
        elif k and gap > 0:
            lines += [
                f"Lenient scoring flatters the softmax tagger: {c} of its {k} repairs "
                f"({c / k:.0%}) were correct, above the {needed:.0%} needed, and worth "
                f"{gap:+.4f}. Strict F1 is the fair basis for the comparison with run 10.",
                "",
            ]

    others = [r for r in runs if r != "9" and effect[r][0] and effect[r][3] < 0]
    for run_id in others:
        k, c, needed, gap = effect[run_id]
        lines += [
            f"Run {run_id} behaves the same way: {c} of {k} repairs correct "
            f"({c / k:.0%}), so lenient scoring lowers it by {abs(gap):.4f}.",
            "",
        ]

    if "10" in effect and abs(effect["10"][3]) < 0.001:
        k, _, _, gap = effect["10"]
        lines += [
            f"For run 10 the question barely arises: {k} repair{'' if k == 1 else 's'} "
            f"in total, and the two modes agree to within {abs(gap):.4f}.",
            "",
        ]
    return lines


MECHANISM = (
    "The mechanism is worth stating precisely, because it is easy to overclaim. "
    "`pytorch-crf` forbids nothing: its transition matrix is initialised uniformly "
    "and every transition stays reachable. The gold sequences contain no illegal "
    "transition at all, so training drives those transition scores down until "
    "Viterbi rarely chooses a path through them. The constraint is **learned from "
    "the data, not imposed by the architecture**. A per-token softmax has no way to "
    "learn it: it scores each position independently, so no transition ever "
    "appears in its objective."
)


def crf_section(by_id, rescored) -> list[str]:
    """Step 5.8 - the CRF ablation, stated from the numbers in hand."""
    if "9" not in by_id.index or "10" not in by_id.index:
        return []

    softmax, crf = by_id.loc["9"], by_id.loc["10"]

    def row(label, key, spec=".4f", delta_spec="+.4f"):
        a, b = get(softmax, key), get(crf, key)
        diff = "-" if a is None or b is None else f"{b - a:{delta_spec}}"
        return f"| {label} | {fmt(a, spec)} | {fmt(b, spec)} | {diff} |"

    illegal_softmax = get(softmax, "illegal_transitions", 0)
    illegal_crf = get(crf, "illegal_transitions", 0)
    secs_softmax = get(softmax, "train_seconds") or 0
    secs_crf = get(crf, "train_seconds") or 0
    f1_delta = get(crf, "entity_f1_strict", 0) - get(softmax, "entity_f1_strict", 0)

    lines = [
        "## The CRF ablation (runs 9 vs 10, step 5.8)",
        "",
        "Identical embedding, architecture, hyperparameters and seed. **`--crf` is "
        "the only flag that differs**, so everything below is attributable to "
        "structured prediction.",
        "",
        "| | Run 9 softmax | Run 10 CRF | Difference |",
        "|---|---|---|---|",
        row("Entity-F1 strict", "entity_f1_strict"),
        row("Precision strict", "entity_precision_strict", ".3f", "+.3f"),
        row("Recall strict", "entity_recall_strict", ".3f", "+.3f"),
        row("Entity-F1 lenient", "entity_f1_lenient"),
        f"| Illegal transitions | {illegal_softmax} | {illegal_crf} | "
        f"{illegal_crf - illegal_softmax:+d} |",
        f"| Sentences affected | {get(softmax, 'illegal_sentences', 0)} | "
        f"{get(crf, 'illegal_sentences', 0)} | "
        f"{get(crf, 'illegal_sentences', 0) - get(softmax, 'illegal_sentences', 0):+d} |",
        f"| Training time | {secs_softmax:.0f} s | {secs_crf:.0f} s | "
        f"x{secs_crf / max(secs_softmax, 1e-9):.1f} |",
        f"| Selected epoch / epochs run | {get(softmax, 'best_epoch', '-')} / "
        f"{get(softmax, 'epochs_run', '-')} | {get(crf, 'best_epoch', '-')} / "
        f"{get(crf, 'epochs_run', '-')} | |",
        "",
    ]

    if illegal_crf == 0 and illegal_softmax > 0:
        lines += [f"**The CRF eliminated illegal sequences entirely** - "
                  f"{illegal_softmax} down to zero.", "", MECHANISM, ""]
    elif illegal_crf < illegal_softmax:
        cut = 1 - illegal_crf / illegal_softmax
        lines += [
            f"**The CRF cut illegal transitions by {cut:.0%}** - {illegal_softmax} to "
            f"{illegal_crf} - without eliminating them.",
            "",
            MECHANISM,
            "",
            f"The {illegal_crf} that remain are what \"learned, not imposed\" predicts: a "
            "path through a heavily penalised transition is still chosen when the "
            "emission scores favour it strongly enough. Report the reduction as what "
            "it is - a large decrease, not a guarantee.",
            "",
        ]
    else:
        lines += [
            f"The CRF did **not** reduce illegal sequences ({illegal_softmax} -> "
            f"{illegal_crf}). That contradicts the architectural argument and should be "
            "diagnosed before the result is reported - check that `use_crf` was "
            "actually set and that decoding used Viterbi rather than argmax.",
            "",
        ]

    soft_curve, crf_curve = dev_illegal_curve("9"), dev_illegal_curve("10")
    if soft_curve and crf_curve and crf_curve["at_best"] is not None:
        lines += [
            "**The training logs show the constraint being learned.** On dev, run 10 "
            f"emitted {crf_curve['first']} illegal transitions after its first epoch, "
            f"{crf_curve['at_best']} at the epoch selected ({crf_curve['best_epoch']}), "
            f"and as few as {crf_curve['min']} over {crf_curve['epochs']} epochs. Run 9 "
            f"started at {soft_curve['first']}, was at {soft_curve['at_best']} at its "
            f"selected epoch ({soft_curve['best_epoch']}), and never went below "
            f"{soft_curve['min']}. The softmax tagger does emit fewer malformed "
            "sequences as its per-token predictions improve, but it plateaus; the CRF "
            "keeps pushing the count toward zero because transitions are part of what "
            "it optimises.",
            "",
        ]

    if abs(f1_delta) < 0.01:
        lines += [
            f"Entity-F1 moved only {f1_delta:+.4f}. **Both halves of that belong in the "
            "report.** On F1 alone the CRF looks like an expensive no-op; the "
            "illegal-sequence count is where its contribution is visible.",
            "",
        ]
    else:
        lines += [
            f"Entity-F1 moved {f1_delta:+.4f} as well, so here the CRF pays on the "
            f"metric axis too - at {secs_crf / max(secs_softmax, 1e-9):.1f}x the "
            "training time.",
            "",
        ]

    if "9" in rescored and "10" in rescored:
        lines += crf_mechanism(by_id, rescored["9"], rescored["10"])
    return lines


def crf_mechanism(by_id, a, b) -> list[str]:
    """What the CRF changed about the entities it emits, from `error_breakdown`."""
    kinds = [("exact", "pred_exact"), ("boundary error", "pred_boundary"),
             ("wrong type", "pred_type"), ("spurious", "pred_spurious")]

    lines = [
        "### What the CRF changed about the entities it emits",
        "",
        "Every predicted entity classified against gold under the strict reading "
        "(`error_breakdown`): exact; overlapping a gold entity of the same label with "
        "the wrong boundaries; overlapping only one of the other label; or "
        "overlapping nothing.",
        "",
        "| Predicted entities | Run 9 | Run 10 | Difference |",
        "|---|---|---|---|",
    ]
    for label, key in kinds:
        lines.append(f"| {label} | {a[key]:,} | {b[key]:,} | {b[key] - a[key]:+d} |")
    total_a = sum(a[key] for _, key in kinds)
    total_b = sum(b[key] for _, key in kinds)
    lines += [f"| **total** | **{total_a:,}** | **{total_b:,}** | **{total_b - total_a:+d}** |",
              ""]

    d_exact = b["pred_exact"] - a["pred_exact"]
    d_wrong = (total_b - b["pred_exact"]) - (total_a - a["pred_exact"])
    p_a = get(by_id.loc["9"], "entity_precision_strict", 0)
    p_b = get(by_id.loc["10"], "entity_precision_strict", 0)
    r_a = get(by_id.loc["9"], "entity_recall_strict", 0)
    r_b = get(by_id.loc["10"], "entity_recall_strict", 0)

    if d_wrong < 0 and abs(d_exact) * 5 <= abs(d_wrong):
        plural = {"boundary error": "boundary errors", "wrong type": "wrong-type entities",
                  "spurious": "spurious entities"}
        parts = [f"{a[key] - b[key]} fewer {plural[label]}"
                 for label, key in kinds[1:] if b[key] < a[key]]
        lines += [
            "**The CRF does not find more entities; it stops emitting broken ones.** "
            f"Exact matches moved by {d_exact:+d}. Wrong predictions fell by "
            f"{abs(d_wrong)}: {', '.join(parts)}. That is why the gain is "
            f"almost entirely precision ({p_a:.3f} to {p_b:.3f}) while recall barely "
            f"moves ({r_a:.3f} to {r_b:.3f}): structured decoding is acting as a "
            "filter on malformed and fragmentary output, not as a source of new "
            "detections.",
            "",
        ]
    else:
        lines += [
            f"Exact matches changed by {d_exact:+d} and wrong predictions by "
            f"{d_wrong:+d}; precision {p_a:.3f} to {p_b:.3f}, recall {r_a:.3f} to "
            f"{r_b:.3f}.",
            "",
        ]
    return lines


def transitions_section(saved) -> list[str]:
    """Which illegal transitions each tagger emits, and why they cluster where they do."""
    runs = [r for r in ("9", "10", "11") if r in saved]
    if not runs:
        return []

    kinds = {r: transition_kinds(saved[r]["pred"]) for r in runs}
    order = sorted(set().union(*kinds.values()),
                   key=lambda k: -sum(kinds[r][k] for r in runs))
    if not order:
        return []

    lines = [
        "### Which transitions are illegal",
        "",
        "| Transition | " + " | ".join(f"Run {r}" for r in runs) + " |",
        "|---|" + "---|" * len(runs),
    ]
    for kind in order:
        lines.append(f"| `{kind}` | " + " | ".join(str(kinds[r][kind]) for r in runs) + " |")

    total = sum(sum(k.values()) for k in kinds.values())
    orphan_effect = sum(kinds[r]["O -> I-EFFECT"] for r in runs)

    shape = Counter()
    for tags in saved[runs[0]]["gold"]:
        for start, end, label in strict_entities(tags):
            shape[(label, end - start > 1)] += 1
    effect_multi = shape[("EFFECT", True)] / max(shape[("EFFECT", True)] + shape[("EFFECT", False)], 1)
    drug_multi = shape[("DRUG", True)] / max(shape[("DRUG", True)] + shape[("DRUG", False)], 1)

    lines.append("")
    if total and orphan_effect / total > 0.5:
        lines += [
            f"**{orphan_effect / total:.0%} of all illegal transitions across the three "
            "taggers are an `I-EFFECT` straight after `O`** - a model continuing an "
            "EFFECT it never opened. That is where the structure lives: "
            f"{effect_multi:.0%} of gold EFFECT entities span more than one token, "
            f"against {drug_multi:.0%} of DRUG entities, so EFFECT is the label on "
            "which B-versus-I decisions are made at all.",
            "",
        ]
    return lines


def per_label_section(df, by_id) -> list[str]:
    lines = ["## Per-entity breakdown", "",
             "DRUG and EFFECT are not equally hard, and the report should say which "
             "is which rather than average them away.", ""]
    for _, row in df.iterrows():
        lines += [f"**Run {row.run_id} - {name_of(row.run_id)}**", ""]
        lines += format_entity_table(row["metrics"])
        lines += [""]

    runs = [r for r in ("9", "10", "11") if r in by_id.index]
    drug = [get(by_id.loc[r], "drug_f1") for r in runs]
    effect = [get(by_id.loc[r], "effect_f1") for r in runs]
    if runs and None not in drug + effect and all(e < d for d, e in zip(drug, effect)):
        lines += [
            "**EFFECT is the harder label for every tagger** - F1 "
            f"{' / '.join(f'{e:.3f}' for e in effect)} against "
            f"{' / '.join(f'{d:.3f}' for d in drug)} for DRUG (runs "
            f"{', '.join(runs)}).",
            "",
        ]
    if "9" in by_id.index and "10" in by_id.index:
        d_drug = get(by_id.loc["10"], "drug_f1", 0) - get(by_id.loc["9"], "drug_f1", 0)
        d_effect = get(by_id.loc["10"], "effect_f1", 0) - get(by_id.loc["9"], "effect_f1", 0)
        lines += [
            f"Between runs 9 and 10, DRUG F1 moved {d_drug:+.4f} and EFFECT F1 "
            f"{d_effect:+.4f}"
            + (" - **the CRF's gain is an EFFECT gain**, which is what the transition "
               "table above predicts." if abs(d_effect) > 3 * abs(d_drug) and d_effect > 0
               else "."),
            "",
        ]
    return lines


def corpus_tag_stats() -> dict:
    """Recount the whole Stage 2 corpus at token level, for the 5.9 comparison.

    Computed from the committed splits through the same converter the training
    scripts use, so this compares what the models actually saw against the
    published figures - not what a separate accounting says they should have.
    """
    import pandas as pd

    from src.bio_convert import ConversionStats, to_bio

    counts, sentences = Counter(), 0
    stats = ConversionStats()

    for split in ("train", "dev", "test"):
        df = pd.read_parquet(REPO_ROOT / "data" / "splits" / f"stage2_{split}.parquet")
        for text, raw in zip(df.text, df.spans):
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            _, tags = to_bio(
                text, [(int(s), int(e), str(label)) for s, e, label in parsed],
                stats=stats, strict=False)
            counts.update(tags)
            sentences += 1

    return {
        "sentences": sentences,
        "tokens": sum(counts.values()),
        "effect_tokens": counts["B-EFFECT"] + counts["I-EFFECT"],
        "drug_tokens": counts["B-DRUG"] + counts["I-DRUG"],
        "spans": stats.spans,
        "entities_tagged": stats.entities_tagged,
        "spans_snapped": stats.spans_snapped,
        "spans_unmatched": stats.spans_unmatched,
        "spans_dropped_nested": stats.spans_dropped_nested,
    }


def sanity_section(by_id, reference) -> list[str]:
    """Step 5.9 - check the numbers against something external."""
    ours = corpus_tag_stats()

    lines = [
        "## Sanity check against external figures (step 5.9)",
        "",
        "An entity-F1 can look entirely reasonable while measuring nothing - the "
        "failure PLAN F5 describes, where the converter produces empty labels and "
        "training proceeds happily on them. Two external checks guard against it: "
        "the BIO conversion against the corpus statistics published for this "
        "benchmark (PRD 6.1 fact 4), and our scores against a published tagger "
        "trained on the same corpus.",
        "",
        "### The conversion against the published statistics",
        "",
        "| Quantity | This project | Published | Difference |",
        "|---|---|---|---|",
    ]

    labels = {
        "sentences": "Sentences with ADE relations",
        "tokens": "Tokens",
        "effect_tokens": "EFFECT tokens (published as \"ADE tags\")",
        "drug_tokens": "DRUG tokens (published as \"Drug tags\")",
    }
    pct, worst = {}, 0.0
    for key, label in labels.items():
        mine, published = ours[key], PUBLISHED[key]
        pct[key] = 100.0 * (mine - published) / published
        worst = max(worst, abs(pct[key]))
        lines.append(f"| {label} | {mine:,} | {published:,} | {pct[key]:+.2f}% |")
    lines.append("")

    if worst <= TOLERANCE_PCT:
        lines += [
            f"**Every count lands within {worst:.1f}% of the published figure.** "
            "Agreement on four counts at once, computed from the committed splits "
            "through the converter the training scripts use, is not something a "
            "broken offset alignment produces by accident - a systematic alignment "
            "error moves entity token counts wholesale, not by a percent or two.",
            "",
        ]
    else:
        lines += [
            f"**At least one count is {worst:.1f}% from the published figure, outside "
            f"the {TOLERANCE_PCT:.0f}% band.** Diagnose the conversion before any "
            "Stage 2 number is reported.",
            "",
        ]

    fewer = ours["tokens"] < PUBLISHED["tokens"]
    lines += [
        "**Why the residuals are not zero.** The published counts are token-level "
        "under the benchmark's own tokenizer. The domain tokenizer from step 2.1 "
        "keeps `5-fluorouracil`, `TNF-alpha` and `20 mg/kg` whole where a naive "
        "tokenizer splits them, so "
        + (f"fewer tokens overall ({ours['tokens']:,} against {PUBLISHED['tokens']:,}) "
           "is the expected direction - the property `results/figures/tokenizer_table.md` "
           "measures directly."
           if fewer else
           f"more tokens than published ({ours['tokens']:,} against "
           f"{PUBLISHED['tokens']:,}) is the opposite of the expected direction and is "
           "not explained by the tokenizer.")
        + f" The two label counts miss in opposite directions (EFFECT "
        f"{pct['effect_tokens']:+.2f}%, DRUG {pct['drug_tokens']:+.2f}%) and are not "
        "decomposed further here: the published figures do not document their "
        "tokenizer or their treatment of nested spans, so any mechanism offered for a "
        "residual this size would be a guess.",
        "",
        "| Conversion diagnostic | Count |",
        "|---|---|",
        f"| Annotated spans | {ours['spans']:,} |",
        f"| Entities tagged | {ours['entities_tagged']:,} |",
        f"| Spans needing boundary snapping | {ours['spans_snapped']:,} |",
        f"| Spans matching no token | **{ours['spans_unmatched']:,}** |",
        f"| Spans dropped as nested/overlapping | {ours['spans_dropped_nested']:,} |",
        "",
        f"**{ours['spans_unmatched']:,} unmatched spans** is the number PLAN F5 is "
        "about. The PRD's reference implementation would have returned all-`O` for "
        "any span it could not align and said nothing; here the count is asserted and "
        "reported, so \"zero\" is a measurement rather than an assumption.",
        "",
    ]

    lines += reference_section(by_id, reference)
    return lines


def reference_section(by_id, reference) -> list[str]:
    lines = ["### Against a published tagger", ""]
    if reference is None:
        return lines + [
            "Not yet run. `python scripts/reference_tagger.py` scores "
            "`jsylee/scibert_scivocab_uncased-finetuned-ner` on the same sentences; "
            "until it has, the entity-F1 figures above have no external point of "
            "comparison.",
            "",
        ]

    scores, run11 = reference["scores"], reference.get("run11_local", {})
    ref = {split: scores[split]["entity_f1_strict"] for split in ("train", "dev", "test")}
    ours_test = get(by_id.loc["11"], "entity_f1_strict") if "11" in by_id.index else None
    mapping = reference.get("label_map", {})
    purity = reference.get("label_purity_on_train", {})

    lines += [
        f"`{reference['reference']}` (Hub revision `{reference['revision'][:12]}`) is "
        "SciBERT fine-tuned for DRUG/EFFECT tagging on ADE Corpus v2 - the reference "
        "PRD 6.1 names. `scripts/reference_tagger.py` runs it over exactly the word "
        "stream runs 9-11 were scored on, with the same strict scorer.",
        "",
        "| Model | Trained on | Our train | Our dev | Our test |",
        "|---|---|---|---|---|",
        f"| Reference SciBERT | ADE Corpus v2, split undocumented | {ref['train']:.4f} | "
        f"{ref['dev']:.4f} | **{ref['test']:.4f}** |",
    ]
    if all(split in run11 for split in ("train", "dev", "test")):
        lines.append(
            f"| Run 11 BiomedBERT (ours) | our train split only | "
            f"{run11['train']['entity_f1_strict']:.4f} *(seen)* | "
            f"{run11['dev']['entity_f1_strict']:.4f} | "
            f"**{run11['test']['entity_f1_strict']:.4f}** |")
    lines.append("")

    if purity:
        low, high = min(purity.values()), max(purity.values())
        if low >= 0.8:
            lines += [
                "**Our gold tags and scorer are not silently broken.** A tagger trained "
                "elsewhere, with its own tokenisation and BIO conversion, agrees with our "
                "gold tags token by token: of the tokens it assigns each label, "
                f"{low:.0%}-{high:.0%} carry the matching gold tag (table below), and it "
                f"reaches {ref['test']:.4f} entity-F1 against them. A misaligned "
                "conversion - the PLAN F5 failure - would break exactly that token-level "
                "agreement.",
                "",
            ]
        else:
            lines += [
                "**One of the reference's labels agrees with our gold tags on only "
                f"{low:.0%} of its tokens.** Either its annotation convention differs "
                "sharply from ours or our conversion is wrong; resolve that before using "
                "either number.",
                "",
            ]

    seen = run11.get("train", {}).get("entity_f1_strict")
    unseen = run11.get("test", {}).get("entity_f1_strict")
    if seen is not None and unseen is not None:
        lines += [
            "**Its documentation cannot rule out that it has seen our test sentences.** "
            "ADE Corpus v2 is published as a single `train` split and the model card "
            "documents no held-out portion, so a model fine-tuned on it may have trained "
            "on sentences our split holds out. Run 11 shows what having trained on a "
            f"sentence is worth to a BERT-class tagger: {seen:.4f} on its own training "
            f"sentences against {unseen:.4f} on held-out ones.",
            "",
        ]
        spread = max(ref.values()) - min(ref.values())
        if abs(ref["test"] - seen) < abs(ref["test"] - unseen):
            lines += [
                f"The reference's {ref['test']:.4f} on our test sits nearer the first - "
                "consistent with contamination - so its test score must not be compared "
                "with run 11's.",
                "",
            ]
        elif spread < 0.02:
            lines += [
                "The reference shows no such lift anywhere: it scores "
                f"{min(ref.values()):.4f}-{max(ref.values()):.4f} on all three of our "
                f"splits, far below the {seen:.4f} run 11 reaches on sentences it was "
                "trained on. Whatever the reference was trained on, contamination is not "
                "inflating its number.",
                "",
            ]
        else:
            lines += [f"The reference's {ref['test']:.4f} on our test sits nearer the "
                      "held-out figure.", ""]

    if ours_test is not None:
        i_share = [purity[k] for k, tag in mapping.items() if tag.startswith("I-") and k in purity]
        b_share = [purity[k] for k, tag in mapping.items() if tag.startswith("B-") and k in purity]
        text = (
            "**It is a sanity check, not a competitor.** Run 11 scores "
            f"{ours_test - ref['test']:+.4f} above it on the same test sentences, but the "
            "reference is being scored under *our* tokenisation and *our* span-to-BIO "
            "conventions, which it was not trained on."
        )
        if i_share and b_share and max(i_share) < min(b_share):
            text += (
                f" Its `I-` labels agree with ours least ({min(i_share):.0%}-"
                f"{max(i_share):.0%}, against {min(b_share):.0%}-{max(b_share):.0%} for "
                "`B-`), which is where a difference in boundary convention would show."
            )
        text += (" How much of the gap is convention rather than model quality cannot be "
                 "separated here.")
        lines += [text, ""]

    if mapping:
        lines += [
            "Its config names its labels only `LABEL_0`..`LABEL_4`. They were identified on "
            "**our train split** by the gold tag each most often lands on - test played no "
            "part - and form a bijection onto our inventory:",
            "",
            "| Raw label | Identified as | Share of its train tokens on that tag |",
            "|---|---|---|",
        ]
        for raw_label in sorted(mapping):
            lines.append(f"| `{raw_label}` | `{mapping[raw_label]}` | "
                         f"{purity.get(raw_label, 0):.1%} |")
        lines.append("")

    over = reference.get("sentences_over_subword_budget", {})
    if any(over.values()):
        lines += [f"Sentences over the {reference.get('max_len')}-subword budget: "
                  + ", ".join(f"{k} {v}" for k, v in over.items()) + ".", ""]
    return lines


def verification_section(df, rescored: dict) -> list[str]:
    """Do the logged numbers survive independent recomputation?"""
    if not rescored:
        return []

    lines = [
        "## Independent recomputation",
        "",
        "Every run's entity-F1, recomputed locally from its saved tag sequences and "
        "compared with the value the remote session logged. The remote runner could "
        "not install seqeval, so the last column is the first time these predictions "
        "have been scored by a second implementation (strict and lenient F1 plus "
        "per-label strict F1).",
        "",
        "| Run | Logged strict F1 | Recomputed | Agrees | seqeval cross-check |",
        "|---|---|---|---|---|",
    ]

    for _, row in df.iterrows():
        logged = get(row, "entity_f1_strict")
        again = rescored.get(row.run_id, {})
        if "entity_f1_strict" not in again:
            lines.append(f"| {row.run_id} | {fmt(logged)} | - | predictions not found | - |")
            continue
        agrees = logged is not None and abs(logged - again["entity_f1_strict"]) < 1e-9
        lines.append(f"| {row.run_id} | {fmt(logged)} | {fmt(again['entity_f1_strict'])} | "
                     f"{'yes' if agrees else '**NO**'} | {again.get('seqeval_crosscheck', '-')} |")

    lines += [""]
    return lines


def main() -> int:
    df = load_runs()
    saved = load_predictions()
    rescored = rescore(saved)
    reference = load_reference()
    has_chart = make_chart(df)
    write_table(df, saved, rescored, has_chart, reference)

    print(f"{len(df)} Stage 2 run(s): {', '.join(df.run_id)}")
    print(f"recomputed from predictions: {', '.join(sorted(rescored)) or 'none found'}")
    print(f"reference tagger: {'found' if reference else 'not run yet'}")
    print(f"wrote {TABLE.relative_to(REPO_ROOT)}")
    if has_chart:
        print(f"wrote {CHART.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
