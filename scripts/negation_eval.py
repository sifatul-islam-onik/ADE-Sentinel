"""Step 6.3 - run 13: every Stage 1 tier on the negation and hedging subset.

    python scripts/negation_eval.py

PRD 8.3's hypothesis: bag-of-words models collapse on negated and hedged
sentences because they cannot represent scope, the BiLSTM partially recovers,
and BERT handles them best. This tests it rather than illustrating it.

**The subset is selected by rule** (`src/challenge_set.py`, PLAN F6), from text
alone, and frozen to data/splits/stage1_test_cues.parquet the first time it is
built. A later run that selects differently stops instead of overwriting.

**Cue sentences are compared with no-cue sentences**, not with the full test
set. The full set contains the subset, so the two would share most of their
errors and any difference would be diluted by construction. The two groups are
disjoint, so each is bootstrapped independently.

**Prevalence-free metrics beside macro-F1.** If cue sentences have a different
class mix, macro-F1 moves for a reason unrelated to scope. Per-class recall does
not depend on the class mix, so a drop in ADE recall or not-ADE recall inside
the subset is the cleaner evidence.

Every Stage 1 run's saved test predictions must reproduce its logged macro-F1
before it is scored here. The sparse baselines need
`scripts/baseline_predictions.py` first.

Writes results/figures/negation_results.md and negation_ladder.png, and appends
one row per Stage 1 run as run `13-<run>`, stage `1-challenge` (skip with
--no-log). The stage label keeps these rows out of the Stage 1 report tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import challenge_set  # noqa: E402
from src.utils import RUNS_CSV  # noqa: E402

SPLITS = REPO_ROOT / "data" / "splits"
FROZEN = SPLITS / "stage1_test_cues.parquet"
STAGE1 = REPO_ROOT / "models" / "stage1"
FIGURES = REPO_ROOT / "results" / "figures"
TABLE = FIGURES / "negation_results.md"
CHART = FIGURES / "negation_ladder.png"

TIERS = {"1": "T1", "2": "T2", "2b": "T2", "3": "T3", "4": "T3", "5": "T3", "6": "T3",
         "3u": "T3", "4u": "T3", "5u": "T3", "6u": "T3", "7": "T4", "8": "T5"}
ORDER = ("1", "2", "2b", "3", "4", "5", "6", "3u", "4u", "5u", "6u", "7", "8")
PRD_RUNS = ("1", "2", "6", "7", "8")        # PRD section 9, run 13
METRICS = ("macro_f1", "recall_ade", "recall_not_ade")
SEED = 42


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--no-log", action="store_true", help="do not append to results/runs.csv")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def stage1_runs() -> dict:
    import pandas as pd

    df = pd.read_csv(RUNS_CSV, dtype={"run_id": str, "stage": str})
    df = df[df.stage == "1"].drop_duplicates(subset="run_id", keep="last")
    return {row.run_id: row for row in df.itertuples()}


def saved_predictions() -> dict:
    import numpy as np

    out = {}
    for path in sorted(STAGE1.glob("run*/test_predictions.npz")):
        run_id = path.parent.name.split("_")[0].removeprefix("run")
        saved = np.load(path)
        out[run_id] = (saved["y_true"].astype(int), saved["y_pred"].astype(int))
    return out


def frozen_subset(test):
    """The rule's selection, frozen on first use and verified on every later one."""
    import pandas as pd

    selected = challenge_set.select(test.text, test.label)
    if not FROZEN.exists():
        selected.to_parquet(FROZEN, index=False)
        print(f"froze the challenge subset to {FROZEN.relative_to(REPO_ROOT)}")
        return selected, True

    frozen = pd.read_parquet(FROZEN)
    same = (len(frozen) == len(selected)
            and list(frozen.text) == list(selected.text)
            and list(frozen.negation) == list(selected.negation)
            and list(frozen.hedging) == list(selected.hedging))
    if not same:
        raise SystemExit(
            "the cue rule no longer selects the frozen subset - the cue lists or the "
            "test split changed since run 13 was defined. Restore them; do not delete "
            f"{FROZEN.name} to make this pass (PLAN F6).")
    return frozen, False


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def scores(y_true, y_pred) -> dict:
    """Binary Stage 1 scores - the same definitions as `src.metrics`."""
    import numpy as np

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {"n": int(len(y_true)), "ade": int(y_true.sum())}
    f1s = []
    for label, name in ((1, "ade"), (0, "not_ade")):
        tp = int(((y_true == label) & (y_pred == label)).sum())
        predicted, actual = int((y_pred == label).sum()), int((y_true == label).sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.update({f"precision_{name}": precision, f"recall_{name}": recall, f"f1_{name}": f1})
        f1s.append(f1)
    out["macro_f1"] = sum(f1s) / 2
    out["accuracy"] = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return out


def _batched(y_true, y_pred, metric):
    """A metric over a batch of resamples: y arrays are (resamples, n)."""
    import numpy as np

    def per_class(label):
        tp = ((y_true == label) & (y_pred == label)).sum(axis=1)
        predicted = (y_pred == label).sum(axis=1)
        actual = (y_true == label).sum(axis=1)
        zeros = np.zeros(len(tp))
        precision = np.divide(tp, predicted, out=zeros.copy(), where=predicted > 0)
        recall = np.divide(tp, actual, out=zeros.copy(), where=actual > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=zeros.copy(),
                       where=(precision + recall) > 0)
        return recall, f1

    if metric == "recall_ade":
        return per_class(1)[0]
    if metric == "recall_not_ade":
        return per_class(0)[0]
    return (per_class(1)[1] + per_class(0)[1]) / 2


def bootstrap_difference(y_true, y_pred, group_a, group_b, metrics, reps, seed=SEED) -> dict:
    """95% interval for metric(group_a) - metric(group_b), each group resampled independently."""
    import numpy as np

    rng = np.random.default_rng(seed)
    a, b = np.flatnonzero(group_a), np.flatnonzero(group_b)
    draws = {m: [] for m in metrics}
    for start in range(0, reps, 500):
        size = min(500, reps - start)
        ia = a[rng.integers(0, len(a), size=(size, len(a)))]
        ib = b[rng.integers(0, len(b), size=(size, len(b)))]
        for m in metrics:
            draws[m].append(_batched(y_true[ia], y_pred[ia], m) - _batched(y_true[ib], y_pred[ib], m))
    return {m: (float(np.percentile(np.concatenate(v), 2.5)),
                float(np.percentile(np.concatenate(v), 97.5))) for m, v in draws.items()}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def diff(r, metric="macro_f1", group="cue"):
    return r["groups"][group][metric] - r["groups"]["no_cue"][metric]


def excludes_zero(interval) -> bool:
    return interval[0] > 0 or interval[1] < 0


def make_chart(results) -> bool:
    """Macro-F1 and ADE recall, with and without a cue, for PRD section 9's runs plus run 3.

    The separate negation and hedging groups are left out on purpose: their class
    mixes differ from the no-cue group's, so bars for them would show a prevalence
    effect as if it were a scope effect.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    runs = [r for r in ("1", "2", "3", "6", "7", "8") if r in results]
    if not runs:
        return False

    x = np.arange(len(runs))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=200)
    for ax, metric, title in ((axes[0], "macro_f1", "Macro-F1"),
                              (axes[1], "recall_ade", "ADE recall - the class mix cannot move it")):
        none = [results[r]["groups"]["no_cue"][metric] for r in runs]
        cue = [results[r]["groups"]["cue"][metric] for r in runs]
        ax.bar(x - width / 2, none, width, color="#607d8b", label="no cue", edgecolor="white")
        ax.bar(x + width / 2, cue, width, color="#c62828", label="negation or hedging cue",
               edgecolor="white")
        for xi, v in zip(x - width / 2, none):
            ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)
        for xi, v in zip(x + width / 2, cue):
            ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"run {r}\n{results[r]['tier']}" for r in runs], fontsize=8)
        ax.set_ylim(max(0.0, min(none + cue) - 0.1), 1.0)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(CHART)
    plt.close(fig)
    return True


def write_report(results, subset, y_true, groups, reps, newly_frozen) -> None:
    import numpy as np

    L = [
        "# Negation and hedging (step 6.3, run 13)",
        "",
        "Generated by `scripts/negation_eval.py` from the saved Stage 1 test predictions and "
        "the frozen challenge subset. Nothing is trained; every run's predictions reproduce "
        "its logged test macro-F1 before they are scored here.",
        "",
        "## The subset",
        "",
        "Selected by rule from the Stage 1 test split (`src/challenge_set.py`), from sentence "
        "text alone, and frozen to `data/splits/stage1_test_cues.parquet` (PLAN F6). It is "
        "selection, not annotation - every label is the corpus's own.",
        "",
        "| Group | Sentences | ADE | ADE rate |",
        "|---|---|---|---|",
    ]
    for name, label in (("full", "full test split"), ("negation", "contains a negation cue"),
                        ("hedging", "contains a hedging cue"),
                        ("cue", "**either - the challenge subset**"), ("no_cue", "neither")):
        mask = groups[name]
        L.append(f"| {label} | {int(mask.sum()):,} | {int(y_true[mask].sum()):,} | "
                 f"{y_true[mask].mean():.1%} |")

    both = int((groups["negation"] & groups["hedging"]).sum())
    rate_cue, rate_none = y_true[groups["cue"]].mean(), y_true[groups["no_cue"]].mean()
    L += [
        "",
        f"{both} sentences contain both kinds of cue. PRD 8.3 suggested hand-picking 50-80 "
        f"sentences; the rule selects {int(groups['cue'].sum()):,}, which turns the comparison "
        "below into a measurement with intervals rather than an anecdote.",
        "",
    ]
    if abs(rate_cue - rate_none) >= 0.03:
        L += [
            f"**The class mix differs between the groups**: {rate_cue:.1%} ADE among cue "
            f"sentences against {rate_none:.1%} without. Macro-F1 moves with the class mix for "
            "reasons unrelated to scope, which is why per-class recalls - which do not depend "
            "on it - are reported beside it below.",
            "",
        ]
    else:
        L += [
            f"The class mix is similar in both groups ({rate_cue:.1%} ADE with a cue, "
            f"{rate_none:.1%} without), so a macro-F1 difference between them is not a "
            "prevalence artefact.",
            "",
        ]
    skewed = [(name, y_true[groups[name]].mean()) for name in ("negation", "hedging")
              if abs(y_true[groups[name]].mean() - rate_none) >= 0.03]
    if skewed:
        L += [
            "That does not hold for the two cue types separately: "
            + " and ".join(f"{rate:.1%} of {name}-cue sentences are ADE" for name, rate in skewed)
            + f", against {rate_none:.1%} with no cue. The Negation and Hedging macro-F1 columns "
            "below therefore mix a scope effect with a class-mix effect and should not be read on "
            "their own; the combined contrast and the per-class recalls are the evidence.",
            "",
        ]

    for kind, column in (("Negation", "negation_cues"), ("Hedging", "hedging_cues")):
        counts = Counter(c for cues in subset[column] if cues for c in cues.split("|"))
        L += [f"| {kind} cue | Sentences |", "|---|---|"]
        L += [f"| `{cue}` | {n:,} |" for cue, n in counts.most_common(8)]
        L.append("")

    L += [
        "**When the cue lists were fixed.** They were written during the Phase 4 "
        "documentation, after the Stage 1 test predictions existed and after four sentences "
        "every Stage 1 model misclassifies had been read; report section 7.7 then used them "
        "to show those universally-missed sentences are not enriched for either cue. They "
        "have not been changed since, and `tests/test_challenge_set.py` pins them. The "
        "selection reads text only - no prediction or label chose a sentence."
        + (" The subset file was frozen by this run." if newly_frozen else ""),
        "",
        "## Macro-F1 with and without cues",
        "",
        "Positive differences mean a run does *better* on cue sentences. Brackets are 95% "
        f"bootstrap intervals for the difference, each group resampled independently "
        f"({reps:,} resamples). Runs PRD section 9 lists for run 13 are in bold.",
        "",
        "| Run | Tier | Model | Full | Negation | Hedging | No cue | Cue - no cue |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run_id, r in results.items():
        g = r["groups"]
        lo, hi = r["ci"]["macro_f1"]
        name = f"**{run_id}**" if run_id in PRD_RUNS else run_id
        L.append(f"| {name} | {r['tier']} | `{r['model']}` | {g['full']['macro_f1']:.4f} | "
                 f"{g['negation']['macro_f1']:.4f} | {g['hedging']['macro_f1']:.4f} | "
                 f"{g['no_cue']['macro_f1']:.4f} | {diff(r):+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    L.append("")

    L += [
        "PRD 8.3 asks for accuracy. It is logged for every group in the run 13 rows of "
        "`results/runs.csv` but not tabulated: with four sentences in five not-ADE, accuracy "
        "rewards exactly the failure this section looks for - a model that stops predicting ADE "
        "when it sees a cue.",
        "",
    ]
    if CHART.exists():
        L += [f"![Macro-F1 and ADE recall with and without cues]({CHART.name})", ""]

    L += [
        "## Per-class recall, which the class mix cannot move",
        "",
        "| Run | Tier | ADE recall: cue / no cue | Difference | not-ADE recall: cue / no cue | Difference |",
        "|---|---|---|---|---|---|",
    ]
    for run_id, r in results.items():
        g, ci = r["groups"], r["ci"]
        name = f"**{run_id}**" if run_id in PRD_RUNS else run_id
        L.append(
            f"| {name} | {r['tier']} | {g['cue']['recall_ade']:.3f} / {g['no_cue']['recall_ade']:.3f} | "
            f"{diff(r, 'recall_ade'):+.3f} [{ci['recall_ade'][0]:+.3f}, {ci['recall_ade'][1]:+.3f}] | "
            f"{g['cue']['recall_not_ade']:.3f} / {g['no_cue']['recall_not_ade']:.3f} | "
            f"{diff(r, 'recall_not_ade'):+.3f} [{ci['recall_not_ade'][0]:+.3f}, "
            f"{ci['recall_not_ade'][1]:+.3f}] |")
    L.append("")

    significant = [run_id for run_id, r in results.items() if excludes_zero(r["ci"]["macro_f1"])]
    recall_drop = [run_id for run_id, r in results.items() if r["ci"]["recall_ade"][1] < 0]
    L += [
        f"**{len(recall_drop)} of {len(results)} runs lose ADE recall on cue sentences with an "
        "interval entirely below zero**"
        + (f" (runs {', '.join(recall_drop)})" if recall_drop else "")
        + f", against {len(significant)} whose macro-F1 difference excludes zero"
        + (f" (runs {', '.join(significant)})." if significant else "."),
        "",
    ]
    offset = [r for r in recall_drop if results[r]["ci"]["recall_not_ade"][0] > 0]
    if offset:
        runs_text = offset[0] if len(offset) == 1 else ", ".join(offset[:-1]) + " and " + offset[-1]
        L += [
            "**Macro-F1 understates the effect because the two classes can move in opposite "
            f"directions.** In run{'s' if len(offset) > 1 else ''} {runs_text}, ADE recall falls "
            "on cue sentences while not-ADE recall *rises*, both intervals clear of zero: a cue "
            "word pushes the model toward not-ADE, and macro-F1 averages a loss on one class with "
            "a gain on the other.",
            "",
        ]

    # ---- PRD 8.3's hypothesis, one claim at a time ---------------------------------
    prd = [r for r in PRD_RUNS if r in results]
    sparse = [r for r in ("1", "2") if r in prd]
    neural = [r for r in prd if r not in sparse]

    def recall_change(r):
        lo, hi = results[r]["ci"]["recall_ade"]
        return f"{diff(results[r], 'recall_ade'):+.3f} [{lo:+.3f}, {hi:+.3f}]"

    L += [
        "## PRD 8.3's hypothesis, tested",
        "",
        "PRD 8.3 expects the bag-of-words models to collapse on these sentences, the BiLSTM to "
        "partially recover, and BERT to handle them best. Each claim is checked below with "
        "**ADE recall** as the primary evidence: it is the class a scope error hides - a negated "
        "or hedged ADE report read as no ADE - and the class mix cannot move it.",
        "",
    ]

    if sparse and neural:
        sparse_mean = sum(diff(results[r], "recall_ade") for r in sparse) / len(sparse)
        neural_mean = sum(diff(results[r], "recall_ade") for r in neural) / len(neural)
        sparse_sig = all(results[r]["ci"]["recall_ade"][1] < 0 for r in sparse)
        neural_clear = all(not excludes_zero(results[r]["ci"]["recall_ade"]) for r in neural)
        kept = min(results[r]["groups"]["cue"]["recall_ade"] for r in sparse)
        claim = (
            "1. **Do the bag-of-words models collapse?** On cue sentences ADE recall changes by "
            + "; ".join(f"{recall_change(r)} for run {r}" for r in sparse)
            + " - against " + "; ".join(f"{recall_change(r)} for run {r}" for r in neural) + ". "
        )
        if sparse_sig and neural_clear and abs(sparse_mean) > 2 * abs(neural_mean):
            claim += (
                "**The direction the PRD predicts holds, and it is significant**: the sparse "
                "models miss clearly more ADE sentences once a cue is present, while the neural "
                f"tiers' losses average {abs(neural_mean) / abs(sparse_mean):.0%} of the sparse "
                "models' and cannot be told apart from zero. 'Collapse' still overstates it: the "
                f"sparse models keep an ADE recall of at least {kept:.2f} on cue sentences."
            )
        elif sparse_sig:
            claim += ("The sparse models' losses are significant, but the neural tiers' are not "
                      "clearly smaller, so the contrast the PRD draws is not established.")
        else:
            claim += ("**Not as predicted**: the sparse models' ADE-recall losses are not "
                      "significant.")
        claim += (" On macro-F1 the same comparison is muted ("
                  + "; ".join(f"{diff(results[r]):+.4f} for run {r}" for r in prd)
                  + "), for the reason given above.")
        L += [claim, ""]

    if "3" in results and "6" in results:
        worst = min(results, key=lambda r: diff(results[r], "recall_ade"))
        claim = (
            "2. **Does the BiLSTM partially recover?** That depends on its embeddings, not on its "
            "being a BiLSTM. With random embeddings (run 3 - not in PRD section 9's list, but the "
            f"same architecture) ADE recall on cue sentences changes by {recall_change('3')}; with "
            f"our frozen FastText vectors (run 6) by {recall_change('6')}."
        )
        if (results["3"]["ci"]["recall_ade"][1] < 0
                and not excludes_zero(results["6"]["ci"]["recall_ade"])):
            claim += (
                " The recovery is real, and it belongs to the pretrained representation rather "
                "than to the recurrent architecture"
                + (f" - run 3 loses more ADE recall than any of the {len(results)} runs, sparse "
                   "models included." if worst == "3" else ".")
            )
            pretrained = [r for r in ("4", "5", "6") if r in results]
            if all(not excludes_zero(results[r]["ci"]["recall_ade"]) for r in pretrained):
                claim += (
                    " Every frozen BiLSTM with pretrained vectors - GloVe (run 4) included - "
                    "avoids a significant loss, so on this question it is pretraining as such, "
                    "not domain pretraining, that matters; section 7.5's domain advantage shows "
                    "up in the level of each score, not in its robustness to cues."
                )
        L += [claim, ""]

    if prd:
        best_level = max(prd, key=lambda r: results[r]["groups"]["cue"]["recall_ade"])
        best_macro = max(prd, key=lambda r: results[r]["groups"]["cue"]["macro_f1"])
        claim = "3. **Does BERT handle them best?** "
        if best_level in ("7", "8") and best_macro == best_level:
            claim += (f"On level, yes: run {best_level} has both the highest ADE recall "
                      f"({results[best_level]['groups']['cue']['recall_ade']:.3f}) and the "
                      f"highest macro-F1 ({results[best_level]['groups']['cue']['macro_f1']:.4f}) "
                      "on cue sentences.")
        else:
            claim += (f"On level, run {best_level} has the highest ADE recall on cue sentences "
                      f"and run {best_macro} the highest macro-F1.")
        if neural:
            spanning = all(not excludes_zero(results[r]["ci"]["recall_ade"]) for r in neural)
            claim += (" On robustness the evidence is weaker: the neural tiers' ADE-recall changes ("
                      + "; ".join(f"{diff(results[r], 'recall_ade'):+.3f} for run {r}" for r in neural)
                      + ")"
                      + (" all have intervals spanning zero, so none is shown to be affected by "
                         "cues, and none is shown to be less affected than another."
                         if spanning else " do not all span zero; see the table above."))
        L += [claim, ""]

        rank_cue = sorted(prd, key=lambda r: -results[r]["groups"]["cue"]["macro_f1"])
        rank_none = sorted(prd, key=lambda r: -results[r]["groups"]["no_cue"]["macro_f1"])
        L += [
            "4. **Do the cues reorder the tiers?** "
            + ("No - the macro-F1 ranking is the same with and without cues: "
               + " > ".join(f"run {r}" for r in rank_cue) + "."
               if rank_cue == rank_none else
               "Yes - with cues: " + " > ".join(f"run {r}" for r in rank_cue)
               + "; without: " + " > ".join(f"run {r}" for r in rank_none) + "."),
            "",
        ]

    widths = [r["ci"]["macro_f1"][1] - r["ci"]["macro_f1"][0] for r in results.values()]
    L += [
        "## Threats to validity",
        "",
        "| Threat | Status |",
        "|---|---|",
        "| Subset chosen after seeing which sentences models fail | controlled - selected by a "
        "text-only rule and frozen; the cue lists' history is stated above in full |",
        "| A cue word is not a cue in scope | not controlled - surface patterns; the subset is "
        "enriched for scope phenomena, not made of them |",
        "| Class mix moving macro-F1 | controlled - per-class recalls reported beside it, and "
        "ADE recall used as the primary evidence |",
        "| Negation-only and hedging-only columns | flagged - their class mixes differ from the "
        "no-cue group's; the combined contrast is the one to read |",
        f"| Subset too small to separate tiers | measured - difference intervals are "
        f"{min(widths):.3f}-{max(widths):.3f} macro-F1 wide |",
        "| Full-test comparison diluted by overlap | controlled - cue sentences compared with "
        "the disjoint no-cue group |",
        "| Single seed | **not controlled** - one model per tier, seed 42 |",
        "",
    ]

    TABLE.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)

    import numpy as np
    import pandas as pd

    test = pd.read_parquet(SPLITS / "stage1_test.parquet")
    subset, newly_frozen = frozen_subset(test)
    y_true = test.label.values.astype(int)

    negation = subset.negation.values.astype(bool)
    hedging = subset.hedging.values.astype(bool)
    groups = {"full": np.ones(len(test), dtype=bool), "negation": negation,
              "hedging": hedging, "cue": negation | hedging, "no_cue": ~(negation | hedging)}

    runs, predictions = stage1_runs(), saved_predictions()
    missing = [r for r in ORDER if r in runs and r not in predictions]
    if missing:
        raise SystemExit(f"no saved test predictions for runs {', '.join(missing)} - run "
                         "scripts/baseline_predictions.py for the sparse baselines")

    results = {}
    for run_id in [r for r in ORDER if r in predictions]:
        yt, yp = predictions[run_id]
        if not np.array_equal(yt, y_true):
            raise SystemExit(f"run {run_id}'s predictions are not aligned with stage1_test")

        per_group = {name: scores(yt[mask], yp[mask]) for name, mask in groups.items()}
        logged = json.loads(runs[run_id].metrics_json)["macro_f1"]
        if abs(per_group["full"]["macro_f1"] - logged) > 1e-9:
            raise SystemExit(f"run {run_id}: saved predictions give macro-F1 "
                             f"{per_group['full']['macro_f1']:.6f}, logged {logged:.6f}")

        ci = bootstrap_difference(yt, yp, groups["cue"], groups["no_cue"], METRICS, args.bootstrap)
        ci_negation = bootstrap_difference(yt, yp, negation, groups["no_cue"], ("macro_f1",),
                                           args.bootstrap)
        ci_hedging = bootstrap_difference(yt, yp, hedging, groups["no_cue"], ("macro_f1",),
                                          args.bootstrap)
        results[run_id] = {
            "tier": TIERS.get(run_id, "?"), "model": runs[run_id].model,
            "embedding": runs[run_id].embedding, "groups": per_group, "ci": ci,
            "ci_negation": ci_negation["macro_f1"], "ci_hedging": ci_hedging["macro_f1"],
        }
        print(f"run {run_id:>3}: full {per_group['full']['macro_f1']:.4f} | cue "
              f"{per_group['cue']['macro_f1']:.4f} | no cue {per_group['no_cue']['macro_f1']:.4f} "
              f"| diff {diff(results[run_id]):+.4f} [{ci['macro_f1'][0]:+.4f}, {ci['macro_f1'][1]:+.4f}]")

    FIGURES.mkdir(parents=True, exist_ok=True)
    make_chart(results)
    write_report(results, subset, y_true, groups, args.bootstrap, newly_frozen)
    print(f"wrote {TABLE.relative_to(REPO_ROOT)} and {CHART.relative_to(REPO_ROOT)}")

    if not args.no_log:
        from src.utils import log_run

        for run_id, r in results.items():
            g = r["groups"]
            metrics = {
                "macro_f1": g["cue"]["macro_f1"],
                **{f"{group}_{key}": value for group in ("full", "negation", "hedging", "cue", "no_cue")
                   for key, value in g[group].items()},
                "cue_minus_no_cue_macro_f1": diff(r),
                "ci95_cue_minus_no_cue_macro_f1": r["ci"]["macro_f1"],
                "ci95_cue_minus_no_cue_recall_ade": r["ci"]["recall_ade"],
                "ci95_cue_minus_no_cue_recall_not_ade": r["ci"]["recall_not_ade"],
                "negation_minus_no_cue_macro_f1": diff(r, group="negation"),
                "ci95_negation_minus_no_cue_macro_f1": r["ci_negation"],
                "hedging_minus_no_cue_macro_f1": diff(r, group="hedging"),
                "ci95_hedging_minus_no_cue_macro_f1": r["ci_hedging"],
            }
            log_run(
                run_id=f"13-{run_id}", stage="1-challenge", model=r["model"],
                embedding=r["embedding"] if isinstance(r["embedding"], str) else "",
                metrics=metrics,
                params={"source_run": run_id, "subset": str(FROZEN.relative_to(REPO_ROOT)),
                        "cue_module": "src/challenge_set.py", "comparison": "cue vs no cue",
                        "bootstrap_resamples": args.bootstrap, "bootstrap_seed": SEED},
                seed=SEED, device_count=0,
                notes=f"Step 6.3 run 13 on Stage 1 run {run_id}; macro_f1 is the cue subset's.",
            )
        print(f"appended {len(results)} run-13 rows to results/runs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
