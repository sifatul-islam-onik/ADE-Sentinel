"""Step 4.5 - the Stage 1 results table and the four-bar embedding chart.

Reads `results/runs.csv` and writes:

    results/figures/stage1_results.md      every Stage 1 run, one table
    results/figures/stage1_embeddings.png  the four-bar chart (PRD 7.4)

The chart is the single most important figure in the report, so it is generated
from the run log rather than assembled by hand. Two consequences worth the
constraint: it cannot drift from the numbers in `runs.csv`, and re-running it
after the remote session is a one-liner rather than an afternoon.

Runs the report needs but the log may not have yet are simply absent from the
table - this script is meant to be run repeatedly as rows arrive, including
locally after only runs 1-2 exist.

Usage:  .venv\\Scripts\\python scripts\\stage1_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.metrics import format_confusion  # noqa: E402
from src.utils import RUNS_CSV  # noqa: E402

FIGURES = REPO_ROOT / "results" / "figures"
TABLE = FIGURES / "stage1_results.md"
CHART = FIGURES / "stage1_embeddings.png"

# The four bars, in ablation order. Labels are what the chart shows.
ABLATION = [
    ("3", "E0_random", "E0\nrandom"),
    ("4", "E1", "E1\nGloVe"),
    ("5", "E2", "E2\nour W2V"),
    ("6", "E3", "E3\nour FastText"),
]

TIERS = {
    "1": "T1 generative", "2": "T2 discriminative", "2b": "T2 discriminative",
    "3": "T3 BiLSTM", "4": "T3 BiLSTM", "5": "T3 BiLSTM", "6": "T3 BiLSTM",
    "3u": "T3 BiLSTM", "4u": "T3 BiLSTM", "5u": "T3 BiLSTM", "6u": "T3 BiLSTM",
    "7": "T4 transformer", "8": "T5 transformer",
}


def load_runs():
    """Stage 1 rows, most recent row per run_id wins.

    Last-wins rather than first: re-running a run to fix a bug should supersede
    the broken row, and `log_run` only ever appends.
    """
    import pandas as pd

    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} does not exist - no runs logged yet")

    df = pd.read_csv(RUNS_CSV, dtype={"run_id": str, "stage": str})
    df = df[df.stage == "1"].copy()
    if df.empty:
        raise SystemExit("no Stage 1 runs in results/runs.csv yet")

    df = df.drop_duplicates(subset="run_id", keep="last")
    df["metrics"] = df.metrics_json.apply(json.loads)
    df["tier"] = df.run_id.map(TIERS).fillna("")

    # Sort numerically by run number, keeping the 'u' variants beside their base.
    df["_order"] = df.run_id.str.extract(r"^(\d+)").astype(float)
    df["_suffix"] = df.run_id.str.extract(r"^\d+(.*)$").fillna("")
    return df.sort_values(["_order", "_suffix"]).reset_index(drop=True)


def get(row, key, default=None):
    return row["metrics"].get(key, default)


def fmt(value, spec=".4f", missing="-"):
    return missing if value is None else f"{value:{spec}}"


def make_chart(df) -> bool:
    """The four-bar chart: macro-F1 by embedding, frozen and fine-tuned.

    Returns False if the ablation runs are not in the log yet, so the table can
    still be written locally before the remote session happens.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_id = df.set_index("run_id")
    frozen = [(label, by_id.loc[rid]) for rid, _, label in ABLATION if rid in by_id.index]
    if not frozen:
        return False

    tuned = [(label, by_id.loc[f"{rid}u"]) for rid, _, label in ABLATION
             if f"{rid}u" in by_id.index]

    labels = [label for label, _ in frozen]
    frozen_f1 = [get(row, "macro_f1") for _, row in frozen]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)

    # E0 is the floor, so it is greyed: the eye should read the other three
    # against it rather than treating all four as peers.
    colours = ["#9e9e9e", "#5b8ff9", "#2e7d32", "#1b5e20"][: len(labels)]

    from matplotlib.patches import Patch

    handles = []
    if tuned:
        width = 0.38
        tuned_f1 = [get(row, "macro_f1") for _, row in tuned]
        ax.bar(x - width / 2, frozen_f1, width, color=colours, edgecolor="white")
        ax.bar(x + width / 2, tuned_f1, width, color=colours, alpha=0.45,
               edgecolor="white", hatch="//")
        for xi, value in zip(x - width / 2, frozen_f1):
            ax.text(xi, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)
        for xi, value in zip(x + width / 2, tuned_f1):
            ax.text(xi, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)
        handles += [
            Patch(facecolor="#6b6b6b", label="frozen embeddings"),
            Patch(facecolor="#6b6b6b", alpha=0.45, hatch="//", label="fine-tuned"),
        ]
    else:
        bars = ax.bar(x, frozen_f1, 0.6, color=colours, edgecolor="white")
        for bar, value in zip(bars, frozen_f1):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008,
                    f"{value:.3f}", ha="center", fontsize=9)

    # The sparse baseline as a reference line: a BiLSTM below it is a diagnosis,
    # not a result, and the reader should be able to see that at a glance.
    # It goes in the legend rather than as floating text, which would land on
    # top of whichever bar happened to be tallest.
    baseline = df[df.run_id == "2"]
    if not baseline.empty:
        value = get(baseline.iloc[0], "macro_f1")
        handles.append(ax.axhline(
            value, ls="--", lw=1.2, color="#c62828", zorder=0,
            label=f"run 2, TF-IDF LogReg ({value:.3f})"))

    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("macro-F1 (test)")
    ax.set_title("Stage 1: BiLSTM macro-F1 by embedding\n"
                 "identical architecture, seed and hyperparameters", fontsize=11)

    values = frozen_f1 + ([get(r, "macro_f1") for _, r in tuned] if tuned else [])
    low = min(values + ([get(baseline.iloc[0], "macro_f1")] if not baseline.empty else []))
    ax.set_ylim(max(0.0, low - 0.08), max(values) + 0.075)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)

    fig.tight_layout()
    fig.savefig(CHART)
    plt.close(fig)
    return True


def write_table(df, has_chart: bool) -> None:
    lines = [
        "# Stage 1 results (step 4.5, runs 1-8)",
        "",
        "Generated by `scripts/stage1_report.py` from `results/runs.csv`. Every number "
        "here is a straight read of the run log - nothing is transcribed by hand.",
        "",
        "All runs share the frozen split (`data/splits/stage1_*.parquet`), seed 42, and "
        "the project tokenizer. Selection is on dev; test is scored once.",
        "",
        "## All runs",
        "",
        "| Run | Tier | Model | Embedding | Macro-F1 | ADE P | ADE R | ADE F1 | PR-AUC | Acc. | Dev F1 | Eff. batch | Devices |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    best_id, best_f1 = None, -1.0
    for _, row in df.iterrows():
        f1 = get(row, "macro_f1")
        if f1 is not None and f1 > best_f1:
            best_id, best_f1 = row.run_id, f1

    for _, row in df.iterrows():
        f1 = get(row, "macro_f1")
        mark = "**" if row.run_id == best_id else ""
        batch = row.get("effective_batch")
        devices = row.get("device_count")
        lines.append(
            f"| {row.run_id} | {row.tier} | `{row.model}` | {row.embedding or '-'} | "
            f"{mark}{fmt(f1)}{mark} | {fmt(get(row, 'precision_ade'), '.3f')} | "
            f"{fmt(get(row, 'recall_ade'), '.3f')} | {fmt(get(row, 'f1_ade'), '.3f')} | "
            f"{fmt(get(row, 'pr_auc'), '.3f')} | {fmt(get(row, 'accuracy'), '.3f')} | "
            f"{fmt(get(row, 'dev_macro_f1'), '.4f')} | "
            f"{'-' if batch != batch or batch == '' else int(float(batch))} | "
            f"{'-' if devices != devices or devices == '' else int(float(devices))} |"
        )

    lines += [
        "",
        f"Best Stage 1 run so far: **{best_id}** at macro-F1 {best_f1:.4f}. That is "
        "the model Phase 6.1 carries into the end-to-end pipeline (run 12).",
        "",
    ]

    by_id = df.set_index("run_id")

    # Only worth explaining the suffix if such rows are actually in the table.
    if any(df.run_id.str.endswith("u")):
        lines += [
            "Runs with a `u` suffix are the fine-tuned-embedding condition of the "
            "same ablation - see below.",
            "",
        ]
    present = [(rid, name, label) for rid, name, label in ABLATION if rid in by_id.index]

    if present:
        lines += [
            "## The embedding ablation (runs 3-6)",
            "",
            "The project's headline experiment (PRD section 7). Identical architecture, "
            "seed, hyperparameters, split and device count; **the embedding matrix is "
            "the only thing that changes**.",
            "",
            "| Run | Embedding | Macro-F1 | vs E0 floor | vs E1 GloVe |",
            "|---|---|---|---|---|",
        ]
        floor = get(by_id.loc["3"], "macro_f1") if "3" in by_id.index else None
        glove = get(by_id.loc["4"], "macro_f1") if "4" in by_id.index else None

        for rid, name, label in present:
            f1 = get(by_id.loc[rid], "macro_f1")
            d_floor = f"{f1 - floor:+.4f}" if floor is not None and rid != "3" else "-"
            d_glove = f"{f1 - glove:+.4f}" if glove is not None and rid not in ("3", "4") else "-"
            lines.append(
                f"| {rid} | {label.replace(chr(10), ' ')} | "
                f"{fmt(f1)} | {d_floor} | {d_glove} |"
            )

        if has_chart:
            lines += [
                "",
                f"![Stage 1 macro-F1 by embedding]({CHART.name})",
                "",
                "The dashed line is run 2, the TF-IDF logistic regression. It is on the "
                "chart deliberately: a BiLSTM below it is a sign of an undertrained "
                "model, not evidence about embeddings, and the reader should be able to "
                "see that without reading the table.",
            ]

        tuned = [rid for rid, _, _ in ABLATION if f"{rid}u" in by_id.index]
        if tuned:
            lines += [
                "",
                "### Frozen vs fine-tuned",
                "",
                "The two conditions answer different questions. **Frozen** measures the "
                "vectors themselves: nothing about the embedding layer changes during "
                "training, so a difference in F1 is a difference in what the "
                "pretraining corpus taught. **Fine-tuned** asks whether that advantage "
                "survives task supervision - E0's random rows can learn from the 14.6k "
                "training sentences, and the gap it has to close is the question.",
                "",
                "| Run | Embedding | Frozen | Fine-tuned | Delta |",
                "|---|---|---|---|---|",
            ]
            for rid, name, _ in ABLATION:
                if rid not in by_id.index or f"{rid}u" not in by_id.index:
                    continue
                a = get(by_id.loc[rid], "macro_f1")
                b = get(by_id.loc[f"{rid}u"], "macro_f1")
                lines.append(f"| {rid} / {rid}u | {name} | {fmt(a)} | {fmt(b)} | "
                             f"{b - a:+.4f} |")
            lines += [
                "",
                "If fine-tuning closes the gap between E1 and E2/E3, the domain "
                "advantage is an *initialisation* advantage that task supervision can "
                "partly recover. That is a real finding and belongs in the report as "
                "one - PRD section 12 asks for negative and partial results to be "
                "explained rather than buried.",
            ]

    transformers = df[df.run_id.isin(["7", "8"])]
    if len(transformers) == 2:
        general, domain = (transformers[transformers.run_id == r].iloc[0] for r in ("7", "8"))
        delta = get(domain, "macro_f1") - get(general, "macro_f1")
        lines += [
            "",
            "## Domain vs general, replayed at the transformer level (runs 7-8)",
            "",
            f"| Run | Model | Pretraining corpus | Macro-F1 |",
            "|---|---|---|---|",
            f"| 7 | `bert-base-uncased` | books + Wikipedia | {fmt(get(general, 'macro_f1'))} |",
            f"| 8 | BiomedBERT | PubMed abstracts | {fmt(get(domain, 'macro_f1'))} |",
            "",
            f"Domain minus general: **{delta:+.4f}**. Identical architecture, identical "
            "recipe, identical split - the pretraining corpus is the only difference, "
            "exactly as it is between E1 and E2/E3 in the ablation above. Whether the "
            "sign agrees in both places is the symmetry PRD 8.1 is after.",
        ]

    lines += ["", "## Confusion matrices (test)", ""]
    for _, row in df.iterrows():
        cm = get(row, "confusion_matrix")
        if not cm:
            continue
        lines += [f"**Run {row.run_id} - `{row.model}`"
                  f"{f' ({row.embedding})' if row.embedding else ''}**", ""]
        lines += format_confusion(cm)
        lines += [""]

    lines += [
        "## Provenance",
        "",
        "| Run | Commit | Dirty | Dataset version | Epochs | LR |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        dirty = "yes" if str(row.get("git_dirty")) == "1" else "no"
        lines.append(
            f"| {row.run_id} | `{row.git_commit}` | {dirty} | "
            f"{row.dataset_version if row.dataset_version == row.dataset_version and row.dataset_version else '-'} | "
            f"{row.epochs if row.epochs == row.epochs else '-'} | "
            f"{row.lr if row.lr == row.lr else '-'} |"
        )

    lines += [
        "",
        "`git_dirty = yes` means the working tree held uncommitted changes when the "
        "run executed, so the commit hash does not fully describe the code that "
        "produced the number. Treat those rows as provisional and re-run them from a "
        "clean tree before the report is final.",
        "",
    ]

    TABLE.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    df = load_runs()
    has_chart = make_chart(df)
    write_table(df, has_chart)

    print(f"{len(df)} Stage 1 run(s): {', '.join(df.run_id)}")
    print(f"wrote {TABLE.relative_to(REPO_ROOT)}")
    if has_chart:
        print(f"wrote {CHART.relative_to(REPO_ROOT)}")
    else:
        print("chart skipped - runs 3-6 are not in results/runs.csv yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
