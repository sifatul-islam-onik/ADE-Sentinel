"""Writes report/analysis_documentation.md - report sections 9-11 (Phase 6).

Sections 9 (pipeline integration and error propagation), 10 (negation and
hedging) and 11 (error analysis) of the PRD section 13 outline. Each is
assembled from the result file its script generates, so the report and the
result files cannot disagree: a re-run of any Phase 6 script changes this
document by regenerating it, never by editing it.

What this adds is the section numbering, figure paths that resolve from
`report/`, and a summary whose figures are read from `results/runs.csv` and
`results/error_taxonomy.json` rather than typed.

Usage:  .venv\\Scripts\\python scripts\\document_analysis.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import RUNS_CSV  # noqa: E402

RESULTS = REPO_ROOT / "results"
FIGURES = RESULTS / "figures"
TAXONOMY_JSON = RESULTS / "error_taxonomy.json"
SAMPLE_CSV = RESULTS / "error_sample.csv"
OUT = REPO_ROOT / "report" / "analysis_documentation.md"

SECTIONS = (
    ("9", "Pipeline integration and error propagation", FIGURES / "pipeline_results.md"),
    ("10", "Negation and hedging", FIGURES / "negation_results.md"),
    ("11", "Error analysis", RESULTS / "error_taxonomy.md"),
)
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)/\\]+)\)")

PROPERTY = {"negation": "a negation cue", "hedging": "a hedging cue",
            "multi_drug": "two or more drugs", "abbreviation": "an abbreviation in an entity"}
POPULATION = {"gate": "gate misses", "alarm": "gate false alarms", "stage2": "Stage 2 errors"}


def renumber(path: Path, number: str) -> list[str]:
    """A result file's body, with its `##` headings numbered under this section."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    folder = path.parent.relative_to(REPO_ROOT).as_posix()

    out, sub = [], 0
    for line in lines:
        if line.startswith("## "):
            sub += 1
            out.append(f"### {number}.{sub} {line[3:]}")
        elif line.startswith("### "):
            out.append("#" + line)
        else:
            out.append(IMAGE.sub(lambda m: f"![{m.group(1)}](../{folder}/{m.group(2)})", line))
    return out


def run_metrics() -> dict:
    import pandas as pd

    df = pd.read_csv(RUNS_CSV, dtype={"run_id": str, "stage": str})
    df = df.drop_duplicates(subset="run_id", keep="last")
    return {row.run_id: json.loads(row.metrics_json) for row in df.itertuples()}


def interval(ci, spec=".4f") -> str:
    return f"[{ci[0]:{spec}}, {ci[1]:{spec}}]" if ci else "not computed"


def manual_categories():
    """(filled, total) rows of the 6.4 sample's manual_category column."""
    if not SAMPLE_CSV.exists():
        return None
    with SAMPLE_CSV.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return sum(bool((r.get("manual_category") or "").strip()) for r in rows), len(rows)


def summary(runs: dict) -> list[str]:
    L = ["## Summary", ""]

    m = runs.get("12")
    if m:
        total = m["loss_total"]
        L.append(
            "- **Error propagation (section 9, run 12).** Putting the best Stage 1 gate in front "
            "of the best Stage 2 tagger lowers strict entity-F1 from "
            f"{m['oracle_entity_f1_strict']:.4f} (oracle) to {m['entity_f1_strict']:.4f}. Stage 1 "
            f"false positives account for {m['loss_from_stage1_fp']:.4f} of the {total:.4f} lost "
            f"({m['loss_from_stage1_fp'] / total:.0%}, 95% interval "
            f"{interval(m.get('ci95_loss_fp'))}) and its false negatives for "
            f"{m['loss_from_stage1_fn']:.4f} ({interval(m.get('ci95_loss_fn'))}).")

    challenge = {rid: runs[f"13-{rid}"] for rid in ("1", "2", "3", "6", "7", "8")
                 if f"13-{rid}" in runs}
    if challenge:
        parts, below = [], []
        for rid, r in challenge.items():
            parts.append(f"{r['cue_recall_ade'] - r['no_cue_recall_ade']:+.3f} for run {rid}")
            if r["ci95_cue_minus_no_cue_recall_ade"][1] < 0:
                below.append(rid)
        L.append(
            "- **Negation and hedging (section 10, run 13).** On test sentences containing a "
            "negation or hedging cue, ADE recall changes by " + "; ".join(parts) + ". "
            + (f"Of these, only runs {', '.join(below)} lose recall with an interval entirely "
               "below zero." if below else "No interval lies entirely below zero."))

    if TAXONOMY_JSON.exists():
        t = json.loads(TAXONOMY_JSON.read_text(encoding="utf-8"))
        significant = t.get("significant", [])
        found = "; ".join(
            f"{PROPERTY.get(s['property'], s['property'])} among "
            f"{POPULATION.get(s['population'], s['population'])} "
            f"({s['fail_rate']:.1%} vs {s['ok_rate']:.1%}, p = {s['p']:.3f})"
            for s in significant)
        L.append(
            f"- **Error analysis (section 11).** {t['failures']} of {t['sentences']:,} test "
            f"sentences are pipeline failures: {t['gate_miss']} gate misses, "
            f"{t['gate_false_alarm']} gate false alarms and {t['stage2_error']} Stage 2 entity "
            "errors. "
            + (f"The only sentence property whose over-representation among failures survives a "
               f"Bonferroni correction over {t['tests']} tests is {found}."
               if len(significant) == 1 else
               f"Sentence properties surviving a Bonferroni correction over {t['tests']} tests: "
               f"{found}." if significant else
               f"No sentence property's over-representation survives a Bonferroni correction "
               f"over {t['tests']} tests."))

    manual = manual_categories()
    if manual is not None:
        filled, total = manual
        if filled == 0:
            L.append(
                f"- **Still to do by hand.** `results/error_sample.csv` lists {total} sampled "
                "failures with empty `manual_category` and `notes` columns. Section 11 reports "
                "what failing sentences *contain*; what *caused* each failure should be quoted "
                "only once those columns are filled in.")
        else:
            L.append(f"- Manual categories are filled in for {filled} of the {total} sampled "
                     "failures in `results/error_sample.csv`.")

    L.append("")
    return L


def main() -> int:
    missing = [str(path.relative_to(REPO_ROOT)) for _, _, path in SECTIONS if not path.exists()]
    if missing:
        raise SystemExit(f"missing result files: {', '.join(missing)} - run the Phase 6 "
                         "scripts first")

    L = [
        "# Pipeline, negation and error analysis (report sections 9-11)",
        "",
        "Generated by `scripts/document_analysis.py`, which assembles the three Phase 6 result "
        "files - `results/figures/pipeline_results.md`, `results/figures/negation_results.md` "
        "and `results/error_taxonomy.md` - under the PRD section 13 numbering. Each of those is "
        "generated from saved predictions by its own script, so nothing here is transcribed by "
        "hand.",
        "",
        "**Nothing in Phase 6 was trained.** Every number is inference over checkpoints and "
        "predictions that sections 7 and 8 already report, and each script checks that those "
        "reproduce their logged test scores before using them.",
        "",
        "---",
        "",
    ]
    L += summary(run_metrics())
    L += ["---", ""]

    for number, title, path in SECTIONS:
        L += [f"## {number} {title}", ""]
        L += renumber(path, number)
        L += ["", "---", ""]

    L += [
        "## Reproduction",
        "",
        "```bash",
        "python scripts/check_stage2_inference.py   # CPU decode == remote decode (gate)",
        "python scripts/baseline_predictions.py     # runs 1/2/2b predictions, refit exactly",
        "python scripts/pipeline_eval.py            # section 9, runs 12/12b",
        "python scripts/negation_eval.py            # section 10, run 13",
        "python scripts/error_taxonomy.py           # section 11",
        "python scripts/document_analysis.py        # this document",
        "```",
        "",
        "`pipeline_eval.py` and `negation_eval.py` append rows to `results/runs.csv`; add "
        "`--no-log` to regenerate their result files without adding rows. Run "
        "`baseline_predictions.py` before them, or the sparse baselines are missing from the "
        "gate ladder in section 9 and from every table in section 10.",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
