"""Writes report/stage2_documentation.md - the Phase 5 / Stage 2 report section.

Companion to `document_stage1.py`, built the same way: every figure is read from
`results/runs.csv`, the saved tag sequences in `models/stage2/`, the frozen
splits and the step 5.9 reference output, so nothing here can drift from the
artefacts that produced it. Nothing is transcribed by hand.

The strict/lenient, CRF and sanity-check arguments are the sections
`scripts/stage2_report.py` writes to `results/figures/stage2_results.md`,
imported and reused rather than re-derived, so the figure file and the report
section cannot disagree. What this adds is the narrative around them: the data
and training setup, where the entity errors fall, the entities no tagger finds,
provenance, cost, and threats to validity.

Usage:  .venv\\Scripts\\python scripts\\document_stage2.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_report as report  # noqa: E402
from src.stage2_metrics import strict_entities  # noqa: E402
from src.utils import RUNS_CSV  # noqa: E402

SPLITS = REPO_ROOT / "data" / "splits"
STAGE1 = REPO_ROOT / "models" / "stage1"
STAGE2 = REPO_ROOT / "models" / "stage2"
MATRICES = REPO_ROOT / "models" / "emb_matrices"
REPRODUCTION = REPO_ROOT / "results" / "stage2_reproduction.json"
OUT = REPO_ROOT / "report" / "stage2_documentation.md"

TIERS = {
    "9": ("S1", "BiLSTM + per-token softmax", "E3 FastText, frozen", "no-CRF ablation"),
    "10": ("S2", "BiLSTM + CRF", "E3 FastText, frozen", "structured prediction"),
    "11": ("S3", "BiomedBERT token classification", "domain WordPiece", "transformer tagger"),
}

SHAPES = (("DRUG", False, "DRUG, one token"), ("DRUG", True, "DRUG, several tokens"),
          ("EFFECT", False, "EFFECT, one token"), ("EFFECT", True, "EFFECT, several tokens"))


def f(value, spec=".4f", missing="-"):
    return missing if value is None else f"{value:{spec}}"


def i(value, missing="-"):
    """Render a CSV-round-tripped number as the integer it is."""
    if value is None or value != value or value == "":
        return missing
    return f"{int(float(value)):,}"


def get(row, key, default=None):
    return row["metrics"].get(key, default)


def body(lines: list[str]) -> list[str]:
    """A `stage2_report` section without its own `##` title."""
    if lines and lines[0].startswith("## "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return lines


def model_name(run_id: str) -> str:
    return TIERS.get(run_id, ("", run_id))[1]


def join_and(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------

def split_stats() -> dict:
    """Per split: sentences, scored tokens, entities by shape, nested spans dropped."""
    import pandas as pd

    from src.bio_convert import ConversionStats, to_bio
    from src.models.encoding import is_indexable

    out = {}
    for split in ("train", "dev", "test"):
        df = pd.read_parquet(SPLITS / f"stage2_{split}.parquet")
        stats, shape, tokens, outside = ConversionStats(), Counter(), 0, 0
        for text, raw in zip(df.text, df.spans):
            spans = [(int(s), int(e), str(label)) for s, e, label in json.loads(raw)]
            words, tags = to_bio(text, spans, stats=stats, strict=False)
            kept = [tag for word, tag in zip(words, tags) if is_indexable(word)]
            tokens += len(kept)
            outside += kept.count("O")
            for start, end, label in strict_entities(kept):
                shape[(label, end - start > 1)] += 1
        out[split] = {"sentences": len(df), "tokens": tokens,
                      "o_share": outside / max(tokens, 1), "shape": shape,
                      "nested": stats.spans_dropped_nested}
    return out


def f4_check() -> dict:
    """Is the Stage 2 test split exactly Stage 1's test positives (PLAN F4)?"""
    import pandas as pd

    stage1 = pd.read_parquet(SPLITS / "stage1_test.parquet")
    stage2 = set(pd.read_parquet(SPLITS / "stage2_test.parquet").text)
    positives = set(stage1.text[stage1.label == 1])
    leaked = sum(len(stage2 & set(pd.read_parquet(SPLITS / f"stage1_{s}.parquet").text))
                 for s in ("train", "dev"))
    return {"stage2_test": len(stage2), "matched": len(stage2 & positives),
            "stage1_positives": len(positives), "leaked": leaked}


def embedding_provenance():
    """Are the frozen embedding rows inside the BiLSTM checkpoints the local E3, bit for bit?

    Frozen rows receive no gradient, so a checkpoint's embedding weight is the
    matrix it was given. This substitutes for the Dataset version the Stage 2
    run rows did not record. Needs torch; returns None without it.
    """
    try:
        import numpy as np
        import torch
    except ImportError:
        return None

    matrix = MATRICES / "E3.npy"
    if not matrix.exists():
        return None
    e3 = np.load(matrix)

    candidates = [("run 9", STAGE2 / "run9_softmax" / "checkpoint.pt"),
                  ("run 10", STAGE2 / "run10_crf" / "checkpoint.pt"),
                  ("Stage 1 run 6", STAGE1 / "run6_E3" / "checkpoint.pt")]
    out = {}
    for label, path in candidates:
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            weight = saved["state_dict"]["embedding.weight"].numpy()
            out[label] = bool(weight.shape == e3.shape and np.array_equal(weight, e3))
    return out or None


def bert_history():
    """Run 11's per-epoch dev evaluations, from the Trainer's saved state."""
    path = STAGE2 / "run11_biomedbert" / "trainer_state.json"
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    return [h for h in state.get("log_history", []) if "eval_entity_f1_strict" in h] or None


def stage1_metric(all_runs, run_id: str, key: str = "macro_f1"):
    rows = all_runs[(all_runs.stage == "1") & (all_runs.run_id == run_id)]
    return json.loads(rows.iloc[-1].metrics_json).get(key) if len(rows) else None


def hub_revisions() -> list[str]:
    path = STAGE2 / "run11_biomedbert" / "huggingface_repos.json"
    if not path.exists():
        return []
    repos = json.loads(path.read_text(encoding="utf-8")).get("repos", [])
    return [f"`{r['commitHash'][:12]}` ({', '.join(r['filePaths'])})" for r in repos]


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def data_section(splits, f4, conversion) -> list[str]:
    L = [
        "## 8.1 Task and data",
        "",
        "Each sentence is tokenised with the project tokenizer (`src/tokenizer.py`) and "
        "its annotated character spans are converted to BIO tags over those tokens "
        "(`src/bio_convert.py`, with the overlap test and zero-match assertion of PLAN "
        "F5). Five tags: `O`, `B-DRUG`, `I-DRUG`, `B-EFFECT`, `I-EFFECT`. "
        "Punctuation-only tokens are dropped in lockstep with their tags before training "
        "and scoring; the test suite verifies over the real splits that this changes no "
        "entity count.",
        "",
        "| Split | Sentences | Scored tokens | DRUG entities | EFFECT entities | DRUG spanning >1 token | EFFECT spanning >1 token | Nested spans dropped |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ("train", "dev", "test"):
        x, sh = splits[name], splits[name]["shape"]
        drug = sh[("DRUG", False)] + sh[("DRUG", True)]
        effect = sh[("EFFECT", False)] + sh[("EFFECT", True)]
        L.append(f"| {name} | {x['sentences']:,} | {x['tokens']:,} | {drug:,} | {effect:,} | "
                 f"{sh[('DRUG', True)] / max(drug, 1):.1%} | "
                 f"{sh[('EFFECT', True)] / max(effect, 1):.1%} | {x['nested']:,} |")
    L.append("")

    if (f4["matched"] == f4["stage2_test"] == f4["stage1_positives"]) and f4["leaked"] == 0:
        L += [
            f"**The split is the global one** (PLAN F4). The {f4['stage2_test']:,} Stage 2 "
            f"test sentences are exactly the {f4['stage1_positives']:,} ADE-positive "
            "sentences of the Stage 1 test split, and none of them appears in Stage 1 "
            "train or dev. That is what makes run 12 - both stages chained over one test "
            "set - a valid measurement rather than a Stage 1 model scoring sentences it "
            "was trained on.",
            "",
        ]
    else:
        L += [
            f"**The Stage 2 test split does not line up with Stage 1's**: {f4['matched']} of "
            f"{f4['stage2_test']} test sentences are Stage 1 test positives (of "
            f"{f4['stage1_positives']}), and {f4['leaked']} appear in Stage 1 train or dev. "
            "Run 12 is not valid until this is resolved.",
            "",
        ]

    test = splits["test"]
    sh = test["shape"]
    effect_multi = sh[("EFFECT", True)] / max(sh[("EFFECT", True)] + sh[("EFFECT", False)], 1)
    drug_multi = sh[("DRUG", True)] / max(sh[("DRUG", True)] + sh[("DRUG", False)], 1)
    L += [
        f"**The two labels have different shapes.** {effect_multi:.0%} of test EFFECT "
        "entities span more than one token (`acute renal failure`), against "
        f"{drug_multi:.0%} of DRUG entities. Whether a token continues an entity or "
        "starts one is therefore mostly an EFFECT decision, which 8.6 returns to.",
        "",
        f"`O` is {test['o_share']:.1%} of scored test tokens, so a tagger that predicts "
        "`O` everywhere reaches that token accuracy while finding no entity at all. "
        "Token accuracy appears in 8.4 only to be argued against; entity-level F1 is the "
        "metric (PRD 8.2).",
        "",
        f"**{test['nested']:,} test spans were dropped before tagging** because they nest "
        "inside another span (`theophylline` inside `theophylline intoxication`), and flat "
        "BIO gives each token one tag. They are absent from the gold tags too, so they do "
        "not lower the scores below - but no tagger in this section can be credited for "
        "them, and recovering them would need one BIO plane per label. Across the whole "
        f"corpus the converter aligned every annotated span ({conversion['spans_unmatched']} "
        f"unmatched of {conversion['spans']:,}); `results/bio_conversion_report.md` has "
        "the details.",
        "",
        "---",
        "",
    ]
    return L


def models_section(df, all_runs) -> list[str]:
    L = [
        "## 8.2 Models compared",
        "",
        "Three tiers (PRD 8.2), each changing one component of the one before.",
        "",
        "| Run | Tier | Model | Input representation | Role |",
        "|---|---|---|---|---|",
    ]
    for run_id in df.run_id:
        if run_id in TIERS:
            tier, model, rep, role = TIERS[run_id]
            L.append(f"| {run_id} | {tier} | {model} | {rep} | {role} |")

    L += [
        "",
        "**Run 9 to run 10 changes only the decoder**: per-token argmax becomes Viterbi "
        "decoding over a learned transition matrix (8.6). **Run 10 to run 11 changes the "
        "encoder** - static vectors and a BiLSTM become a pretrained transformer - and "
        "drops the CRF.",
        "",
    ]

    e3, e1 = stage1_metric(all_runs, "6"), stage1_metric(all_runs, "4")
    r8, r7 = stage1_metric(all_runs, "8"), stage1_metric(all_runs, "7")
    if None not in (e3, e1, r8, r7):
        L += [
            "The components were chosen by Phase 4 rather than re-litigated here. E3 won "
            f"the frozen Stage 1 embedding ablation (macro-F1 {e3:.4f}, against {e1:.4f} "
            f"for GloVe), and BiomedBERT won the transformer comparison ({r8:.4f}, against "
            f"{r7:.4f} for `bert-base-uncased`). Using anything weaker would understate its "
            "tier.",
            "",
        ]

    L += [
        "**Run 11 has no CRF.** Its head is a per-token softmax over contextual "
        "encodings, so it can emit illegal transitions exactly as run 9 can. That makes it "
        "a second data point on the question 8.6 asks: whether better per-token decisions "
        "alone keep sequences well-formed.",
        "",
        "---",
        "",
    ]
    return L


def training_section(df, by_id, all_runs) -> list[str]:
    L = ["## 8.3 Training setup", ""]

    if "10" in by_id.index:
        r = by_id.loc["10"]
        p = json.loads(r.params_json)
        L += [
            "### S1/S2 BiLSTM taggers (runs 9-10)",
            "",
            "| Parameter | Value |",
            "|---|---|",
            f"| Embeddings | {r.embedding}, {'frozen' if p.get('freeze_embeddings') else 'fine-tuned'}, "
            f"{i(p.get('vocab_size'))} x {p.get('embed_dim')} |",
            f"| Hidden units per direction | {p.get('hidden_dim')} |",
            f"| Layers | {p.get('num_layers')} |",
            f"| Dropout | {p.get('dropout')} |",
            f"| Max sequence length | {p.get('max_len')} tokens |",
            f"| Optimiser | {p.get('optimiser')}, lr {r.lr} |",
            f"| Batch size | {i(r.per_device_batch)} per device, {i(r.effective_batch)} effective |",
            f"| Gradient clipping | {p.get('grad_clip')} |",
            f"| Early stopping | {p.get('early_stop_on')}, patience {p.get('patience')} |",
            f"| Seed | {r.seed} |",
            "",
        ]

        if "9" in by_id.index:
            q = json.loads(by_id.loc["9"].params_json)
            differing = sorted(k for k in set(p) | set(q) if p.get(k) != q.get(k))
            for column in ("embedding", "lr", "per_device_batch", "seed"):
                if by_id.loc["9"][column] != r[column]:
                    differing.append(column)
            if differing == ["use_crf"]:
                L += [
                    "Both runs use exactly this configuration. **`use_crf` is the only logged "
                    "parameter that differs between runs 9 and 10** - checked field by field "
                    "against `results/runs.csv` when this document is generated.",
                    "",
                ]
            else:
                L += [f"**Runs 9 and 10 differ in more than the CRF**: {', '.join(differing)}. "
                      "The ablation in 8.6 is confounded until that is explained.", ""]

        L += [
            "**Selection on dev strict entity-F1** - not token accuracy, which `O` inflates, "
            "and not lenient F1, which scores sequences only after repairing them. The same "
            "criterion selects run 11's epoch.",
            "",
        ]

    if "11" in by_id.index:
        r = by_id.loc["11"]
        p = json.loads(r.params_json)
        lr8, epochs8 = None, None
        rows8 = all_runs[(all_runs.stage == "1") & (all_runs.run_id == "8")]
        if len(rows8):
            lr8, epochs8 = rows8.iloc[-1].lr, rows8.iloc[-1].epochs

        L += [
            "### S3 BiomedBERT tagger (run 11)",
            "",
            "| Parameter | Value |",
            "|---|---|",
            f"| Checkpoint | `{p.get('checkpoint')}` |",
            f"| Epochs | {i(r.epochs)} |",
            f"| Learning rate | {r.lr} |",
            f"| Batch size | {i(r.per_device_batch)} per device, {i(r.effective_batch)} effective |",
            f"| Max sequence length | {p.get('max_len')} WordPiece tokens |",
            f"| Subword labelling | {p.get('subword_labelling')} |",
            f"| Scored at | {p.get('scored_at')} |",
            f"| Mixed precision | {'fp16' if p.get('fp16') else 'off'} |",
            f"| Checkpoint selection | {p.get('selected_on')} |",
            f"| Seed | {r.seed} |",
            "",
        ]
        if lr8 is not None:
            L += [
                f"The recipe is not Stage 1's: {i(r.epochs)} epochs at lr {r.lr} with a "
                f"{p.get('max_len')}-piece budget, against {i(epochs8)} epochs at lr {lr8} "
                "and 128 pieces for run 8. The budget is larger because a 96-word sentence "
                "fragments well past 128 WordPieces and every word must survive to be "
                f"scored; {i(get(r, 'words_lost_to_truncation'))} test words were lost to "
                "truncation.",
                "",
            ]
        L += [
            "**One label per word, on its first subword.** The remaining pieces are masked "
            "out of the loss, and predictions are read back from the first piece before "
            "scoring. Labelling every piece would weight a word in the loss by how badly "
            "WordPiece fragments it - in this corpus, drug names - and scoring at subword "
            "level would make run 11's entity counts incomparable with the BiLSTMs'.",
            "",
        ]

        history = bert_history()
        if history:
            L += [
                "| Epoch | Dev strict F1 | Dev lenient F1 | Dev illegal transitions | Dev loss |",
                "|---|---|---|---|---|",
            ]
            for h in history:
                L.append(f"| {h['epoch']:.0f} | {h['eval_entity_f1_strict']:.4f} | "
                         f"{h['eval_entity_f1_lenient']:.4f} | {h['eval_illegal_transitions']} | "
                         f"{h['eval_loss']:.3f} |")
            L.append("")

            best = max(history, key=lambda h: h["eval_entity_f1_strict"])
            lowest = min(history, key=lambda h: h["eval_loss"])
            if best is history[-1]:
                L += [
                    "**The selected epoch is the last one.** Dev strict F1 was at its maximum "
                    f"when the {len(history)}-epoch budget ran out, so the budget may be "
                    "binding; longer training was not tried.",
                    "",
                ]
            if lowest["epoch"] != best["epoch"]:
                L += [
                    f"**Dev loss and dev F1 disagree.** Loss was lowest after epoch "
                    f"{lowest['epoch']:.0f} ({lowest['eval_loss']:.3f}) and had risen to "
                    f"{best['eval_loss']:.3f} by the selected epoch, while strict F1 rose "
                    f"from {lowest['eval_entity_f1_strict']:.4f} to "
                    f"{best['eval_entity_f1_strict']:.4f}. Rising loss alongside rising F1 is "
                    "the usual signature of growing over-confidence on the errors that "
                    "remain; either way, selecting on loss would have kept the weaker "
                    "checkpoint.",
                    "",
                ]

    # ---- provenance ------------------------------------------------------------
    devices = sorted({int(d) for d in df.device_count if d == d})
    commits = sorted({str(c) for c in df.git_commit if c == c})
    versions = sorted({str(v) for v in df.dataset_version if v == v and str(v).strip()})
    torch_versions = sorted({json.loads(pj).get("torch", "") for pj in df.params_json} - {""})
    config = STAGE2 / "run11_biomedbert" / "best" / "config.json"
    transformers_version = (json.loads(config.read_text(encoding="utf-8")).get("transformers_version")
                            if config.exists() else None)

    L += [
        "### Where the runs executed",
        "",
        "| Property | Value |",
        "|---|---|",
        "| Hardware | Kaggle, 1x NVIDIA T4 |",
        f"| Device counts observed | {', '.join(map(str, devices))} |",
        f"| Code commit | {', '.join(f'`{c}`' for c in commits)} |",
        f"| Kaggle Dataset version | {', '.join(versions) if versions else '**not recorded**'} |",
        f"| torch | {', '.join(torch_versions) or '-'} |",
        f"| transformers (run 11) | {transformers_version or '-'} |",
        "",
    ]

    dirty = [str(int(d)) for d in df.git_dirty if d == d]
    if dirty and dirty[0] == "0" and set(dirty[1:]) <= {"1"}:
        L += [
            f"`git_dirty` reads {', '.join(dirty)} across runs {', '.join(df.run_id)}: clean "
            "for the first run of the session and dirty after it, which is the pattern "
            "produced by the first run appending its row to the tracked "
            "`results/runs.csv`.",
            "",
        ]

    if not versions:
        prov = embedding_provenance()
        stage1_version = None
        rows6 = all_runs[(all_runs.stage == "1") & (all_runs.run_id == "6")]
        if len(rows6) and rows6.iloc[-1].dataset_version == rows6.iloc[-1].dataset_version:
            stage1_version = i(rows6.iloc[-1].dataset_version)

        if prov and all(prov.values()):
            L += [
                "**The Dataset version was not recorded for these runs** - the notebook's "
                "`DATASET_VERSION` was left blank - so that column cannot vouch for which "
                "embedding matrix runs 9-10 loaded. The checkpoints can. The frozen "
                f"embedding rows saved inside {join_and(prov)} are **bit-identical** to the "
                "local `models/emb_matrices/E3.npy`, and frozen rows receive no gradient, so "
                "this identifies the input exactly: runs 9-10 used the same E3 as the Stage "
                "1 ablation"
                + (f", whose rows recorded Dataset version {stage1_version}." if stage1_version
                   else "."),
                "",
            ]
        else:
            L += [
                "**The Dataset version was not recorded for these runs**, and the embedding "
                "matrix inside the checkpoints could not be verified against `E3.npy` in "
                "this environment. The input to runs 9-10 is unverified.",
                "",
            ]

    revisions = hub_revisions()
    if revisions:
        L += [
            "Run 11 reads no embedding matrix. Its pretrained checkpoint resolved to Hub "
            f"revision {' and '.join(revisions)} in that session "
            "(`models/stage2/run11_biomedbert/huggingface_repos.json`).",
            "",
        ]

    L += ["---", ""]
    return L


def results_section(df, by_id, rescored, splits) -> list[str]:
    L = [
        "## 8.4 Results",
        "",
        "Test-set entity scores in the oracle setting. Checkpoints were selected on dev; "
        "the last column shows the gap between selection and reporting.",
        "",
        "| Run | Model | Strict F1 | Lenient F1 | Overlap F1 | Strict P | Strict R | DRUG F1 | EFFECT F1 | Token acc. | Dev strict F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    best = max(df.run_id, key=lambda r: get(by_id.loc[r], "entity_f1_strict", 0) or 0)
    for run_id in df.run_id:
        r = by_id.loc[run_id]
        mark = "**" if run_id == best else ""
        L.append(
            f"| {run_id} | {model_name(run_id)} | {mark}{f(get(r, 'entity_f1_strict'))}{mark} | "
            f"{f(get(r, 'entity_f1_lenient'))} | {f(rescored.get(run_id, {}).get('overlap_f1'))} | "
            f"{f(get(r, 'entity_precision_strict'), '.3f')} | {f(get(r, 'entity_recall_strict'), '.3f')} | "
            f"{f(get(r, 'drug_f1'), '.3f')} | {f(get(r, 'effect_f1'), '.3f')} | "
            f"{f(get(r, 'token_accuracy'), '.3f')} | {f(get(r, 'dev_entity_f1_strict'))} |")
    L.append("")

    if report.CHART.exists():
        L += ["![Stage 2 entity-F1 and illegal transitions](../results/figures/stage2_crf.png)", ""]

    strict_best = get(by_id.loc[best], "entity_f1_strict")
    L += [
        f"Best: **run {best}** at strict entity-F1 {strict_best:.4f}. That checkpoint is "
        "what Phase 6.1 carries into the end-to-end pipeline (run 12).",
        "",
        "**Three scores, three questions.** *Strict* is exact-match IOB2 entity-F1, the "
        "headline. *Lenient* is seqeval's default: boundaries must still be exact, but a "
        "malformed sequence such as `O I-DRUG` is repaired into an entity first (8.5). "
        "*Overlap* credits a prediction that overlaps a gold entity of the same label even "
        "with the wrong boundaries - the PRD's partial-match stretch goal. Lenient is "
        "easily mistaken for partial matching, and `PLAN.md` step 5.7 makes that mistake; "
        "only the overlap column measures tolerance to boundary errors.",
        "",
    ]

    token_acc = [get(by_id.loc[r], "token_accuracy") for r in df.run_id]
    strict = [get(by_id.loc[r], "entity_f1_strict") for r in df.run_id]
    if None not in token_acc + strict:
        L += [
            f"Token accuracy spans only {min(token_acc):.3f}-{max(token_acc):.3f} across the "
            f"three runs while strict entity-F1 spans {min(strict):.3f}-{max(strict):.3f}, and "
            f"an all-`O` tagger would score {splits['test']['o_share']:.3f} on it. That is the "
            "concrete case against token accuracy PRD 8.2 asks the report to make.",
            "",
        ]

    overlap = rescored.get(best, {}).get("overlap_f1")
    if overlap is not None:
        L += [
            f"Run {best}'s overlap F1 is {overlap:.4f}: crediting every boundary error would "
            f"recover {overlap - strict_best:+.4f} of the {1 - strict_best:.4f} it is short of "
            "a perfect exact-match score. 8.7 breaks the errors down.",
            "",
        ]

    drug = [get(by_id.loc[r], "drug_f1") for r in df.run_id]
    effect = [get(by_id.loc[r], "effect_f1") for r in df.run_id]
    if None not in drug + effect and all(e < d for d, e in zip(drug, effect)):
        L += [
            f"**EFFECT is the harder label for every tagger** (F1 "
            f"{' / '.join(f'{e:.3f}' for e in effect)} against "
            f"{' / '.join(f'{d:.3f}' for d in drug)} for DRUG, runs {', '.join(df.run_id)}).",
            "",
        ]

    L += ["---", ""]
    return L


def transformer_note(by_id) -> list[str]:
    if not all(r in by_id.index for r in ("9", "10", "11")):
        return []
    n9, n10, n11 = (get(by_id.loc[r], "illegal_transitions") for r in ("9", "10", "11"))
    L = ["### The transformer, without a CRF", ""]
    if n10 < n11 < n9:
        L += [
            f"Run 11 emits {n11} illegal transitions on test: {1 - n11 / n9:.0%} fewer than "
            f"run 9's {n9}, and {n11 / max(n10, 1):.1f}x run 10's {n10}. Better per-token "
            "decisions reduce malformed output, but in this comparison only structured "
            "decoding brings it close to zero. A BiomedBERT-CRF was not run, so whether the "
            "CRF's benefit survives a contextual encoder is untested.",
            "",
        ]
    else:
        L += [f"Illegal transitions on test: run 9 {n9}, run 10 {n10}, run 11 {n11}.", ""]

    history = bert_history()
    if history:
        counts = [h["eval_illegal_transitions"] for h in history]
        still_falling = len(counts) > 1 and counts[-1] == min(counts) and counts[-1] < counts[-2]
        L += [
            f"On dev its count went from {counts[0]} after epoch 1 to {counts[-1]} after "
            f"epoch {len(counts)}"
            + (" - the lowest of the run, and still falling when the epoch budget ran out, "
               "so where a contextual softmax tagger levels off is not measured here."
               if still_falling else "."),
            "",
        ]
    return L


def error_section(saved, rescored) -> list[str]:
    from src.stage2_inference import words_of

    runs = [r for r in ("9", "10", "11") if r in saved and r in rescored]
    if not runs:
        return []

    gold_total = sum(rescored[runs[0]][k] for k in
                     ("gold_found", "gold_boundary", "gold_type", "gold_missed"))
    L = [
        "## 8.7 Where the entity errors are",
        "",
        "Exact-match F1 scores a one-token boundary slip exactly like an invented entity. "
        "Every gold test entity is classified here by what each tagger did with it "
        "(`error_breakdown`, strict reading): found exactly, found with the wrong extent, "
        "found under the other label, or missed.",
        "",
        "| Gold entity outcome | " + " | ".join(f"Run {r}" for r in runs) + " |",
        "|---|" + "---|" * len(runs),
    ]
    for label, key in (("found exactly", "gold_found"), ("wrong boundaries", "gold_boundary"),
                       ("wrong label", "gold_type"), ("missed entirely", "gold_missed")):
        L.append(f"| {label} | " + " | ".join(
            f"{rescored[r][key]:,} ({rescored[r][key] / gold_total:.1%})" for r in runs) + " |")
    L += ["| *predictions overlapping no gold entity* | "
          + " | ".join(f"{rescored[r]['pred_spurious']:,}" for r in runs) + " |", ""]

    best = max(runs, key=lambda r: rescored[r]["entity_f1_strict"])
    b = rescored[best]
    if b["gold_boundary"] > max(b["gold_missed"], b["gold_type"]):
        L += [
            f"**For the best tagger (run {best}) the most common failure is a boundary "
            f"error, not a miss**: {b['gold_boundary']} gold entities were found with the "
            f"wrong extent, against {b['gold_missed']} not found at all. Under exact-match "
            "scoring a boundary error typically costs twice - a false negative for the gold "
            "entity and a false positive for the prediction - which makes it the most "
            "expensive error class here, and the one the overlap F1 in 8.4 forgives.",
            "",
        ]
    else:
        largest = max(("gold_boundary", "gold_type", "gold_missed"), key=lambda k: b[k])
        L += [f"For the best tagger (run {best}) the largest failure class is "
              f"`{largest.removeprefix('gold_')}` ({b[largest]} entities).", ""]

    # ---- recall by entity shape ------------------------------------------------
    gold = saved[runs[0]]["gold"]
    found = {r: Counter() for r in runs}
    totals = Counter()
    for idx, tags in enumerate(gold):
        entities = strict_entities(tags)
        for e in entities:
            totals[(e[2], e[1] - e[0] > 1)] += 1
        for r in runs:
            predicted = set(strict_entities(saved[r]["pred"][idx]))
            for e in entities:
                found[r][(e[2], e[1] - e[0] > 1)] += e in predicted

    L += [
        "### Exact recall by entity shape",
        "",
        "| Gold entities | Count | " + " | ".join(f"Run {r}" for r in runs) + " |",
        "|---|---|" + "---|" * len(runs),
    ]
    for label, multi, name in SHAPES:
        n = totals[(label, multi)]
        L.append(f"| {name} | {n:,} | " + " | ".join(
            f"{found[r][(label, multi)] / max(n, 1):.1%}" for r in runs) + " |")
    L.append("")

    def rate(r, key):
        return found[r][key] / max(totals[key], 1)

    hardest = {r: min(((lab, m) for lab, m, _ in SHAPES), key=lambda k: rate(r, k)) for r in runs}
    if set(hardest.values()) == {("EFFECT", True)}:
        L += [
            "**Multi-token EFFECT spans are the hardest entities for every tagger**, and they "
            "are also the most common EFFECT shape - which is why EFFECT F1 trails DRUG F1 "
            "in every run.",
            "",
        ]

    if "9" in runs and "10" in runs:
        d_multi = found["10"][("EFFECT", True)] - found["9"][("EFFECT", True)]
        d_single = found["10"][("EFFECT", False)] - found["9"][("EFFECT", False)]
        if d_multi * d_single < 0:
            L += [
                "**Inside EFFECT, the CRF's impact on recall is a trade rather than a uniform "
                f"gain.** Between runs 9 and 10, exactly-found multi-token EFFECT entities "
                f"changed by {d_multi:+d} and single-token ones by {d_single:+d}, a net of "
                f"{d_multi + d_single:+d}. The CRF's EFFECT gain in 8.6 is therefore a "
                "precision gain - fewer wrong predictions - not additional recall.",
                "",
            ]

    # ---- the residue no tagger finds -------------------------------------------
    never, always = [], 0
    for idx, tags in enumerate(gold):
        predicted = [set(strict_entities(saved[r]["pred"][idx])) for r in runs]
        for e in strict_entities(tags):
            hits = sum(e in p for p in predicted)
            if hits == 0:
                never.append((idx, e))
            elif hits == len(runs):
                always += 1

    if never:
        shape = Counter((e[2], e[1] - e[0] > 1) for _, e in never)
        top = shape.most_common(1)[0]
        top_name = next(name for lab, m, name in SHAPES if (lab, m) == top[0])

        outcome = Counter()
        for idx, (start, end, label) in never:
            overlapping = [p for p in strict_entities(saved[best]["pred"][idx])
                           if p[0] < end and start < p[1]]
            if any(p[2] == label for p in overlapping):
                outcome["boundary"] += 1
            elif overlapping:
                outcome["type"] += 1
            else:
                outcome["missed"] += 1

        if outcome["boundary"] > outcome["missed"]:
            residue = (
                f"Most of this residue is not invisible to the models. Run {best} overlaps "
                f"{outcome['boundary']} of the {len(never)} with a prediction of the right "
                f"label but different boundaries, labels {outcome['type']} as the other type, "
                f"and misses {outcome['missed']} outright."
            )
        else:
            residue = (
                f"Run {best} misses {outcome['missed']} of the {len(never)} outright, overlaps "
                f"{outcome['boundary']} with a same-label prediction of different extent, and "
                f"labels {outcome['type']} as the other type."
            )

        L += [
            "### Entities no tagger finds",
            "",
            f"**{len(never)} of {gold_total:,} test entities ({len(never) / gold_total:.1%}) "
            f"are found exactly by none of the three taggers**, and {always:,} "
            f"({always / gold_total:.1%}) by all of them. The largest group among the "
            f"universally missed is {top_name} ({top[1]} of {len(never)}).",
            "",
            residue,
            "",
            f"The first few, in test-set order, with what run {best} predicted over the same "
            "words:",
            "",
        ]
        texts = saved[best]["text"]
        for idx, (start, end, label) in never[:6]:
            words = words_of(texts[idx])
            overlapping = [p for p in strict_entities(saved[best]["pred"][idx])
                           if p[0] < end and start < p[1]]
            guess = ", ".join(f"`{' '.join(words[p[0]:p[1]])}` ({p[2]})"
                              for p in overlapping) or "nothing"
            snippet = texts[idx] if len(texts[idx]) <= 160 else texts[idx][:157] + "..."
            L.append(f"- {label} `{' '.join(words[start:end])}` - run {best} predicted "
                     f"{guess}. _{snippet}_")
        L += [
            "",
            "These, with the pipeline failures of run 12, are the input to the Phase 6.4 "
            "error taxonomy.",
            "",
        ]

    L += ["---", ""]
    return L


def compute_section(by_id) -> list[str]:
    runs = [r for r in ("9", "10", "11") if r in by_id.index]
    secs = {r: get(by_id.loc[r], "train_seconds") or 0 for r in runs}

    L = [
        "## Compute cost",
        "",
        "| Run | Model | Training time | Epochs | Hardware |",
        "|---|---|---|---|---|",
    ]
    for r in runs:
        L.append(f"| {r} | {model_name(r)} | {secs[r]:.0f} s | {i(by_id.loc[r].epochs)} | "
                 "1x T4, Kaggle |")
    L.append("")

    if "9" in secs and "10" in secs:
        delta = get(by_id.loc["10"], "entity_f1_strict") - get(by_id.loc["9"], "entity_f1_strict")
        L += [f"The CRF costs {secs['10'] / max(secs['9'], 1e-9):.1f}x the softmax tagger's "
              f"training time for {delta:+.4f} strict entity-F1.", ""]
        if "11" in secs and secs["11"] < secs["10"]:
            L += [
                f"**The transformer trained faster than the BiLSTM-CRF** ({secs['11']:.0f} s "
                f"against {secs['10']:.0f} s, all epochs included). `pytorch-crf` computes the "
                "partition function and the Viterbi decode with a Python loop over time "
                "steps, while BERT's cost is dense matrix work a T4 is built for - so on this "
                "hardware the CRF, not the 110M-parameter encoder, is the slow component.",
                "",
            ]

    if REPRODUCTION.exists():
        rep = json.loads(REPRODUCTION.read_text(encoding="utf-8"))
        rows = rep.get("runs", {})
        if rows:
            L += [
                "### Inference on CPU",
                "",
                f"Measured locally by `scripts/check_stage2_inference.py` "
                f"({rep.get('threads')} CPU threads), re-decoding the full test split:",
                "",
                "| Run | Load | Decode | Per sentence (batched) | Identical to remote decode |",
                "|---|---|---|---|---|",
            ]
            for r, v in rows.items():
                per = 1000 * v["decode_seconds"] / max(v["sentences"], 1)
                L.append(f"| {r} | {v['load_seconds']:.1f} s | {v['decode_seconds']:.1f} s | "
                         f"{per:.1f} ms | {v['identical_sentences']}/{v['sentences']} |")
            L.append("")

            if "10" in rows and "11" in rows and "10" in by_id.index and "11" in by_id.index:
                size10 = sum(p.stat().st_size for p in (STAGE2 / "run10_crf" / "checkpoint.pt",
                                                        MATRICES / "E3.npy") if p.exists())
                size11 = sum(p.stat().st_size for p in (STAGE2 / "run11_biomedbert" / "best").glob("*"))
                gap = (get(by_id.loc["11"], "entity_f1_strict")
                       - get(by_id.loc["10"], "entity_f1_strict"))
                load10, load11 = rows["10"]["load_seconds"], rows["11"]["load_seconds"]
                per10 = 1000 * rows["10"]["decode_seconds"] / max(rows["10"]["sentences"], 1)
                per11 = 1000 * rows["11"]["decode_seconds"] / max(rows["11"]["sentences"], 1)
                if load11 >= 5:
                    target = "Run 11's load time alone exceeds the 5-second target. "
                elif load11 > 3.5:
                    target = "Run 11's load time alone is inside the 5-second target, but with little margin. "
                else:
                    target = "Both load well inside the 5-second target. "
                L += [
                    "PLAN 7.2 ships the BiLSTM-CRF in the demo for CPU inference and a "
                    "5-second cold start. Measured: run 10 loads in "
                    + (f"{load10:.1f} s" if load10 >= 0.1 else "under 0.1 s")
                    + f" and decodes in about {max(per10, 1):.0f} ms per sentence, from "
                    f"{size10 / 1e6:.0f} MB on disk; run 11 takes {load11:.1f} s to load and "
                    f"about {per11:.0f} ms per sentence, from {size11 / 1e6:.0f} MB. "
                    + target
                    + f"Shipping run 10 instead gives up {gap:.4f} strict entity-F1.",
                    "",
                ]
    return L


def threats_section(by_id, rescored, splits, reference) -> list[str]:
    prov = embedding_provenance()
    repro = json.loads(REPRODUCTION.read_text(encoding="utf-8")) if REPRODUCTION.exists() else None
    crosschecks = [rescored[r].get("seqeval_crosscheck") for r in sorted(rescored)]
    words_lost = get(by_id.loc["11"], "words_lost_to_truncation") if "11" in by_id.index else None
    crf_delta = None
    if "9" in by_id.index and "10" in by_id.index:
        crf_delta = (get(by_id.loc["10"], "entity_f1_strict")
                     - get(by_id.loc["9"], "entity_f1_strict"))

    contamination = "flagged - used as a sanity check, never as a competitor (8.8)"
    run11 = (reference or {}).get("run11_local", {})
    if reference and "train" in run11 and "test" in run11:
        ref = [reference["scores"][s]["entity_f1_strict"] for s in ("train", "dev", "test")]
        seen = run11["train"]["entity_f1_strict"]
        unseen = run11["test"]["entity_f1_strict"]
        if abs(ref[2] - seen) >= abs(ref[2] - unseen) and max(ref) - min(ref) < 0.02:
            contamination = ("checked - no lift on any of our splits (8.8); still used only "
                             "as a sanity check, never as a competitor")

    embedded_ok = bool(prov and prov.get("run 9") and prov.get("run 10"))
    rows = [
        ("CRF ablation confounded by anything but the decoder",
         "controlled - `use_crf` is the only differing parameter; "
         + ("the frozen embedding rows in both checkpoints are bit-identical to `E3.npy`"
            if embedded_ok else "embedding identity not verified in this environment")),
        ("Scores trusted from the remote session",
         "controlled - every run rescored locally from its saved tags; seqeval "
         + ("agrees on every run" if crosschecks and all(c == "agrees" for c in crosschecks)
            else f"cross-check: {', '.join(map(str, crosschecks))}")),
        ("Local CPU inference (Phase 6) differs from the remote decode",
         ("controlled - re-decode identical for "
          + ", ".join(f"run {r} ({v['identical_sentences']}/{v['sentences']})"
                      for r, v in repro["runs"].items()))
         if repro else "**not yet checked** - run `scripts/check_stage2_inference.py`"),
        ("Test set used for model selection",
         "controlled - checkpoints chosen on dev strict entity-F1; test decoded once per run"),
        ("Run 11 scored at subword level, inflating its entity counts",
         f"controlled - read back to words from the first subword; {i(words_lost)} words lost "
         "to truncation"),
        ("Lenient scoring flattering malformed output",
         "controlled - strict is the headline, and the direction of the lenient gap is "
         "measured (8.5), not assumed"),
        ("Nested spans unrepresentable in flat BIO",
         f"measured, not controlled - {splits['test']['nested']} test spans dropped before "
         "tagging; no tagger here can be credited for them"),
        ("Reference tagger may have trained on our test sentences", contamination),
        ("Kaggle Dataset version not recorded for runs 9-11",
         "mitigated - " + ("the embedding input is verified bit-for-bit instead (8.3)"
                           if embedded_ok else "not verified")),
        ("CRF benefit on a contextual encoder",
         "**not tested** - no BiomedBERT-CRF run exists"),
        ("Oracle setting assumes a perfect Stage 1",
         "by design for this section; the pipeline setting is run 12 (Phase 6.1-6.2)"),
        ("Single-seed results",
         "**not controlled** - seed 42 only"
         + (f"; the CRF's {crf_delta:+.4f} is the smallest effect this section argues "
            "from and has no variance estimate" if crf_delta is not None else "")),
    ]
    L = ["## Threats to validity", "", "| Threat | Status |", "|---|---|"]
    L += [f"| {threat} | {status} |" for threat, status in rows]
    L.append("")
    return L


def main() -> int:
    import pandas as pd

    df = report.load_runs()
    by_id = df.set_index("run_id")
    all_runs = pd.read_csv(RUNS_CSV, dtype={"run_id": str, "stage": str})
    saved = report.load_predictions()
    rescored = report.rescore(saved)
    reference = report.load_reference()
    splits = split_stats()
    conversion = report.corpus_tag_stats()

    L = [
        "# Stage 2 - drug and effect span extraction (report section 8)",
        "",
        "Generated by `scripts/document_stage2.py` from `results/runs.csv`, the saved tag "
        "sequences in `models/stage2/`, the frozen splits and the step 5.9 reference "
        "output. Sections 8.5, 8.6 and 8.8 are the ones `scripts/stage2_report.py` writes "
        "to `results/figures/stage2_results.md`, reused rather than restated, so the two "
        "files cannot disagree.",
        "",
        "Stage 2 answers the question Stage 1 only flags: **which words are the drug, and "
        "which are the adverse effect?** It is sequence labelling over sentences that "
        "report an ADE, scored at entity level.",
        "",
        "**Every number in this section is the oracle setting.** Stage 2 is evaluated on the "
        "gold-positive test sentences, as if Stage 1 were perfect. What the pipeline loses "
        "when Stage 1 is not perfect is measured by run 12 (Phase 6.1-6.2) and reported "
        "there rather than folded in here.",
        "",
        "---",
        "",
    ]

    L += data_section(splits, f4_check(), conversion)
    L += models_section(df, all_runs)
    L += training_section(df, by_id, all_runs)
    L += results_section(df, by_id, rescored, splits)

    L += ["## 8.5 What lenient scoring measures", ""]
    L += body(report.lenient_section(df, rescored))
    L += ["---", ""]

    L += ["## 8.6 The CRF ablation", ""]
    L += body(report.crf_section(by_id, rescored))
    L += report.transitions_section(saved)
    L += transformer_note(by_id)
    L += ["---", ""]

    L += error_section(saved, rescored)

    L += ["## 8.8 External sanity checks", ""]
    L += body(report.sanity_section(by_id, reference))
    L += ["---", ""]

    L += compute_section(by_id)

    L += [
        "## Reproduction",
        "",
        "```bash",
        "# remote - notebooks/stage2_remote.ipynb, 1x T4",
        "python scripts/train_bilstm_tagger.py --run-id 9  --embedding E3",
        "python scripts/train_bilstm_tagger.py --run-id 10 --embedding E3 --crf",
        "python scripts/train_bert_tagger.py   --run-id 11 --model biomedbert",
        "",
        "# local",
        "python scripts/check_stage2_inference.py   # CPU decode must equal the remote decode",
        "python scripts/reference_tagger.py         # step 5.9 reference tagger",
        "python scripts/stage2_report.py            # steps 5.7-5.9 tables + chart",
        "python scripts/document_stage2.py          # this document",
        "```",
        "",
        "`--crf` is the only difference between the first two commands. The notebook clones "
        "the repository and calls exactly these scripts, so that statement is about a "
        "command line rather than about notebook cells.",
        "",
    ]

    L += threats_section(by_id, rescored, splits, reference)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")

    print(f"{len(df)} Stage 2 runs documented: {', '.join(df.run_id)}")
    print(f"reference tagger: {'included' if reference else 'not run yet'}")
    print(f"CPU reproduction check: {'included' if REPRODUCTION.exists() else 'not run yet'}")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
