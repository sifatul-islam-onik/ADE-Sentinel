"""Step 5.9 - the external sanity check: a published ADE tagger on our split.

    python scripts/reference_tagger.py

PRD 6.1 names `jsylee/scibert_scivocab_uncased-finetuned-ner` - SciBERT already
fine-tuned for DRUG/EFFECT tagging on ADE Corpus v2 - as the way to confirm our
entity-F1 "lands in a plausible range rather than being silently broken by a
BIO conversion bug". This runs it over exactly the word stream runs 9-11 were
scored on (`src.stage2_inference.words_of`), scores it with the same strict
scorer, and saves its predictions where `stage2_report.py` looks for them.

Two properties of the reference decide how its number may be read.

**Its labels are unnamed.** The config calls them `LABEL_0`..`LABEL_4`. The
mapping onto our tag inventory is identified on the TRAIN split, by the gold tag
each raw label most often lands on, and must come out a bijection or the script
stops. Test plays no part in choosing it.

**It has almost certainly seen our test sentences.** The model card documents no
split, and ADE Corpus v2 is published as a single `train` split, so a model
fine-tuned on it cannot have held out what our frozen split holds out. Its test
score is therefore not comparable with run 11's. To make that caveat a number
rather than a disclaimer, run 11 is also scored on OUR train split - sentences
it was trained on - which measures how far a BERT-class tagger's score rises on
data it has seen. A reference score near run 11's train score is what
contamination looks like; one near run 11's test score is what a clean model
looks like.

Not logged to `results/runs.csv`: it is not one of the thirteen runs.
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

REFERENCE = "jsylee/scibert_scivocab_uncased-finetuned-ner"
SPLITS = REPO_ROOT / "data" / "splits"
STAGE2 = REPO_ROOT / "models" / "stage2"
OUT = STAGE2 / "ref_scibert_ade"
RUN11 = STAGE2 / "run11_biomedbert"

KEEP = ("entity_f1_strict", "entity_f1_lenient", "entity_precision_strict",
        "entity_recall_strict", "drug_f1", "effect_f1", "illegal_transitions",
        "gold_entities", "pred_entities", "seqeval_crosscheck")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--revision", default=None,
                   help="Hub commit to pin; default resolves main and records its sha")
    p.add_argument("--skip-run11", action="store_true",
                   help="skip the seen-vs-unseen measurement on run 11")
    return p.parse_args(argv)


def load_split(name: str):
    """(texts, words, gold tags) for one Stage 2 split, on the scored word stream."""
    import pandas as pd

    from src.bio_convert import to_bio
    from src.models.encoding import is_indexable

    df = pd.read_parquet(SPLITS / f"stage2_{name}.parquet")
    texts, words, gold = [], [], []
    for text, raw in zip(df.text, df.spans):
        spans = [(int(s), int(e), str(label)) for s, e, label in json.loads(raw)]
        tokens, tags = to_bio(text, spans, strict=False)
        kept = [(t, g) for t, g in zip(tokens, tags) if is_indexable(t)]
        if not kept:
            continue
        texts.append(text)
        words.append([t for t, _ in kept])
        gold.append([g for _, g in kept])
    return texts, words, gold


def identify_labels(raw_pred, gold, raw_labels):
    """Name the reference's labels by co-occurrence with our gold tags - train only.

    Returns the mapping and, per raw label, the share of its tokens that land on
    the tag it is mapped to. A low share would mean the reference's annotation
    convention differs from ours, which the report needs to know before reading
    its F1.
    """
    from src.bio_convert import TAGS

    counts = {label: Counter() for label in raw_labels}
    for p_seq, g_seq in zip(raw_pred, gold):
        for p, g in zip(p_seq, g_seq):
            if p in counts:
                counts[p][g] += 1

    mapping, purity = {}, {}
    for label, seen in counts.items():
        if not seen:
            raise SystemExit(f"{label} was never predicted on train - cannot name it")
        tag, n = seen.most_common(1)[0]
        mapping[label] = tag
        purity[label] = n / sum(seen.values())

    if sorted(mapping.values()) != sorted(TAGS):
        raise SystemExit(f"label identification is not a bijection: {mapping}\n{counts}")
    return mapping, purity


def over_budget(tagger, sentences) -> int:
    """Sentences whose subword count exceeds the budget, so lose words to truncation."""
    encoded = tagger.tokenizer(sentences, is_split_into_words=True, truncation=False)
    return sum(len(ids) > tagger.max_len for ids in encoded["input_ids"])


def main(argv=None) -> int:
    args = parse_args(argv)

    import torch
    import transformers
    from huggingface_hub import HfApi

    from src.stage2_inference import BertTagger
    from src.stage2_metrics import entity_metrics

    revision = args.revision or HfApi().model_info(REFERENCE).sha
    splits = {name: load_split(name) for name in ("train", "dev", "test")}
    for name, (texts, _, _) in splits.items():
        print(f"stage2_{name}: {len(texts):,} sentences")

    # ---- the reference ------------------------------------------------------
    t0 = time.time()
    reference = BertTagger(REFERENCE, revision=revision)
    print(f"\n{REFERENCE} @ {revision[:12]} | raw labels {reference.raw_labels}")

    raw = {}
    for name, (_, words, _) in splits.items():
        raw[name] = reference.tag(words)
        print(f"  tagged {name} ({time.time() - t0:.0f}s)")

    mapping, purity = identify_labels(raw["train"], splits["train"][2],
                                      reference.raw_labels)
    print("  label map (identified on train): " + ", ".join(
        f"{k}->{v} ({purity[k]:.1%})" for k, v in mapping.items()))

    scores, over = {}, {}
    for name, (texts, words, gold) in splits.items():
        pred = [[mapping.get(t, t) for t in seq] for seq in raw[name]]
        metrics = entity_metrics(gold, pred)
        scores[name] = {k: metrics[k] for k in KEEP if k in metrics}
        over[name] = over_budget(reference, words)
        print(f"  {name:5} strict F1 {metrics['entity_f1_strict']:.4f} | lenient "
              f"{metrics['entity_f1_lenient']:.4f} | over budget {over[name]}")
        if name == "test":
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "test_predictions.json").write_text(
                json.dumps({"gold": gold, "pred": pred, "text": texts}), encoding="utf-8")
    reference_seconds = time.time() - t0

    # ---- run 11 on data it has and has not seen -------------------------------
    run11 = {}
    if not args.skip_run11:
        t0 = time.time()
        ours = BertTagger(RUN11 / "best")
        saved = json.loads((RUN11 / "test_predictions.json").read_text(encoding="utf-8"))
        for name, (texts, words, gold) in splits.items():
            pred = ours.tag(words)
            metrics = entity_metrics(gold, pred)
            run11[name] = {k: metrics[k] for k in KEEP if k in metrics}
            if name == "test":
                if texts != saved["text"]:
                    raise SystemExit("run 11's saved predictions cover different sentences")
                run11["test_reproduction"] = {
                    "identical_sentences": sum(a == b for a, b in zip(pred, saved["pred"])),
                    "sentences": len(pred),
                    "saved_entity_f1_strict": entity_metrics(
                        saved["gold"], saved["pred"])["entity_f1_strict"],
                }
            print(f"  run 11 {name:5} strict F1 {metrics['entity_f1_strict']:.4f}")
        run11["seconds"] = round(time.time() - t0, 1)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps({
        "reference": REFERENCE,
        "revision": revision,
        "label_map": mapping,
        "label_purity_on_train": purity,
        "scores": scores,
        "sentences_over_subword_budget": over,
        "max_len": reference.max_len,
        "run11_local": run11,
        "reference_seconds": round(reference_seconds, 1),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {(OUT / 'metrics.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
