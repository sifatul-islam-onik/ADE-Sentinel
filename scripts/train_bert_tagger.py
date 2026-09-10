"""Step 5.6 - Stage 2 BERT token classification. Run 11.

    python scripts/train_bert_tagger.py --run-id 11 --model biomedbert

BiomedBERT is the default: it won the Stage 1 transformer comparison (macro-F1
0.9402 against 0.9159 for bert-base-uncased), and using the weaker one here
would understate the transformer tier for no reason.

**The subword alignment problem, and why it is the whole risk of this script.**
The BIO tags are one-per-*word*, produced by the project tokenizer. WordPiece
splits words further: `hepatotoxicity` becomes several pieces. Something has to
decide what tag each piece carries, and getting it wrong does not raise - it
trains on shifted labels and reports a plausible F1.

Two rules, both standard and both load-bearing:

1. **Label the first subword of each word; mask the rest** with -100 so they are
   excluded from the loss. Labelling every piece would let one long word
   dominate the loss in proportion to how badly WordPiece fragments it - which
   in this corpus means drug names, exactly the entities that matter.
2. **Decode back to word level before scoring.** Predictions are read off the
   first subword of each word, so the tag sequence handed to `seqeval` has one
   tag per word and is directly comparable with runs 9 and 10. Scoring at
   subword level would make run 11's numbers incomparable with the BiLSTMs' -
   the same entity would count for a different number of tokens.

`is_split_into_words=True` plus `word_ids()` does the bookkeeping; the assertion
after alignment checks the invariant rather than trusting it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import pin_single_gpu  # noqa: E402

MODELS = {
    "biomedbert": ("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                   "BiomedBERT, pretrained from scratch on PubMed abstracts"),
    "bert-base-uncased": ("bert-base-uncased",
                          "general-domain BERT (books + Wikipedia)"),
    "biobert": ("dmis-lab/biobert-base-cased-v1.2",
                "BioBERT, general BERT further pretrained on PubMed"),
    "scibert-ade": ("jsylee/scibert_scivocab_uncased-finetuned-ner",
                    "SciBERT already fine-tuned for ADE NER - the step 5.9 reference"),
}

IGNORE_INDEX = -100


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", default="biomedbert", choices=sorted(MODELS),
                   help="default biomedbert - the Stage 1 transformer winner")
    p.add_argument("--splits", type=Path, default=REPO_ROOT / "data" / "splits")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "stage2")

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--effective-batch", type=int, default=16,
                   help="divided across visible GPUs, not multiplied by them (F8)")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=192,
                   help="subword budget; 96 words fragment well past 128 pieces")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--dataset-version", default="")
    p.add_argument("--allow-multi-gpu", action="store_true")
    return p.parse_args(argv)


def load_word_level(split_path: Path):
    """Parquet -> (words, tags) per sentence, at WORD level.

    Uses the same converter and the same punctuation filter as runs 9-10, so all
    three Stage 2 runs are scored over an identical token stream.
    """
    import pandas as pd

    from src.bio_convert import ConversionStats, to_bio
    from src.models.encoding import is_indexable

    df = pd.read_parquet(split_path)
    stats = ConversionStats()
    rows = []

    for text, spans in zip(df.text, df.spans):
        raw = json.loads(spans) if isinstance(spans, str) else spans
        parsed = [(int(s), int(e), str(label)) for s, e, label in raw]

        tokens, tags = to_bio(text, parsed, stats=stats, strict=False)
        kept = [(t, tag) for t, tag in zip(tokens, tags) if is_indexable(t)]
        if not kept:
            continue

        words, word_tags = zip(*kept)
        rows.append({"words": list(words), "tags": list(word_tags), "text": text})

    return rows, stats


def align_labels(tokenizer, rows, tag_to_id, max_len):
    """Tokenise into subwords and project word-level tags onto the first piece."""
    encoded = tokenizer(
        [r["words"] for r in rows],
        is_split_into_words=True, truncation=True, max_length=max_len,
    )

    all_labels = []
    for i, row in enumerate(rows):
        word_ids = encoded.word_ids(batch_index=i)
        labels, previous = [], None

        for word_id in word_ids:
            if word_id is None:                 # [CLS], [SEP], padding
                labels.append(IGNORE_INDEX)
            elif word_id != previous:           # first subword of a new word
                labels.append(tag_to_id[row["tags"][word_id]])
            else:                               # continuation piece
                labels.append(IGNORE_INDEX)
            previous = word_id

        # The invariant: exactly one scored position per word that survived
        # truncation. If this ever fails, the labels are shifted.
        scored = sum(label != IGNORE_INDEX for label in labels)
        expected = len(set(w for w in word_ids if w is not None))
        assert scored == expected, (
            f"sentence {i}: {scored} scored positions for {expected} words")

        all_labels.append(labels)

    encoded = {k: v for k, v in encoded.items()}
    encoded["labels"] = all_labels
    return encoded


def decode_to_words(tokenizer, rows, predictions, tags_list, max_len):
    """Read predictions off the first subword of each word -> word-level tags.

    Words lost to truncation are filled with `O` so gold and predicted sequences
    stay the same length; the count is returned so the report can state it
    rather than leave it implicit.
    """
    encoded = tokenizer(
        [r["words"] for r in rows],
        is_split_into_words=True, truncation=True, max_length=max_len,
    )

    gold_all, pred_all, truncated_words = [], [], 0

    for i, row in enumerate(rows):
        word_ids = encoded.word_ids(batch_index=i)
        pred_by_word, previous = {}, None

        for position, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous:
                pred_by_word[word_id] = tags_list[predictions[i][position]]
            previous = word_id

        truncated_words += len(row["tags"]) - len(pred_by_word)
        pred_all.append([pred_by_word.get(w, "O") for w in range(len(row["tags"]))])
        gold_all.append(list(row["tags"]))

    return gold_all, pred_all, truncated_words


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.allow_multi_gpu:
        pin_single_gpu()

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    from src.bio_convert import TAG_TO_ID, TAGS
    from src.stage2_metrics import entity_metrics, token_accuracy
    from src.utils import derive_per_device_batch, get_device_count, log_run, set_seed

    seed = set_seed(args.seed)
    checkpoint, description = MODELS[args.model]
    device_count = get_device_count()
    per_device = derive_per_device_batch(args.effective_batch, device_count)

    print(f"run {args.run_id} | {checkpoint}")
    print(f"  {description}")
    print(f"  device_count {device_count} | per-device {per_device} | "
          f"effective {per_device * max(device_count, 1)} | lr {args.lr}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    rows, datasets = {}, {}
    for split in ("train", "dev", "test"):
        split_rows, stats = load_word_level(args.splits / f"stage2_{split}.parquet")
        rows[split] = split_rows
        encoded = align_labels(tokenizer, split_rows, TAG_TO_ID, args.max_len)
        datasets[split] = Dataset.from_dict(encoded)
        print(f"  stage2_{split}: {len(split_rows):,} sentences | "
              f"{stats.entities_tagged:,} entities tagged")

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint, num_labels=len(TAGS),
        id2label={i: t for i, t in enumerate(TAGS)},
        label2id=dict(TAG_TO_ID),
    )

    training_args = TrainingArguments(
        output_dir=str(args.out / f"run{args.run_id}_{args.model}" / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device * 4,
        learning_rate=args.lr,
        fp16=(not args.no_fp16) and torch.cuda.is_available(),
        seed=seed, data_seed=seed,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1_strict", greater_is_better=True,
        save_total_limit=1, logging_steps=100, report_to=[],
    )

    def compute_metrics(eval_pred):
        """Score at WORD level, strictly - the same criterion runs 9-10 use."""
        logits, labels = eval_pred
        predicted = logits.argmax(-1)

        gold, pred = [], []
        for row_pred, row_labels in zip(predicted, labels):
            keep = row_labels != IGNORE_INDEX
            gold.append([TAGS[i] for i in row_labels[keep]])
            pred.append([TAGS[i] for i in row_pred[keep]])

        scores = entity_metrics(gold, pred)
        return {"entity_f1_strict": scores["entity_f1_strict"],
                "entity_f1_lenient": scores["entity_f1_lenient"],
                "illegal_transitions": scores["illegal_transitions"]}

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=datasets["train"], eval_dataset=datasets["dev"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    dev_f1 = trainer.evaluate(datasets["dev"])["eval_entity_f1_strict"]

    output = trainer.predict(datasets["test"])
    predictions = output.predictions.argmax(-1)
    gold, pred, truncated = decode_to_words(
        tokenizer, rows["test"], predictions, TAGS, args.max_len)

    metrics = entity_metrics(gold, pred)
    metrics.update({
        "token_accuracy": token_accuracy(gold, pred),
        "dev_entity_f1_strict": float(dev_f1),
        "train_seconds": round(train_secs, 1),
        "words_lost_to_truncation": int(truncated),
    })

    print(f"\n  TEST entity-F1 strict {metrics['entity_f1_strict']:.4f} "
          f"| lenient {metrics['entity_f1_lenient']:.4f} "
          f"(gap {metrics['strict_lenient_gap']:+.4f})")
    print(f"  P {metrics['entity_precision_strict']:.3f} "
          f"R {metrics['entity_recall_strict']:.3f} "
          f"| illegal transitions {metrics['illegal_transitions']}")
    print(f"  words lost to truncation: {truncated} | {train_secs / 60:.1f} min")

    out = args.out / f"run{args.run_id}_{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out / "best"))
    tokenizer.save_pretrained(str(out / "best"))
    (out / "test_predictions.json").write_text(
        json.dumps({"gold": gold, "pred": pred,
                    "text": [r["text"] for r in rows["test"]]}), encoding="utf-8")
    (out / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "checkpoint": checkpoint,
                    "args": {k: str(v) for k, v in vars(args).items()}}, indent=2),
        encoding="utf-8")

    log_run(
        run_id=args.run_id, stage="2", model=args.model, embedding=checkpoint,
        metrics=metrics,
        params={"checkpoint": checkpoint, "max_len": args.max_len,
                "fp16": training_args.fp16, "optimiser": "adamw",
                "subword_labelling": "first piece only, rest -100",
                "scored_at": "word level",
                "selected_on": "dev entity_f1_strict per epoch",
                "tags": list(TAGS), "torch": torch.__version__},
        seed=seed, dataset_version=args.dataset_version,
        per_device_batch=per_device, device_count=device_count,
        epochs=args.epochs, lr=args.lr,
        notes=f"Step 5.6 run {args.run_id}; {description}; word-level scoring.",
    )

    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
