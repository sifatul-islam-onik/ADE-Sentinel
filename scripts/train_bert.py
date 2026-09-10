"""Step 4.4 - Stage 1 transformer fine-tuning. Runs 7 and 8.

    python scripts/train_bert.py --run-id 7 --model bert-base-uncased
    python scripts/train_bert.py --run-id 8 --model biomedbert

Runs 7 and 8 replay the project's central claim one level up. Runs 3-6 ask
whether *static* vectors trained on biomedical text beat general-purpose ones;
these two ask the same question of *contextual* ones, with the pretraining
corpus as the only difference between two otherwise identical architectures.
If domain wins in both places, the report has a symmetry worth more than either
result alone (PRD 8.1).

**Effective batch, not per-device batch (PLAN F8).** HF `Trainer` silently
multiplies `per_device_train_batch_size` by the visible device count, so the
PRD's "batch 16 at lr 2e-5" means one thing on Kaggle's 2xT4 and another on a
single GPU. This script takes the *effective* batch as the argument and derives
per-device from the devices it can actually see, so the recipe is honoured on
either machine and the derived pair is written to `runs.csv`.

Everything else follows the PRD recipe: 3 epochs, lr 2e-5, max_len 128, fp16.
The best of the three epochs is chosen by dev macro-F1 and scored once on test.
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
    "bert-base-uncased": (
        "bert-base-uncased", "general-domain BERT (books + Wikipedia)"),
    "biomedbert": (
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "BiomedBERT, pretrained from scratch on PubMed abstracts"),
    "biobert": (
        "dmis-lab/biobert-base-cased-v1.2",
        "BioBERT, general BERT further pretrained on PubMed"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--splits", type=Path, default=REPO_ROOT / "data" / "splits")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "stage1")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--effective-batch", type=int, default=16,
                   help="PRD recipe. Divided across visible GPUs, not multiplied "
                        "by them (PLAN F8)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--dataset-version", default="")
    p.add_argument("--allow-multi-gpu", action="store_true",
                   help="use every visible GPU. The effective batch is still the "
                        "one you asked for - it is split across devices, not "
                        "multiplied - but device_count then differs from runs 3-6")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.allow_multi_gpu:
        # Before torch is imported. Runs 7-8 do not strictly need one device -
        # `derive_per_device_batch` honours the recipe on any count - but keeping
        # every Phase 4 run on the same device makes the `device_count` column
        # uniform, and one T4 fine-tunes BERT-base on 14.6k sentences comfortably.
        pin_single_gpu()

    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    from src.metrics import classification_metrics, macro_f1
    from src.utils import derive_per_device_batch, get_device_count, log_run, set_seed

    seed = set_seed(args.seed)
    checkpoint, description = MODELS[args.model]
    device_count = get_device_count()
    per_device = derive_per_device_batch(args.effective_batch, device_count)

    print(f"run {args.run_id} | {checkpoint}")
    print(f"  {description}")
    print(f"  device_count {device_count} | per-device batch {per_device} "
          f"| effective {per_device * max(device_count, 1)} | lr {args.lr} "
          f"| {args.epochs} epochs")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    frames, datasets = {}, {}
    for split in ("train", "dev", "test"):
        df = pd.read_parquet(args.splits / f"stage1_{split}.parquet")
        frames[split] = df
        datasets[split] = Dataset.from_pandas(
            df[["text", "label"]], preserve_index=False
        ).map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_len),
            batched=True, remove_columns=["text"],
        )

    truncated = sum(
        len(tokenizer(t, truncation=False)["input_ids"]) > args.max_len
        for t in frames["train"].text
    )
    print(f"  train {len(frames['train']):,} | {truncated} sentences exceed "
          f"max_len={args.max_len} ({truncated / len(frames['train']):.2%})")

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2,
        id2label={0: "not_ade", 1: "ade"}, label2id={"not_ade": 0, "ade": 1},
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
        metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1,
        logging_steps=100, report_to=[],
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=datasets["train"], eval_dataset=datasets["dev"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda ep: {
            "macro_f1": macro_f1(ep.label_ids, ep.predictions.argmax(-1))
        },
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    dev_f1 = trainer.evaluate(datasets["dev"])["eval_macro_f1"]

    # Test is scored once, on the epoch dev chose.
    output = trainer.predict(datasets["test"])
    logits = torch.as_tensor(output.predictions, dtype=torch.float32)
    scores = torch.softmax(logits, dim=1)[:, 1].numpy()
    y_pred = output.predictions.argmax(-1)
    y_true = frames["test"].label.values

    metrics = classification_metrics(y_true, y_pred, scores)
    metrics.update({
        "dev_macro_f1": float(dev_f1),
        "train_seconds": round(train_secs, 1),
        "truncated_train_sentences": int(truncated),
    })

    print(f"\n  TEST macro-F1 {metrics['macro_f1']:.4f} | ADE F1 {metrics['f1_ade']:.4f} "
          f"| ADE P {metrics['precision_ade']:.3f} R {metrics['recall_ade']:.3f} "
          f"| PR-AUC {metrics['pr_auc']:.3f}")
    print(f"  {train_secs / 60:.1f} min on {device_count} GPU(s)")

    out = args.out / f"run{args.run_id}_{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out / "best"))
    tokenizer.save_pretrained(str(out / "best"))
    np.savez(out / "test_predictions.npz", y_true=y_true, y_pred=y_pred, score=scores)
    (out / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "checkpoint": checkpoint,
                    "args": {k: str(v) for k, v in vars(args).items()}}, indent=2),
        encoding="utf-8")

    log_run(
        run_id=args.run_id, stage="1", model=args.model, embedding=checkpoint,
        metrics=metrics,
        params={"checkpoint": checkpoint, "max_len": args.max_len,
                "fp16": training_args.fp16, "optimiser": "adamw",
                "selected_on": "dev macro_f1 per epoch",
                "per_device_eval_batch": per_device * 4,
                "torch": torch.__version__},
        seed=seed, dataset_version=args.dataset_version,
        per_device_batch=per_device, device_count=device_count,
        epochs=args.epochs, lr=args.lr,
        notes=f"Step 4.4 run {args.run_id}; {description}; "
              f"effective batch {per_device * max(device_count, 1)}.",
    )

    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
