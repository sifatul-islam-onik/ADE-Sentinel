"""Steps 5.4-5.5 - trains the Stage 2 BiLSTM tagger. Runs 9 and 10.

    python scripts/train_bilstm_tagger.py --run-id 9  --embedding E3
    python scripts/train_bilstm_tagger.py --run-id 10 --embedding E3 --crf

**`--crf` is the only flag that may differ between those two commands.** Same
embedding, same seed, same hyperparameters, same split - so the difference
between runs 9 and 10 is attributable to structured prediction and nothing else.

E3 is the default because it won the Stage 1 ablation (macro-F1 0.8785 frozen,
against 0.7942 for GloVe). PLAN 5.4 says "best of E1-E3"; that is what the
Phase 4 chart selected, and using anything else here would throw away the reason
Phase 3 was run first.

**Selection on dev entity-F1, strict mode.** Not token accuracy, which is 79%
for a model that predicts `O` everywhere, and not lenient entity-F1, which
repairs malformed sequences and would therefore hide exactly the difference run
10 is meant to demonstrate.

**Single GPU, always** (PLAN F8). `DataParallel` splits the batch across devices
and the CRF computes its loss inside `forward`, so two GPUs return a loss vector
rather than a scalar. The script pins one device before torch is imported.

Test predictions are written as tag strings alongside the checkpoint: Phase 6
needs them for the end-to-end pipeline and the error taxonomy, and re-running a
GPU job to recover what you already computed is quota spent on nothing.
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

MAX_LEN = 96          # covers 100% of Stage 2 sentences; p95 is 38 tokens
PATIENCE = 4
EMBEDDINGS = ("E0_random", "E1", "E2", "E3")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--embedding", default="E3", choices=EMBEDDINGS,
                   help="default E3 - the Stage 1 ablation winner (PLAN 5.4)")
    p.add_argument("--crf", action="store_true",
                   help="run 10: linear-chain CRF instead of per-token softmax")

    p.add_argument("--matrices", type=Path,
                   default=REPO_ROOT / "models" / "emb_matrices")
    p.add_argument("--splits", type=Path, default=REPO_ROOT / "data" / "splits")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "stage2")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32, help="PER DEVICE")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unfreeze-embeddings", action="store_true")
    p.add_argument("--dataset-version", default="")
    p.add_argument("--allow-multi-gpu", action="store_true",
                   help="opt out of the single-GPU pin. The CRF returns a per-device "
                        "loss vector under DataParallel, so this needs care (PLAN F8)")
    return p.parse_args(argv)


def build_dataset(split_path: Path, index):
    """Parquet -> (ids, lengths, tag ids, gold tag strings).

    Conversion happens here rather than being cached to disk on purpose: the BIO
    tags are a *derived* artefact, and deriving them from the committed spans on
    every run means the converter and the training data can never drift apart.
    It costs about a second for 4,271 sentences.
    """
    import pandas as pd

    from src.bio_convert import ConversionStats, to_bio
    from src.models.encoding import encode_tokens_with_tags

    df = pd.read_parquet(split_path)
    stats = ConversionStats()
    rows = []

    for text, spans in zip(df.text, df.spans):
        raw = json.loads(spans) if isinstance(spans, str) else spans
        parsed = [(int(s), int(e), str(label)) for s, e, label in raw]

        tokens, tags = to_bio(text, parsed, stats=stats, strict=False)
        ids, tags = encode_tokens_with_tags(tokens, tags, index)
        if not ids:
            continue

        rows.append((ids[:MAX_LEN], tags[:MAX_LEN], text))

    return rows, stats


def collate(rows, tag_to_id, pad_id=0):
    import torch

    width = max(len(ids) for ids, _, _ in rows)
    id_matrix, tag_matrix, lengths = [], [], []

    for ids, tags, _ in rows:
        pad = width - len(ids)
        id_matrix.append(ids + [pad_id] * pad)
        # Padded tag slots are never scored: the softmax branch masks them to
        # IGNORE_INDEX and the CRF excludes them via the mask.
        tag_matrix.append([tag_to_id[t] for t in tags] + [0] * pad)
        lengths.append(len(ids))

    return (torch.tensor(id_matrix, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(tag_matrix, dtype=torch.long))


def batches(rows, batch_size, tag_to_id, shuffle=False, seed=42):
    import random

    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)

    for start in range(0, len(order), batch_size):
        chunk = [rows[i] for i in order[start:start + batch_size]]
        yield collate(chunk, tag_to_id), chunk


def predict(model, rows, batch_size, tag_to_id, tags_list, device):
    """Decode every sentence, returning (gold, predicted) ragged tag strings."""
    model.eval()
    gold_all, pred_all = [], []

    for (ids, lengths, _), chunk in batches(rows, batch_size, tag_to_id):
        decoded = model.decode(ids.to(device), lengths.to(device))
        for (_, gold_tags, _), path in zip(chunk, decoded):
            pred_all.append([tags_list[i] for i in path])
            gold_all.append(list(gold_tags))

    return gold_all, pred_all


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.allow_multi_gpu:
        pin_single_gpu()

    import numpy as np
    import torch

    from src.bio_convert import TAG_TO_ID, TAGS
    from src.models.bilstm_tagger import BiLSTMTagger
    from src.models.encoding import load_vocab
    from src.stage2_metrics import entity_metrics, token_accuracy
    from src.utils import get_device_count, log_run, set_seed

    seed = set_seed(args.seed)
    device_count = get_device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_count > 1 and not args.allow_multi_gpu:
        raise SystemExit(
            f"{device_count} GPUs are visible. Runs 9 and 10 must differ only in "
            "the CRF flag, and DataParallel changes both the effective batch and "
            "how the CRF loss is reduced (PLAN F8).\n"
            "Fix: set CUDA_VISIBLE_DEVICES to a single device before importing "
            "torch, or pass --allow-multi-gpu on BOTH runs."
        )

    vocab, index = load_vocab(args.matrices / "vocab.json")
    matrix = np.load(args.matrices / f"{args.embedding}.npy")
    if matrix.shape[0] != len(vocab):
        raise ValueError(
            f"{args.embedding}.npy has {matrix.shape[0]} rows but vocab.json has "
            f"{len(vocab)} - they came from different builds.")

    splits = {}
    for split in ("train", "dev", "test"):
        rows, stats = build_dataset(args.splits / f"stage2_{split}.parquet", index)
        splits[split] = rows
        print(f"  stage2_{split}: {len(rows):,} sentences | {stats.spans:,} spans "
              f"| {stats.entities_tagged:,} tagged | {stats.spans_snapped:,} snapped "
              f"| {stats.spans_unmatched:,} unmatched")

    print(f"\nrun {args.run_id} | {args.embedding} | "
          f"{'BiLSTM-CRF' if args.crf else 'BiLSTM softmax'}")
    print(f"  tags {list(TAGS)}")
    print(f"  device {device} | device_count {device_count} | batch {args.batch_size}")

    model = BiLSTMTagger(
        matrix, num_tags=len(TAGS), hidden_dim=args.hidden_dim,
        dropout=args.dropout, use_crf=args.crf,
        freeze_embeddings=not args.unfreeze_embeddings,
    ).to(device)
    print(f"  embeddings {'fine-tuned' if args.unfreeze_embeddings else 'FROZEN'} "
          f"| trainable params {model.trainable_parameters():,}")

    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best = {"dev_f1": -1.0, "epoch": 0, "state": None}
    history, t0 = [], time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, count = 0.0, 0

        for (ids, lengths, tags), _ in batches(
                splits["train"], args.batch_size, TAG_TO_ID,
                shuffle=True, seed=seed + epoch):
            optimiser.zero_grad()
            loss = model(ids.to(device), lengths.to(device), tags.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            running += loss.item()
            count += 1

        gold, pred = predict(model, splits["dev"], args.batch_size,
                             TAG_TO_ID, TAGS, device)
        dev = entity_metrics(gold, pred)
        dev_f1 = dev["entity_f1_strict"]
        history.append({"epoch": epoch, "train_loss": running / count,
                        "dev_entity_f1_strict": dev_f1,
                        "dev_illegal": dev["illegal_transitions"]})

        marker = ""
        if dev_f1 > best["dev_f1"]:
            best = {"dev_f1": dev_f1, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            marker = "  <- best"

        print(f"  epoch {epoch:>2}  loss {running / count:>8.4f}  "
              f"dev entity-F1 {dev_f1:.4f}  illegal {dev['illegal_transitions']:>3}"
              f"{marker}")

        if epoch - best["epoch"] >= PATIENCE:
            print(f"  early stop: {PATIENCE} epochs without improvement")
            break

    train_secs = time.time() - t0

    model.load_state_dict(best["state"])
    gold, pred = predict(model, splits["test"], args.batch_size,
                         TAG_TO_ID, TAGS, device)

    metrics = entity_metrics(gold, pred)
    metrics.update({
        "token_accuracy": token_accuracy(gold, pred),
        "dev_entity_f1_strict": best["dev_f1"],
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "train_seconds": round(train_secs, 1),
    })

    print(f"\n  TEST entity-F1 strict {metrics['entity_f1_strict']:.4f} "
          f"| lenient {metrics['entity_f1_lenient']:.4f} "
          f"(gap {metrics['strict_lenient_gap']:+.4f})")
    print(f"  P {metrics['entity_precision_strict']:.3f} "
          f"R {metrics['entity_recall_strict']:.3f} "
          f"| illegal transitions {metrics['illegal_transitions']} "
          f"in {metrics['illegal_sentences']} sentences")
    print(f"  best epoch {best['epoch']}/{len(history)} | {train_secs:.0f}s")

    out = args.out / f"run{args.run_id}_{'crf' if args.crf else 'softmax'}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "config": model.config,
                "embedding": args.embedding, "run_id": args.run_id,
                "tags": list(TAGS)}, out / "checkpoint.pt")
    (out / "test_predictions.json").write_text(
        json.dumps({"gold": gold, "pred": pred,
                    "text": [t for _, _, t in splits["test"]]}), encoding="utf-8")
    (out / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "history": history,
                    "args": {k: str(v) for k, v in vars(args).items()}}, indent=2),
        encoding="utf-8")

    log_run(
        run_id=args.run_id, stage="2",
        model="bilstm_crf" if args.crf else "bilstm_softmax",
        embedding=args.embedding, metrics=metrics,
        params={**model.config, "max_len": MAX_LEN, "patience": PATIENCE,
                "optimiser": "adam", "grad_clip": 5.0,
                "early_stop_on": "dev entity_f1_strict",
                "tags": list(TAGS), "torch": torch.__version__},
        seed=seed, dataset_version=args.dataset_version,
        per_device_batch=args.batch_size, device_count=device_count,
        epochs=len(history), lr=args.lr,
        notes=(f"Step {'5.5' if args.crf else '5.4'} run {args.run_id}; "
               f"{'BiLSTM-CRF' if args.crf else 'BiLSTM softmax'} on {args.embedding}; "
               f"best epoch {best['epoch']}."),
    )

    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
