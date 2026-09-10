"""Step 4.3 - trains the Stage 1 BiLSTM. One invocation per run, 3 through 6.

    python scripts/train_bilstm.py --run-id 3 --embedding E0_random
    python scripts/train_bilstm.py --run-id 4 --embedding E1
    python scripts/train_bilstm.py --run-id 5 --embedding E2
    python scripts/train_bilstm.py --run-id 6 --embedding E3

**`--embedding` is the only flag that may differ between those four commands.**
That is the entire design of this script: the ablation is a claim about the
vectors, and it is only true if nothing else moved. Everything else - seed,
batch size, learning rate, epoch budget, early-stopping rule, class weights,
device count - is either a constant here or pinned by default.

Three specific traps this closes:

**Device count (PLAN F8).** Two visible GPUs would make the effective batch 64
where one makes it 32. If one run in the ablation lands on a different device
count than another, the four bars in the headline chart are no longer
comparable. This script therefore pins itself to a single GPU by default and
records `device_count` in every row.

**Early stopping on dev macro-F1, not dev loss.** Under 1:4 imbalance those two
disagree: loss keeps improving while the model quietly trades ADE recall for
`not-ADE` precision. Selecting on the metric the report leads with is the honest
choice, and it is applied identically to all four runs.

**Test set touched once.** Selection is entirely on dev; the best checkpoint by
dev macro-F1 is restored and scored on test exactly once, at the end.

Test predictions are written alongside the checkpoint - Phase 6.4 needs the
per-sentence failures of the best Stage 1 model, and re-running a GPU job to
recover predictions you already computed is a waste of quota.
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

MAX_LEN = 96          # covers 100% of training sentences; p95 is 33 tokens
PATIENCE = 3
EMBEDDINGS = {
    "E0_random": "randomly initialised (ablation floor)",
    "E1": "GloVe 300d (general purpose)",
    "E2": "Word2Vec skip-gram 300d (ours, PubMed)",
    "E3": "FastText skip-gram 300d (ours, PubMed)",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-id", required=True,
                   help="row identifier in results/runs.csv, e.g. 3 or 3u")
    p.add_argument("--embedding", required=True, choices=sorted(EMBEDDINGS),
                   help="which matrix from models/emb_matrices/ to load")
    p.add_argument("--matrices", type=Path,
                   default=REPO_ROOT / "models" / "emb_matrices")
    p.add_argument("--splits", type=Path, default=REPO_ROOT / "data" / "splits")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "stage1")

    # Fine-tuning the embedding layer answers a different question than freezing
    # it; the notebook runs both conditions rather than picking one silently.
    p.add_argument("--unfreeze-embeddings", action="store_true",
                   help="let the embedding layer receive gradient")

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32, help="PER DEVICE")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--dataset-version", default="",
                   help="Kaggle Dataset version supplying the matrices (PLAN F10)")

    p.add_argument("--allow-multi-gpu", action="store_true",
                   help="opt out of the single-GPU pin. Contaminates runs 3-6 "
                        "unless every run in the ablation also passes it (F8)")
    return p.parse_args(argv)


def build_loaders(args, index):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import pandas as pd

    from src.models.encoding import encode_batch

    loaders, frames = {}, {}
    for split in ("train", "dev", "test"):
        df = pd.read_parquet(args.splits / f"stage1_{split}.parquet")
        ids, lengths = encode_batch(df.text.tolist(), index, MAX_LEN)

        dataset = TensorDataset(
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(df.label.values, dtype=torch.long),
        )
        # Only train is shuffled; dev and test keep parquet row order so saved
        # predictions line up with the split for Phase 6 error analysis.
        loaders[split] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(split == "train"),
            num_workers=0,
        )
        frames[split] = df

    return loaders, frames


def evaluate(model, loader, device):
    """Returns (y_true, y_pred, positive-class probability)."""
    import numpy as np
    import torch

    model.eval()
    trues, preds, scores = [], [], []

    with torch.no_grad():
        for ids, lengths, labels in loader:
            logits = model(ids.to(device), lengths.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds.append(logits.argmax(dim=1).cpu().numpy())
            scores.append(probs.cpu().numpy())
            trues.append(labels.numpy())

    return (np.concatenate(trues), np.concatenate(preds), np.concatenate(scores))


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.allow_multi_gpu:
        pin_single_gpu()

    import numpy as np
    import torch
    import torch.nn as nn

    from src.metrics import classification_metrics, macro_f1
    from src.models.bilstm import BiLSTMClassifier, class_weights
    from src.models.encoding import load_vocab, unk_rate
    from src.utils import get_device_count, log_run, set_seed

    seed = set_seed(args.seed)
    device_count = get_device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_count > 1 and not args.allow_multi_gpu:
        # A warning would scroll past in a notebook. This one has to stop the
        # run: DataParallel would double the effective batch, and if it happened
        # to only some of runs 3-6 the ablation would compare batch sizes.
        raise SystemExit(
            f"{device_count} GPUs are visible, so the effective batch would be "
            f"{args.batch_size * device_count}, not {args.batch_size}. Runs 3-6 "
            "must all see the same device count (PLAN F8).\n"
            "Fix: set CUDA_VISIBLE_DEVICES to a single device before importing "
            "torch, or pass --allow-multi-gpu on EVERY run in the ablation."
        )

    vocab, index = load_vocab(args.matrices / "vocab.json")
    matrix_path = args.matrices / f"{args.embedding}.npy"
    matrix = np.load(matrix_path)

    if matrix.shape[0] != len(vocab):
        raise ValueError(
            f"{matrix_path.name} has {matrix.shape[0]} rows but vocab.json has "
            f"{len(vocab)}. The matrices and the vocabulary must come from the "
            "same run of scripts/build_embedding_matrices.py."
        )

    loaders, frames = build_loaders(args, index)
    rate, unks, total = unk_rate(frames["train"].text, index)

    print(f"run {args.run_id} | {args.embedding} - {EMBEDDINGS[args.embedding]}")
    print(f"  vocab {len(vocab):,} x {matrix.shape[1]}  |  train UNK rate {rate:.4f} "
          f"({unks:,}/{total:,})")
    print(f"  device {device} | device_count {device_count} | per-device batch "
          f"{args.batch_size} | effective {args.batch_size * max(device_count, 1)}")

    model = BiLSTMClassifier(
        matrix, num_classes=2, hidden_dim=args.hidden_dim, dropout=args.dropout,
        freeze_embeddings=not args.unfreeze_embeddings,
    ).to(device)

    print(f"  embeddings {'fine-tuned' if args.unfreeze_embeddings else 'FROZEN'} "
          f"| trainable params {model.trainable_parameters():,}")

    weights = None
    if not args.no_class_weights:
        weights = class_weights(frames["train"].label.values).to(device)
        print(f"  class weights {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    best = {"dev_macro_f1": -1.0, "epoch": 0, "state": None}
    history, t0 = [], time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, batches = 0.0, 0

        for ids, lengths, labels in loaders["train"]:
            optimiser.zero_grad()
            loss = criterion(model(ids.to(device), lengths.to(device)),
                             labels.to(device))
            loss.backward()
            # Exploding gradients are the standard LSTM failure and clipping is
            # cheap; without it a single bad batch can wipe an otherwise good run.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            running += loss.item()
            batches += 1

        y_true, y_pred, _ = evaluate(model, loaders["dev"], device)
        dev_f1 = macro_f1(y_true, y_pred)
        history.append({"epoch": epoch, "train_loss": running / batches,
                        "dev_macro_f1": dev_f1})

        marker = ""
        if dev_f1 > best["dev_macro_f1"]:
            best = {"dev_macro_f1": dev_f1, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            marker = "  <- best"

        print(f"  epoch {epoch:>2}  loss {running / batches:.4f}  "
              f"dev macro-F1 {dev_f1:.4f}{marker}")

        if epoch - best["epoch"] >= PATIENCE:
            print(f"  early stop: {PATIENCE} epochs without improvement")
            break

    train_secs = time.time() - t0

    # Test is scored once, on the checkpoint dev chose.
    model.load_state_dict(best["state"])
    y_true, y_pred, scores = evaluate(model, loaders["test"], device)

    metrics = classification_metrics(y_true, y_pred, scores)
    metrics.update({
        "dev_macro_f1": best["dev_macro_f1"],
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "train_seconds": round(train_secs, 1),
        "train_unk_rate": round(rate, 5),
    })

    print(f"\n  TEST macro-F1 {metrics['macro_f1']:.4f} | ADE F1 {metrics['f1_ade']:.4f} "
          f"| ADE P {metrics['precision_ade']:.3f} R {metrics['recall_ade']:.3f} "
          f"| PR-AUC {metrics['pr_auc']:.3f}")
    print(f"  best epoch {best['epoch']}/{len(history)} | {train_secs:.0f}s")

    out = args.out / f"run{args.run_id}_{args.embedding}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "config": model.config,
                "embedding": args.embedding, "run_id": args.run_id},
               out / "checkpoint.pt")
    np.savez(out / "test_predictions.npz",
             y_true=y_true, y_pred=y_pred, score=scores)
    (out / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "history": history,
                    "args": {k: str(v) for k, v in vars(args).items()}},
                   indent=2), encoding="utf-8")

    log_run(
        run_id=args.run_id, stage="1", model="bilstm_attn", embedding=args.embedding,
        metrics=metrics,
        params={**model.config, "max_len": MAX_LEN, "patience": PATIENCE,
                "optimiser": "adam", "grad_clip": 5.0,
                "class_weights": not args.no_class_weights,
                "early_stop_on": "dev macro_f1",
                "torch": torch.__version__},
        seed=seed, dataset_version=args.dataset_version,
        per_device_batch=args.batch_size, device_count=device_count,
        epochs=len(history), lr=args.lr,
        notes=(f"Step 4.3 run {args.run_id}; {EMBEDDINGS[args.embedding]}; "
               f"embeddings {'fine-tuned' if args.unfreeze_embeddings else 'frozen'}; "
               f"best epoch {best['epoch']}."),
    )

    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
