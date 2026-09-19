"""Reproducibility spine: seeding, and the run log.

Two jobs:

  set_seed()  - seed every RNG that can affect a run.
  log_run()   - append one row to results/runs.csv, capturing not just the
                score but the provenance: git commit, GPU count, effective
                batch size, seed.

Every experiment in the project went through `log_run`. Nothing in
`results/runs.csv` was typed by hand, which is why the notebooks can read their
tables straight out of it.

**The GPU count matters more than it looks.** Hugging Face's `Trainer`
multiplies `per_device_train_batch_size` by the number of visible devices, so
the same config trains at batch 16 on one T4 and batch 32 on two. The embedding
ablation (runs 3-6) is only valid if that number is identical across all four
runs, so it is a column rather than a footnote.

torch is imported lazily throughout, so this module stays importable on a
machine with no torch installed.
"""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_CSV = REPO_ROOT / "results" / "runs.csv"

DEFAULT_SEED = 42

# Fixed column order. Anything not named here is folded into metrics_json, so
# adding a metric later never invalidates rows already written.
FIELDNAMES = [
    "timestamp", "run_id", "stage", "model", "embedding", "seed",
    "git_commit", "git_dirty", "dataset_version", "device_count",
    "per_device_batch", "effective_batch", "epochs", "lr",
    "macro_f1", "entity_f1_strict", "entity_f1_lenient",
    "metrics_json", "params_json", "notes",
]


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed python, numpy and torch. Returns the seed, for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism costs throughput but makes the embedding ablation
        # defensible: runs 3-6 must differ only in the embedding matrix.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def get_git_commit() -> tuple[str, bool]:
    """(short commit hash, working tree is dirty).

    A dirty tree means the logged commit does not fully describe the code that
    produced the row, so the flag is recorded rather than quietly dropped.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
        return commit, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def get_device_count() -> int:
    """Visible CUDA devices. 0 on a CPU laptop, 1 or 2 on Kaggle."""
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def pin_single_gpu() -> None:
    """Restrict the process to one CUDA device.

    Two things make this a function rather than one line at the call site:

    1. **It must run before anything initialises CUDA.** `CUDA_VISIBLE_DEVICES`
       is read once, when the context is created. Set it after even a
       `torch.cuda.device_count()` call and it is silently ignored for the life
       of the process. So every caller runs this *before importing torch*.
    2. **`setdefault`, not assignment.** An explicit `CUDA_VISIBLE_DEVICES=1`
       in the environment is someone choosing the second card because the first
       is busy; overriding that would be wrong.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def derive_per_device_batch(effective: int, device_count: int) -> int:
    """Split an intended *effective* batch across the visible GPUs.

    The inverse of what `Trainer` does. It takes a per-device batch and
    multiplies by the device count, so a recipe saying "batch 16" silently
    becomes 32 on Kaggle's 2xT4. Stating the effective batch and deriving
    per-device from it means the recipe in the report is the recipe that ran.

    Raises rather than rounding: a run whose recipe does not match the report is
    worse than a run that refused to start.
    """
    devices = max(device_count, 1)
    if effective % devices:
        raise ValueError(
            f"effective batch {effective} is not divisible by {devices} visible "
            f"GPUs. Pass a multiple of {devices}, or pin CUDA_VISIBLE_DEVICES "
            f"to a single device.")
    return effective // devices


def log_run(
    run_id: str | int,
    stage: str,
    model: str,
    *,
    embedding: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
    dataset_version: str | None = None,
    per_device_batch: int | None = None,
    device_count: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    notes: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    """Append one run to results/runs.csv, creating it with a header if absent.

    `effective_batch` is derived, not asked for: per_device_batch times the
    visible device count is the number that actually describes the run.
    """
    metrics = dict(metrics or {})
    params = dict(params or {})

    if device_count is None:
        device_count = get_device_count()

    effective_batch = None
    if per_device_batch is not None:
        effective_batch = per_device_batch * max(device_count, 1)

    commit, dirty = get_git_commit()

    row: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "stage": stage,
        "model": model,
        "embedding": embedding or "",
        "seed": seed,
        "git_commit": commit,
        "git_dirty": int(dirty),
        "dataset_version": dataset_version or "",
        "device_count": device_count,
        "per_device_batch": per_device_batch if per_device_batch is not None else "",
        "effective_batch": effective_batch if effective_batch is not None else "",
        "epochs": epochs if epochs is not None else "",
        "lr": lr if lr is not None else "",
        # Promoted metrics get their own column so the notebook tables are a
        # straight read of the CSV; everything else survives in metrics_json.
        "macro_f1": metrics.get("macro_f1", ""),
        "entity_f1_strict": metrics.get("entity_f1_strict", ""),
        "entity_f1_lenient": metrics.get("entity_f1_lenient", ""),
        "metrics_json": json.dumps(metrics, sort_keys=True, default=str),
        "params_json": json.dumps(params, sort_keys=True, default=str),
        "notes": notes,
    }

    target = path or RUNS_CSV
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists() or target.stat().st_size == 0

    with target.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return row
