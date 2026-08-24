"""Reproducibility spine for ADE-Sentinel.

Two jobs, both required by PLAN.md F8/F10:

  set_seed()  - seed every RNG that can affect a run.
  log_run()   - append one row to results/runs.csv, capturing not just the
                metrics but the *provenance*: git commit, Kaggle Dataset
                version, GPU count and effective batch size.

The GPU count matters more than it looks. HF Trainer multiplies
`per_device_train_batch_size` by the visible device count, so the same config
trains at batch 16 on one T4 and batch 32 on two. Runs 3-6 (the embedding
ablation) are only valid if that number is identical across all four, which is
why it is a first-class column rather than a note in a notebook.

torch is imported lazily throughout: the local environment deliberately has no
torch (PLAN F7), and this module must stay importable there.
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
    "timestamp",
    "run_id",
    "stage",
    "model",
    "embedding",
    "seed",
    "git_commit",
    "git_dirty",
    "dataset_version",
    "device_count",
    "per_device_batch",
    "effective_batch",
    "epochs",
    "lr",
    "macro_f1",
    "entity_f1_strict",
    "entity_f1_lenient",
    "metrics_json",
    "params_json",
    "notes",
]


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed python, numpy and (if present) torch. Returns the seed for logging."""
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
        # Determinism costs throughput but makes the embedding ablation defensible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def get_git_commit() -> tuple[str, bool]:
    """Return (short commit hash, working-tree-is-dirty).

    A dirty tree means the logged commit does not fully describe the code that
    produced the row, so the flag is recorded rather than silently dropped.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def get_device_count() -> int:
    """Visible CUDA devices. 0 locally, 1 or 2 on Kaggle depending on settings."""
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


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

    `effective_batch` is derived, not asked for: it is per_device_batch times the
    visible device count, which is the number that actually describes the run
    (PLAN F8).
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
        # Promoted metrics get their own column so the report tables are a
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


def _selftest() -> None:
    """Step 0.4 exit criterion: prove a dummy run reaches results/runs.csv."""
    seed = set_seed()
    row = log_run(
        run_id="0-selftest",
        stage="0",
        model="selftest",
        embedding="none",
        metrics={"macro_f1": 0.0, "note": "not a real result"},
        params={"purpose": "verify log_run plumbing"},
        seed=seed,
        per_device_batch=8,
        notes="Phase 0.4 plumbing check. Delete this row before Phase 4.",
    )
    print(f"wrote row to {RUNS_CSV}")
    for key in ("run_id", "git_commit", "git_dirty", "device_count", "effective_batch"):
        print(f"  {key:16s} {row[key]}")


if __name__ == "__main__":
    _selftest()
