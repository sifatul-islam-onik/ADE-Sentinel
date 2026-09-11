"""Gate for Phase 6: does local CPU inference reproduce the remote decode exactly?

    python scripts/check_stage2_inference.py

Phase 6.1 runs the Stage 2 taggers over sentences the remote session never
decoded - Stage 1's false positives - so it needs inference here, on CPU. That
is a measurement of the pipeline only if this path makes the same decisions the
remote one did. This re-decodes the test split from each saved checkpoint and
compares tag for tag with the saved predictions.

It also records wall-clock cost on this machine, which is the number PLAN 7.2's
choice of demo model turns on. Run it on an otherwise idle machine, or the
timings measure the contention rather than the models.

Writes results/stage2_reproduction.json. Exits non-zero if any run differs.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

STAGE2 = REPO_ROOT / "models" / "stage2"
MATRICES = REPO_ROOT / "models" / "emb_matrices"
OUT = REPO_ROOT / "results" / "stage2_reproduction.json"


def main() -> int:
    import torch

    from src.stage2_inference import BertTagger, BiLSTMTaggerRunner, reproduce_saved

    taggers = {
        "9": ("run9_softmax",
              lambda: BiLSTMTaggerRunner(STAGE2 / "run9_softmax" / "checkpoint.pt", MATRICES)),
        "10": ("run10_crf",
               lambda: BiLSTMTaggerRunner(STAGE2 / "run10_crf" / "checkpoint.pt", MATRICES)),
        "11": ("run11_biomedbert",
               lambda: BertTagger(STAGE2 / "run11_biomedbert" / "best")),
    }

    results, all_exact = {}, True
    for run_id, (directory, build) in taggers.items():
        path = STAGE2 / directory / "test_predictions.json"
        if not path.exists():
            print(f"run {run_id}: no saved predictions, skipped")
            continue

        t0 = time.perf_counter()
        tagger = build()
        load = time.perf_counter() - t0

        t0 = time.perf_counter()
        check = reproduce_saved(tagger, path)
        decode = time.perf_counter() - t0

        check = {k: v for k, v in check.items() if k not in ("gold", "pred")}
        check.update({"load_seconds": round(load, 2), "decode_seconds": round(decode, 2)})
        results[run_id] = check

        exact = (check["identical_sentences"] == check["sentences"]
                 and check["misaligned_sentences"] == 0)
        all_exact &= exact
        print(f"run {run_id}: {check['identical_sentences']}/{check['sentences']} sentences "
              f"identical, {check['differing_tokens']} tokens differ | load {load:.1f}s, "
              f"decode {decode:.1f}s | {'EXACT' if exact else 'DIFFERS'}")

    OUT.write_text(json.dumps({
        "runs": results,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "processor": platform.processor(),
    }, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
