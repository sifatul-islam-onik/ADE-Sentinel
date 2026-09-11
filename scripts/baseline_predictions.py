"""Saves test predictions for the sparse baselines (runs 1, 2, 2b), refit exactly.

    python scripts/baseline_predictions.py

`train_baselines.py` logged its runs but did not save per-sentence predictions.
The neural runs do, and Phase 6 needs them from every Stage 1 tier: run 13
scores each tier on a slice of the test set, and the pipeline ladder puts each
tier in front of Stage 2.

Re-running `train_baselines.py` would re-select and re-log. Instead each
baseline is refit once with the hyperparameters its run row recorded, on the
same frozen train split with the same vectoriser, and its test macro-F1 and
PR-AUC must reproduce the logged values to 1e-9 before anything is written.
Every step is deterministic (fixed seed, no sampling), so exact agreement is
the expected outcome and anything else is a reason to stop.

Writes models/stage1/run{1,2,2b}_*/test_predictions.npz in the neural runs'
format: y_true, y_pred, score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_baselines as tb  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402
from src.utils import RUNS_CSV, set_seed  # noqa: E402

STAGE1 = REPO_ROOT / "models" / "stage1"
DIRECTORIES = {"1": "run1_naive_bayes", "2": "run2_logreg", "2b": "run2b_linear_svm"}
TOLERANCE = 1e-9


def logged(run_id: str):
    import pandas as pd

    df = pd.read_csv(RUNS_CSV, dtype={"run_id": str, "stage": str})
    rows = df[(df.stage == "1") & (df.run_id == run_id)]
    if rows.empty:
        raise SystemExit(f"run {run_id} is not in results/runs.csv")
    row = rows.iloc[-1]
    return json.loads(row.params_json), json.loads(row.metrics_json), int(row.seed)


def class_weight(value):
    """train_baselines logged every hyperparameter as a string, None included."""
    return None if value in (None, "None") else value


def main() -> int:
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    train, _, test = tb.load_splits()
    ytr, yte = train.label.values, test.label.values

    shared = dict(tokenizer=tb.project_tokenizer, lowercase=False,
                  ngram_range=tb.NGRAM, min_df=tb.MIN_DF, token_pattern=None)
    count_vec = CountVectorizer(**shared)
    tfidf_vec = TfidfVectorizer(**shared, sublinear_tf=True)
    features = {
        "count-1-2gram": (count_vec.fit_transform(train.text), count_vec.transform(test.text)),
        "tfidf-1-2gram": (tfidf_vec.fit_transform(train.text), tfidf_vec.transform(test.text)),
    }

    builders = {
        "1": ("count-1-2gram",
              lambda p, seed: MultinomialNB(alpha=float(p["alpha"]))),
        "2": ("tfidf-1-2gram",
              lambda p, seed: LogisticRegression(max_iter=2000, random_state=seed,
                                                 C=float(p["C"]),
                                                 class_weight=class_weight(p["class_weight"]))),
        "2b": ("tfidf-1-2gram",
               lambda p, seed: LinearSVC(random_state=seed, C=float(p["C"]),
                                         class_weight=class_weight(p["class_weight"]))),
    }

    status = 0
    for run_id, (feature_name, build) in builders.items():
        params, metrics, seed = logged(run_id)
        set_seed(seed)

        if list(params["ngram_range"]) != list(tb.NGRAM) or params["min_df"] != tb.MIN_DF:
            raise SystemExit(f"run {run_id} was logged with a different vectoriser than "
                             "train_baselines.py now builds")
        Xtr, Xte = features[feature_name]
        if Xtr.shape[1] != params["n_features"]:
            raise SystemExit(f"run {run_id}: {Xtr.shape[1]} features now, "
                             f"{params['n_features']} when logged")

        model = build(params, seed).fit(Xtr, ytr)
        y_pred = model.predict(Xte)
        score = tb.decision_scores(model, Xte)
        again = classification_metrics(yte, y_pred, score)

        drift = {key: abs(again[key] - metrics[key]) for key in ("macro_f1", "pr_auc")}
        if any(d > TOLERANCE for d in drift.values()):
            print(f"run {run_id}: refit does NOT reproduce the logged run "
                  f"({', '.join(f'{k} off by {v:.2e}' for k, v in drift.items())}) - not written")
            status = 1
            continue

        out = STAGE1 / DIRECTORIES[run_id]
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "test_predictions.npz", y_true=yte.astype(np.int64),
                 y_pred=np.asarray(y_pred).astype(np.int64),
                 score=np.asarray(score).astype(np.float32))
        print(f"run {run_id}: macro-F1 {again['macro_f1']:.10f}, PR-AUC {again['pr_auc']:.10f} "
              f"reproduce the logged run; wrote {out.relative_to(REPO_ROOT)}")

    return status


if __name__ == "__main__":
    raise SystemExit(main())
