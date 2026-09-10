"""Step 4.1 - Stage 1 sparse baselines. Runs 1, 2 (and 2b).

| Run | Model | Features | PRD tier |
|-----|-------|----------|----------|
| 1   | Multinomial Naive Bayes | 1-2gram counts | T1, generative |
| 2   | Logistic Regression     | 1-2gram TF-IDF | T2, discriminative |
| 2b  | Linear SVM              | 1-2gram TF-IDF | T2 alternative |

These take seconds and they are not decoration. The embedding ablation (runs
3-6) only means something against a floor, and a well-tuned TF-IDF linear model
is a much harder floor than an untuned one - "our embeddings beat a baseline we
did not bother to tune" is the criticism this step exists to pre-empt.

Three choices that keep the comparison honest against the neural runs:

**The project tokenizer, not sklearn's.** `\\w+` would split `5-fluorouracil`
and lowercase `ALL` into a stopword. Every model in this project sees the same
tokens; the tokenizer is a controlled variable, not a per-model detail.

**Selection on dev, reporting on test.** Each model gets a small grid, chosen by
dev macro-F1. `class_weight` is *in* the grid rather than fixed: under 1:4
imbalance it is a real modelling decision, and letting dev settle it is more
defensible than asserting it.

**Fit on train only.** Refitting the winner on train+dev would beat the neural
runs, which need dev for early stopping, on data volume alone.

Usage:  .venv\\Scripts\\python scripts\\train_baselines.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.metrics import classification_metrics, format_confusion  # noqa: E402
from src.tokenizer import tokenize  # noqa: E402
from src.utils import DEFAULT_SEED, log_run, set_seed  # noqa: E402

SPLITS = REPO_ROOT / "data" / "splits"
REPORT = REPO_ROOT / "results" / "figures" / "stage1_baselines.md"

NGRAM = (1, 2)
MIN_DF = 2          # a feature seen once cannot generalise; it only inflates the matrix


def project_tokenizer(text: str) -> list[str]:
    """The `tokenize` half of the project tokenizer, shaped for sklearn.

    Drops tokens with no alphanumeric character, matching how the neural
    vocabulary was built (`src.models.encoding.is_indexable`), so the sparse and
    dense models see the same token stream.
    """
    tokens, _ = tokenize(text)
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


def load_splits():
    import pandas as pd

    return tuple(
        pd.read_parquet(SPLITS / f"stage1_{s}.parquet") for s in ("train", "dev", "test")
    )


def decision_scores(model, X):
    """Positive-class score for PR-AUC. LinearSVC has no `predict_proba`."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def grid_search(build, grid, Xtr, ytr, Xdev, ydev):
    """Smallest thing that does the job: fit each config, keep the best dev macro-F1."""
    from sklearn.metrics import f1_score

    trials = []
    for params in grid:
        model = build(**params)
        model.fit(Xtr, ytr)
        dev_f1 = f1_score(ydev, model.predict(Xdev), average="macro", zero_division=0)
        trials.append((dev_f1, params, model))

    trials.sort(key=lambda t: t[0], reverse=True)
    return trials[0], trials


def top_features(vectorizer, model, n: int = 12) -> tuple[list[str], list[str]]:
    """Most positively and negatively weighted features of a linear model.

    A report figure and a sanity check at once: if the top ADE features were
    function words, something upstream would be wrong.
    """
    import numpy as np

    names = np.asarray(vectorizer.get_feature_names_out())
    weights = model.coef_[0]
    order = np.argsort(weights)
    return list(names[order[-n:]][::-1]), list(names[order[:n]])


def main() -> int:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    seed = set_seed(DEFAULT_SEED)
    train, dev, test = load_splits()
    ytr, ydev, yte = train.label.values, dev.label.values, test.label.values

    print(f"train {len(train):,} | dev {len(dev):,} | test {len(test):,} "
          f"| positive rate {ytr.mean():.3f}")

    shared = dict(tokenizer=project_tokenizer, lowercase=False,
                  ngram_range=NGRAM, min_df=MIN_DF, token_pattern=None)

    count_vec = CountVectorizer(**shared)
    tfidf_vec = TfidfVectorizer(**shared, sublinear_tf=True)

    t0 = time.time()
    Xtr_c = count_vec.fit_transform(train.text)
    Xdev_c, Xte_c = count_vec.transform(dev.text), count_vec.transform(test.text)
    Xtr_t = tfidf_vec.fit_transform(train.text)
    Xdev_t, Xte_t = tfidf_vec.transform(dev.text), tfidf_vec.transform(test.text)
    vec_secs = time.time() - t0
    print(f"vectorised in {vec_secs:.1f}s | count {Xtr_c.shape[1]:,} features "
          f"| tfidf {Xtr_t.shape[1]:,} features")

    specs = [
        ("1", "naive_bayes", "count-1-2gram", MultinomialNB,
         [{"alpha": a} for a in (0.1, 0.3, 1.0, 3.0)],
         (Xtr_c, Xdev_c, Xte_c), count_vec),
        ("2", "logreg", "tfidf-1-2gram",
         lambda **kw: LogisticRegression(max_iter=2000, random_state=seed, **kw),
         [{"C": c, "class_weight": cw} for c in (0.5, 1.0, 4.0, 16.0)
          for cw in (None, "balanced")],
         (Xtr_t, Xdev_t, Xte_t), tfidf_vec),
        ("2b", "linear_svm", "tfidf-1-2gram",
         lambda **kw: LinearSVC(random_state=seed, **kw),
         [{"C": c, "class_weight": cw} for c in (0.1, 0.5, 1.0, 4.0)
          for cw in (None, "balanced")],
         (Xtr_t, Xdev_t, Xte_t), tfidf_vec),
    ]

    results = []
    for run_id, name, features, build, grid, (Xtr, Xdev, Xte), vec in specs:
        t0 = time.time()
        (dev_f1, params, model), trials = grid_search(build, grid, Xtr, ytr, Xdev, ydev)
        secs = time.time() - t0

        metrics = classification_metrics(yte, model.predict(Xte),
                                         decision_scores(model, Xte))
        metrics["dev_macro_f1"] = float(dev_f1)
        metrics["train_seconds"] = round(secs, 2)

        results.append({
            "run_id": run_id, "name": name, "features": features, "params": params,
            "metrics": metrics, "vec": vec, "model": model,
            "n_features": int(Xtr.shape[1]), "trials": trials,
        })

        print(f"\nrun {run_id}  {name}  {params}")
        print(f"  dev macro-F1 {dev_f1:.4f} | test macro-F1 {metrics['macro_f1']:.4f} "
              f"| ADE F1 {metrics['f1_ade']:.4f} | PR-AUC {metrics['pr_auc']:.4f} "
              f"| {secs:.1f}s over {len(grid)} configs")

        log_run(
            run_id=run_id, stage="1", model=name, embedding=features,
            metrics=metrics,
            params={**{k: str(v) for k, v in params.items()},
                    "ngram_range": list(NGRAM), "min_df": MIN_DF,
                    "n_features": int(Xtr.shape[1]),
                    "tokenizer": "src.tokenizer.tokenize",
                    "selected_on": "dev macro_f1", "fit_on": "train only",
                    "grid_size": len(grid)},
            seed=seed, notes=f"Step 4.1 sparse baseline; sklearn, CPU, {secs:.1f}s.",
        )

    write_report(results, train, dev, test, vec_secs)
    print(f"\nwrote {REPORT.relative_to(REPO_ROOT)}")
    print(f"appended runs {', '.join(r['run_id'] for r in results)} to results/runs.csv")
    return 0


def write_report(results, train, dev, test, vec_secs) -> None:
    majority_acc = 1.0 - test.label.mean()

    lines = [
        "# Stage 1 - sparse baselines (step 4.1, runs 1-2)",
        "",
        "Produced by `scripts/train_baselines.py`. Trained on the frozen split "
        f"(`data/splits/stage1_*.parquet`): {len(train):,} train / {len(dev):,} dev / "
        f"{len(test):,} test, {train.label.mean():.1%} positive.",
        "",
        "Hyperparameters were selected on **dev** macro-F1 and the winner is reported "
        "on **test**, fitted on train only - the neural runs need dev for early "
        "stopping, so refitting these on train+dev would hand them a data advantage "
        "that has nothing to do with the model.",
        "",
        "## Test-set results",
        "",
        "| Run | Model | Features | Macro-F1 | ADE P | ADE R | ADE F1 | PR-AUC | Acc. |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r["metrics"]
        lines.append(
            f"| {r['run_id']} | {r['name']} | {r['features']} | **{m['macro_f1']:.4f}** | "
            f"{m['precision_ade']:.3f} | {m['recall_ade']:.3f} | {m['f1_ade']:.3f} | "
            f"{m['pr_auc']:.3f} | {m['accuracy']:.3f} |"
        )

    lines += [
        "",
        f"Majority-class accuracy on this test split is **{majority_acc:.3f}**. Every "
        "model above beats it on accuracy while differing sharply on macro-F1, which is "
        "the concrete reason PRD section 5 deprecates accuracy for this task.",
        "",
        "## Selected hyperparameters",
        "",
        "| Run | Chosen | Grid | Dev macro-F1 | Features | Fit time |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        params = ", ".join(f"`{k}={v}`" for k, v in r["params"].items())
        lines.append(
            f"| {r['run_id']} | {params} | {len(r['trials'])} configs | "
            f"{r['metrics']['dev_macro_f1']:.4f} | {r['n_features']:,} | "
            f"{r['metrics']['train_seconds']:.1f}s |"
        )

    lines += ["", "## Confusion matrices (test)", ""]
    for r in results:
        lines += [f"**Run {r['run_id']} - {r['name']}**", ""]
        lines += format_confusion(r["metrics"]["confusion_matrix"])
        lines += [""]

    linear = [r for r in results if hasattr(r["model"], "coef_")]
    if linear:
        lines += [
            "## Most informative features",
            "",
            "Top-weighted 1-2grams from the linear models. This is a sanity check as "
            "much as a figure: if the ADE side were dominated by function words, the "
            "tokenizer or the split would be wrong.",
            "",
        ]
        for r in linear:
            pos, neg = top_features(r["vec"], r["model"])
            lines += [
                f"**Run {r['run_id']} - {r['name']}**",
                "",
                "| Direction | Features |",
                "|---|---|",
                "| toward **ADE** | " + ", ".join(f"`{f}`" for f in pos) + " |",
                "| toward **not-ADE** | " + ", ".join(f"`{f}`" for f in neg) + " |",
                "",
            ]

    lines += [
        "## Reading these numbers",
        "",
        "These are the floor the embedding ablation has to clear. A BiLSTM that fails "
        "to beat TF-IDF + LogReg is not evidence that embeddings do not help - it is "
        "evidence that the BiLSTM is undertrained, and should be diagnosed as such "
        "before any conclusion is drawn from runs 3-6.",
        "",
        f"Vectorisation cost {vec_secs:.1f}s and all three models fit in seconds on CPU. "
        "That asymmetry belongs in the report: the transformer runs cost roughly four "
        "orders of magnitude more compute for the margin recorded in "
        "`stage1_results.md`.",
        "",
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
