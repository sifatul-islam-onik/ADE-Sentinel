"""Steps 3.1-3.2 - train the domain embeddings (E2 Word2Vec, E3 FastText).

These are the project's headline contribution (PRD section 7): vectors trained on
our own biomedical corpus, to be compared against general-purpose GloVe three
independent ways - coverage, nearest neighbours, and downstream F1.

Hyperparameters come from the PRD and are IDENTICAL for both models. That is the
whole point: E2 vs E3 must isolate the effect of subword information, so anything
else that differs would confound the comparison.

    skip-gram (sg=1), 300d, window 5, min_count 5, negative 10, 5 epochs

FastText is included deliberately. Biomedical morphology is unusually regular -
`-emia`, `-itis`, `-osis`, `-toxicity`, `-pathy` - so subword units should reach
rare drug and condition names that Word2Vec can only treat as OOV. If E3 beats
E2, that morphology is the explanation; if it does not, that is a finding too.

Both are CPU-only, so this runs locally. gensim never touches a GPU.

Usage:
    .venv\\Scripts\\python scripts\\train_embeddings.py --model w2v
    .venv\\Scripts\\python scripts\\train_embeddings.py --model ft
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SENTENCES = REPO_ROOT / "data" / "pubmed_sentences.txt"
MODEL_DIR = REPO_ROOT / "models"

# PRD section 7.1 / Phase 3. Shared by both models, deliberately.
PARAMS = dict(
    vector_size=300,
    window=5,
    min_count=5,
    negative=10,
    epochs=5,
    sg=1,            # skip-gram, not CBOW: better for rare words, and the rare
                     # words here are the drug names the whole project is about
    sample=1e-4,
    seed=42,
)


def train(kind: str, workers: int) -> Path:
    from gensim.models import FastText, Word2Vec
    from gensim.models.word2vec import LineSentence

    if not SENTENCES.exists():
        sys.exit(f"{SENTENCES} not found - run scripts/prepare_corpus.py first.")

    logging.basicConfig(format="    %(message)s", level=logging.INFO)
    logging.getLogger("gensim").setLevel(logging.WARNING)

    corpus = LineSentence(str(SENTENCES))     # streamed, never fully in memory
    started = time.time()

    print(f"training {kind} | {PARAMS} | workers={workers}")
    if kind == "w2v":
        model = Word2Vec(corpus, workers=workers, **PARAMS)
    else:
        # bucket controls the subword hash space. The default 2M would make the
        # saved vectors several GB for no measurable gain at this corpus size.
        model = FastText(corpus, workers=workers, bucket=500_000, **PARAMS)

    elapsed = time.time() - started
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / f"{kind}.kv"

    # Save KeyedVectors only. The full trainable model carries optimiser state
    # and the input matrix, which nothing downstream needs.
    model.wv.save(str(out))

    print(f"\n{kind}: {len(model.wv):,} vectors, {elapsed / 60:.1f} min -> {out.name}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["w2v", "ft", "both"], default="both")
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 1, 1))
    args = ap.parse_args(argv)

    kinds = ["w2v", "ft"] if args.model == "both" else [args.model]
    for kind in kinds:
        train(kind, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
