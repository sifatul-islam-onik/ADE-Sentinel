"""Step 0.5 - publish large artefacts as a VERSIONED private Kaggle Dataset.

Why versioned rather than a Drive folder (PLAN F7): every row in results/runs.csv
records the dataset version it consumed. A Drive folder has no version identity,
so once its contents change you can no longer say which inputs produced which
number. That is the difference between a result and an anecdote.

What goes here: the PubMed corpus, trained w2v/fastText vectors, and the E0-E3
embedding matrices. What does NOT go here: data/splits/, which is small enough
to commit to git and is therefore reproduced by a clone.

Prerequisites (MANUAL - see README):
  1. pip install kaggle   (already in requirements-local.txt)
  2. Kaggle -> Settings -> API -> "Create New Token" -> kaggle.json
  3. Move it to  %USERPROFILE%\\.kaggle\\kaggle.json
  4. Set KAGGLE_USERNAME below or via the environment

Usage:
    python scripts/push_to_kaggle.py --init            # first time only
    python scripts/push_to_kaggle.py -m "add w2v+fasttext vectors"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGING = REPO_ROOT / "data" / ".kaggle_staging"

DATASET_SLUG = "ade-sentinel-artifacts"
DATASET_TITLE = "ADE-Sentinel artifacts"

# Artefacts to upload, as (path relative to repo root, required?).
PAYLOAD: list[tuple[str, bool]] = [
    ("data/pubmed_corpus.jsonl", False),
    ("data/pubmed_sentences.txt", False),
    ("models/w2v.kv", False),
    ("models/ft.kv", False),
    ("models/emb_matrices", False),
]


def kaggle_username() -> str:
    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    if user:
        return user
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if cred.exists():
        return json.loads(cred.read_text()).get("username", "")
    return ""


def find_kaggle_cli() -> str:
    """Locate the kaggle CLI.

    `shutil.which` alone is not enough: running `.venv\\Scripts\\python script.py`
    without activating the venv leaves the venv's Scripts directory off PATH, so
    the CLI installed right next to the interpreter is invisible. Check there too.
    `python -m kaggle` is not an option - the package ships no __main__.
    """
    found = shutil.which("kaggle")
    if found:
        return found

    scripts_dir = Path(sys.executable).parent
    for name in ("kaggle.exe", "kaggle"):
        candidate = scripts_dir / name
        if candidate.exists():
            return str(candidate)

    sys.exit(
        "kaggle CLI not found. Install it into the environment you are running:\n"
        f"  {sys.executable} -m pip install kaggle"
    )


def check_cli() -> str:
    cli = find_kaggle_cli()
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists() and not os.environ.get("KAGGLE_KEY"):
        sys.exit(
            f"No credentials at {cred}.\n"
            "Kaggle -> Settings -> API -> Create New Token, then move kaggle.json there."
        )
    return cli


def stage(user: str) -> int:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    staged = 0
    for rel, required in PAYLOAD:
        src = REPO_ROOT / rel
        if not src.exists():
            if required:
                sys.exit(f"missing required artefact: {rel}")
            print(f"  skip (absent) : {rel}")
            continue
        dst = STAGING / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        size_mb = sum(f.stat().st_size for f in ([src] if src.is_file() else src.rglob("*")) if f.is_file()) / 1e6
        print(f"  staged        : {rel}  ({size_mb:,.1f} MB)")
        staged += 1

    (STAGING / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_TITLE,
        "id": f"{user}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2), encoding="utf-8")
    return staged


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd))
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true",
                    help="create the dataset (first run only)")
    ap.add_argument("-m", "--message", default="update artefacts",
                    help="version note - make it specific, it is your provenance record")
    args = ap.parse_args(argv)

    cli = check_cli()
    user = kaggle_username()
    if not user:
        sys.exit("Could not determine Kaggle username. Set KAGGLE_USERNAME.")

    print(f"dataset: {user}/{DATASET_SLUG}")
    if stage(user) == 0:
        sys.exit("\nNothing to upload yet. Run this after Phase 1.4 / Phase 3.")

    if args.init:
        code = run([cli, "datasets", "create", "-p", str(STAGING),
                    "--dir-mode", "zip"])
        print("\nNOTE: new datasets are PUBLIC by default.")
        print("      Open the dataset on kaggle.com -> Settings -> set to Private.")
    else:
        code = run([cli, "datasets", "version", "-p", str(STAGING),
                    "-m", args.message, "--dir-mode", "zip"])

    if code == 0:
        print("\n[ok] uploaded. Record the new version number in results/runs.csv")
        print(f"     https://www.kaggle.com/datasets/{user}/{DATASET_SLUG}")
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
