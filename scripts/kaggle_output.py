"""Download a Kaggle notebook's saved output files.

Why this exists rather than `kaggle kernels output`:

Kaggle now issues `KGAT_`-prefixed access tokens, which authenticate as
`Authorization: Bearer <token>`. The `kaggle` CLI pinned at 1.7.4.5 -- the newest
release installable on Python 3.10 -- only does HTTP Basic auth with the
username/key pair in kaggle.json, so it returns 401 for these tokens. The CLI
release that handles them (2.2.4) requires Python >= 3.11.

Rather than force a Python upgrade for one API call, this talks to the REST API
directly with the Bearer token.

IMPORTANT -- what "output" means here: the API serves files only from a **saved
version** of a notebook. Files sitting in /kaggle/working during an interactive
session are not exposed. If this reports no runs, open the notebook and use
Save Version, then rerun.

Setup (already done if you followed the README):
    setx KAGGLE_API_TOKEN "KGAT_..."
    setx KAGGLE_USERNAME  "your-kaggle-username"

Usage:
    python scripts/kaggle_output.py notebook13a5828c6c --list
    python scripts/kaggle_output.py notebook13a5828c6c -o data/
    python scripts/kaggle_output.py notebook13a5828c6c -o data/ -f pubmed_corpus.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
API = "https://www.kaggle.com/api/v1"


def auth_headers() -> dict[str, str]:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        sys.exit(
            "KAGGLE_API_TOKEN is not set.\n"
            '  setx KAGGLE_API_TOKEN "KGAT_..."   (then open a new terminal)'
        )
    return {"Authorization": f"Bearer {token}"}


def get_json(path: str, params: dict) -> dict | list:
    resp = requests.get(f"{API}/{path}", params=params, headers=auth_headers(), timeout=60)
    if resp.status_code == 401:
        sys.exit("401 Unauthorized - the token is invalid or expired. Regenerate it on Kaggle.")
    try:
        return resp.json()
    except ValueError:
        sys.exit(f"Unexpected non-JSON response ({resp.status_code}): {resp.text[:300]}")


def kernel_files(user: str, slug: str) -> list[dict]:
    data = get_json("kernels/output", {"userName": user, "kernelSlug": slug})

    if isinstance(data, dict) and data.get("code") in (403, 404):
        status = get_json("kernels/status", {"userName": user, "kernelSlug": slug})
        detail = status.get("message") if isinstance(status, dict) else status
        sys.exit(
            f"No downloadable output for {user}/{slug}.\n"
            f"  API says: {detail}\n\n"
            "The API only serves files from a SAVED VERSION. A notebook that was run\n"
            "interactively has files in /kaggle/working that are not exposed.\n\n"
            "Fix: open the notebook -> Save Version -> Quick Save -> wait for it to\n"
            "finish, then rerun this command."
        )

    files = data.get("files", []) if isinstance(data, dict) else []
    if not files:
        sys.exit(f"The saved version of {user}/{slug} contains no output files.")
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug", help="notebook slug, e.g. notebook13a5828c6c")
    ap.add_argument("-u", "--user", default=os.environ.get("KAGGLE_USERNAME", ""))
    ap.add_argument("-o", "--out", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("-f", "--file", help="download only this filename")
    ap.add_argument("--list", action="store_true", help="list files, download nothing")
    args = ap.parse_args(argv)

    if not args.user:
        sys.exit("No Kaggle username. Pass --user or set KAGGLE_USERNAME.")

    files = kernel_files(args.user, args.slug)

    print(f"{args.user}/{args.slug} - {len(files)} output file(s):")
    for f in files:
        size = f.get("fileSize")
        size_s = f"{size / 1e6:,.1f} MB" if isinstance(size, int) else "?"
        print(f"  {f.get('fileName'):50s} {size_s:>12s}")

    if args.list:
        return 0

    wanted = [f for f in files if not args.file or f.get("fileName") == args.file]
    if not wanted:
        sys.exit(f"\nNo output file named {args.file!r}.")

    args.out.mkdir(parents=True, exist_ok=True)
    for f in wanted:
        name, url = f.get("fileName"), f.get("url")
        dest = args.out / name
        print(f"\ndownloading {name} -> {dest.relative_to(REPO_ROOT)}")

        with requests.get(url, headers=auth_headers(), stream=True, timeout=600) as r:
            r.raise_for_status()
            written = 0
            with dest.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    written += len(chunk)
                    print(f"\r  {written / 1e6:,.1f} MB", end="", flush=True)
        print(f"\r  done: {written / 1e6:,.1f} MB")

        if name.endswith(".jsonl"):
            with dest.open(encoding="utf-8") as fh:
                lines = sum(1 for _ in fh)
            print(f"  {lines:,} records")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
