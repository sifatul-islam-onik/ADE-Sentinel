"""Step 1.4 - fetch the unlabeled PubMed domain corpus.

This is a rewrite, not a tidy-up, of the sketch in PRD section 6.2. That sketch
cannot work (PLAN F1): it pages with `retstart` in steps of 10,000, and NCBI
hard-errors above 9,998 --

    "ERROR":"Search Backend failed: Exception: 'retstart' cannot be larger than
     9998. For PubMed, ESearch can only retrieve the first 9,999 records
     matching the query."

so it dies on its second iteration with ~10k abstracts instead of 100k.

The fix here is date windowing: one esearch per publication year, subdivided
into months when a year exceeds the cap (the union query below runs ~85k/year,
so month subdivision is the normal path, not the exception). No History server
needed. Each year is written to its own shard, which makes the whole job
resumable: rerun after an interruption and completed years are skipped.

QUERY CHOICE (PLAN F2): the PRD's table and its script specify different queries.
Measured, with `hasabstract` applied, the narrow MeSH-only query tops out at
92,864 records across all time -- it cannot reach the PRD's 100k floor at any
date range. So the "adverse effects" subheading is unioned in, and the result is
capped per year to keep the corpus temporally balanced rather than dominated by
whichever years happen to be largest.

Usage
-----
    python scripts/fetch_pubmed.py --out data/pubmed_corpus.jsonl

    # 3 req/s becomes 10 req/s with a free key: https://account.ncbi.nlm.nih.gov/settings/
    NCBI_API_KEY=xxxx python scripts/fetch_pubmed.py

Expect roughly 35-45 min with an API key, measured at ~18s per 1,400 abstracts.
The wall-clock time is printed at the end and belongs in
report/data_documentation.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator
from xml.etree import ElementTree as ET

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# The MeSH term alone cannot reach the PRD's 100k floor. Measured with abstracts:
#   MeSH only, 1995-2025 ................. 76,392
#   MeSH only, all time, no date limit ... 92,864
# So the "adverse effects" subheading is unioned in. `hasabstract` filters
# server-side, so we never spend an efetch on a record with nothing to read.
QUERY = (
    '("drug-related side effects and adverse reactions"[MeSH]'
    ' OR "adverse effects"[Subheading]) AND hasabstract'
)
MIN_YEAR, MAX_YEAR = 2010, 2025

# The union query returns ~85k/year, far more than needed. Cap each year and
# spread the cap evenly across that year's windows, so the corpus stays
# temporally balanced instead of being 2010-2012 plus nothing. 16 years x 10k
# targets ~160k, comfortably over the floor.
PER_YEAR_CAP = 10_000

ESEARCH_CAP = 9998        # NCBI's hard limit on retstart for PubMed
EFETCH_BATCH = 200        # records per efetch call
RETMAX = 9999             # max PMIDs one esearch may return

SHARD_DIR = REPO_ROOT / "data" / "raw" / "pubmed_shards"
DEFAULT_OUT = REPO_ROOT / "data" / "pubmed_corpus.jsonl"

def api_key() -> str:
    """Read the key on every call, not once at import.

    A module-level constant is a trap in a notebook: import the module, then set
    the key in a later cell, and the module has already cached its absence -- the
    fetch silently crawls at the anonymous rate. Reading lazily makes cell order
    irrelevant.
    """
    return os.environ.get("NCBI_API_KEY", "").strip()


def sleep_seconds() -> float:
    """NCBI allows 3 req/s anonymously, 10 req/s with a key. Stay just inside."""
    return 0.11 if api_key() else 0.35


class FetchError(RuntimeError):
    pass


def _request(endpoint: str, params: dict, *, attempts: int = 5) -> requests.Response:
    """GET with exponential backoff. NCBI throttles with 429 and sheds with 5xx."""
    key = api_key()
    if key:
        params = {**params, "api_key": key}

    delay = 1.0
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(f"{BASE}/{endpoint}", params=params, timeout=90)
            if resp.status_code == 200:
                time.sleep(sleep_seconds())
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                last = FetchError(f"HTTP {resp.status_code}")
            else:
                raise FetchError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as exc:
            last = exc

        if attempt < attempts:
            print(f"    retry {attempt}/{attempts - 1} in {delay:.0f}s ({last})")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)

    raise FetchError(f"{endpoint} failed after {attempts} attempts: {last}")


def count_hits(mindate: str, maxdate: str) -> int:
    resp = _request("esearch.fcgi", {
        "db": "pubmed", "term": QUERY, "retmax": 0, "retmode": "json",
        "datetype": "pdat", "mindate": mindate, "maxdate": maxdate,
    })
    return int(resp.json()["esearchresult"]["count"])


def get_pmids(mindate: str, maxdate: str, limit: int | None = None) -> list[str]:
    """PMIDs for one date window, newest first.

    `sort=pub_date` is not cosmetic: without an explicit sort NCBI orders by
    relevance, which is not stable across calls, so a capped fetch would return a
    different sample every run and the corpus would stop being reproducible.
    """
    resp = _request("esearch.fcgi", {
        "db": "pubmed", "term": QUERY, "retmax": min(limit or RETMAX, RETMAX),
        "retmode": "json", "sort": "pub_date",
        "datetype": "pdat", "mindate": mindate, "maxdate": maxdate,
    })
    result = resp.json()["esearchresult"]
    if "ERROR" in result:
        raise FetchError(result["ERROR"])
    ids = result.get("idlist", [])
    return ids[:limit] if limit else ids


def windows_for_year(year: int) -> list[tuple[str, str]]:
    """One window per year, subdivided into months only if the cap demands it."""
    lo, hi = f"{year}/01/01", f"{year}/12/31"
    hits = count_hits(lo, hi)
    if hits <= ESEARCH_CAP:
        return [(lo, hi)]

    print(f"  {year}: {hits:,} hits exceeds the {ESEARCH_CAP:,} cap - splitting by month")
    months = []
    for m in range(1, 13):
        last_day = 31 if m in (1, 3, 5, 7, 8, 10, 12) else 30 if m != 2 else 29
        months.append((f"{year}/{m:02d}/01", f"{year}/{m:02d}/{last_day}"))
    return months


def _text(elem) -> str:
    """Flatten an element including nested markup.

    PubMed wraps drug names and gene symbols in <i>, <sub>, <sup>. ElementTree's
    findtext() returns only the leading text node, so it silently truncates
    exactly the domain terms this corpus exists to capture.
    """
    return "".join(elem.itertext()).strip() if elem is not None else ""


def parse_articles(xml: str) -> Iterator[dict]:
    root = ET.fromstring(xml)
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title = _text(art.find(".//ArticleTitle"))

        # Structured abstracts split into several labelled AbstractText nodes.
        parts = []
        for node in art.iter("AbstractText"):
            body = _text(node)
            if not body:
                continue
            label = (node.get("Label") or "").strip()
            parts.append(f"{label}: {body}" if label else body)

        abstract = " ".join(parts).strip()
        if abstract:
            yield {"pmid": pmid, "title": title, "abstract": abstract}


def fetch_abstracts(pmids: list[str]) -> Iterator[dict]:
    resp = _request("efetch.fcgi", {
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
    })
    try:
        yield from parse_articles(resp.text)
    except ET.ParseError as exc:
        print(f"    [warn] XML parse failed for a batch of {len(pmids)}: {exc}")


def fetch_year(year: int, shard: Path, cap: int | None = PER_YEAR_CAP) -> int:
    windows = windows_for_year(year)

    # Split the year's cap evenly across its windows rather than taking the cap
    # from the first window, which would silently make every year a January
    # sample once month subdivision kicks in.
    per_window = None
    if cap:
        per_window = -(-cap // len(windows))     # ceiling division

    pmids: list[str] = []
    for lo, hi in windows:
        pmids.extend(get_pmids(lo, hi, limit=per_window))

    pmids = list(dict.fromkeys(pmids))          # de-dup, preserve order
    if cap:
        pmids = pmids[:cap]
    written = 0
    tmp = shard.with_suffix(".partial")

    with tmp.open("w", encoding="utf-8") as fh:
        for i in range(0, len(pmids), EFETCH_BATCH):
            for rec in fetch_abstracts(pmids[i:i + EFETCH_BATCH]):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    tmp.replace(shard)                          # atomic: a shard exists only if complete
    kept = 100.0 * written / len(pmids) if pmids else 0.0
    print(f"  {year}: {len(pmids):,} pmids -> {written:,} with abstracts ({kept:.0f}%)")
    return written


def merge_shards(out_path: Path) -> tuple[int, int]:
    """Concatenate shards, dropping PMIDs seen in an earlier year."""
    seen: set[str] = set()
    total = duplicates = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for shard in sorted(SHARD_DIR.glob("*.jsonl")):
            for line in shard.open(encoding="utf-8"):
                rec = json.loads(line)
                if rec["pmid"] in seen:
                    duplicates += 1
                    continue
                seen.add(rec["pmid"])
                out.write(line)
                total += 1
    return total, duplicates


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-year", type=int, default=MIN_YEAR)
    ap.add_argument("--max-year", type=int, default=MAX_YEAR)
    ap.add_argument("--cap", type=int, default=PER_YEAR_CAP,
                    help="max abstracts per year (0 = uncapped)")
    ap.add_argument("--force", action="store_true",
                    help="refetch years whose shard already exists")
    args = ap.parse_args(argv)

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print(f"query      : {QUERY}")
    print(f"date range : {args.min_year}/01/01 - {args.max_year}/12/31")
    print(f"per-year   : {args.cap:,} abstracts" if args.cap else "per-year   : uncapped")
    print(f"api key    : {'yes (10 req/s)' if api_key() else 'no (3 req/s) - see NCBI_API_KEY'}")
    print(f"shards     : {SHARD_DIR.relative_to(REPO_ROOT)}")
    print()

    total_hits = count_hits(f"{args.min_year}/01/01", f"{args.max_year}/12/31")
    print(f"total hits in range: {total_hits:,}")
    print()

    for year in range(args.min_year, args.max_year + 1):
        shard = SHARD_DIR / f"{year}.jsonl"
        if shard.exists() and not args.force:
            print(f"  {year}: shard exists, skipping (--force to refetch)")
            continue
        try:
            fetch_year(year, shard, cap=args.cap or None)
        except FetchError as exc:
            print(f"  {year}: FAILED - {exc}")
            print("         rerun the script; completed years are skipped")
            return 1

    total, duplicates = merge_shards(args.out)
    elapsed = time.time() - started

    print()
    print(f"[ok] {total:,} unique abstracts -> {args.out.relative_to(REPO_ROOT)}")
    print(f"     {duplicates:,} cross-year duplicate PMIDs dropped")
    print(f"     wall clock: {elapsed / 60:.1f} min")
    print()
    print("Record in report/data_documentation.md: query, date range, fetch date,")
    print(f"record count ({total:,}), and the wall-clock time above (PLAN step 1.5).")

    if total < 100_000:
        print()
        print(f"[!!] {total:,} is below the 100k floor in PRD section 6.2.")
        print("     Widen --min-year, or add the subheading clauses from the PRD's")
        print("     query table (month windows will kick in automatically).")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
