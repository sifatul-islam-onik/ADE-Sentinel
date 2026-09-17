"""Phase 7 gate: the demo's models, its measured accuracy and its cold start.

    python scripts/check_demo.py            # the default BiLSTM pair
    python scripts/check_demo.py --bert     # also the BiomedBERT pair (several minutes on CPU)

Three things have to hold before the demo is shown to anyone:

1. **Each gate reproduces its logged decisions.** The demo runs Stage 1 live, on
   CPU, through `src.demo_pipeline`; the report's numbers come from the saved test
   predictions. This re-decodes the test split and compares decision for
   decision. Stage 2 is already checked by `scripts/check_stage2_inference.py`.
2. **The pair it ships is measured.** Runs 12 and 12b both use the BiomedBERT
   gate. The default demo pair - run 6's gate and run 10's tagger - is scored here
   as run 12c, with the same scorer and metric names as `pipeline_eval.py`, so the
   demo quotes its own accuracy rather than the best model's.
3. **Cold start is under five seconds** (PRD Phase 7 exit), measured the way a
   person meets it: a headless Chromium-based browser is already open, `streamlit
   run` is launched, and the clock stops when the first verdict is on the page.
   Nothing cheaper measures this. The server's health check comes before the page
   and the models have even started loading, and `streamlit.testing` skips the
   browser entirely. Launch time varies from run to run, so this repeats the
   launch and passes on the median. Every launch is recorded, including any over
   the limit.

It also runs every preloaded example through each checked pair.

Writes results/demo_check.json. Appends run 12c to results/runs.csv unless the
latest 12c row already has identical metrics (skip with --no-log). Exits non-zero
if any check fails. With no browser found, cold start is reported as not measured
rather than passed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.demo_pipeline import EXAMPLES, PAIRS  # noqa: E402
from src.utils import RUNS_CSV  # noqa: E402

APP = REPO_ROOT / "app" / "streamlit_app.py"
OUT = REPO_ROOT / "results" / "demo_check.json"
COLD_START_LIMIT = 5.0
RUN_12C = ("12c", "6", "10", "the all-BiLSTM pair the demo loads by default (PLAN 7.2)")
BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bert", action="store_true",
                   help="also check the BiomedBERT pair (decodes 3,133 sentences on CPU)")
    p.add_argument("--bootstrap", type=int, default=2000, help="resamples for run 12c's intervals")
    p.add_argument("--launches", type=int, default=6, help="cold starts to time")
    p.add_argument("--browser", help="Chromium-based browser executable (default: search)")
    p.add_argument("--no-log", action="store_true", help="do not append run 12c to runs.csv")
    return p.parse_args(argv)


def reproduce_gate(run_id: str, texts, labels) -> dict:
    """Live CPU decisions against the saved test predictions of a Stage 1 run."""
    import numpy as np

    import pipeline_eval as pe
    from src.demo_pipeline import load_gate

    t0 = time.perf_counter()
    gate = load_gate(run_id)
    load = time.perf_counter() - t0
    t0 = time.perf_counter()
    decisions = gate.classify(texts)
    decode = time.perf_counter() - t0

    saved = pe.stage1_decisions()[run_id]
    score = np.load(next((REPO_ROOT / "models" / "stage1").glob(f"run{run_id}_*"))
                    / "test_predictions.npz")["score"]
    live = np.array([int(d) for _, d in decisions])
    p_ade = np.array([p for p, _ in decisions])
    result = {
        "sentences": len(texts),
        "identical_decisions": int((live == saved["y_pred"]).sum()),
        "max_abs_score_difference": float(np.abs(p_ade - score).max()),
        "load_seconds": round(load, 2), "decode_seconds": round(decode, 2),
    }
    if list(saved["y_true"]) != labels:
        raise SystemExit(f"run {run_id}'s saved y_true is not aligned with stage1_test.parquet")
    return result


def score_12c(texts, labels, words, gold, reps: int):
    import pipeline_eval as pe

    run_id, s1, s2, why = RUN_12C
    decisions = list(pe.stage1_decisions()[s1]["y_pred"])
    decode = pe.stage2_decode(s2, texts, labels, words, refresh=False)
    scored = pe.score_pipeline(gold, labels, decisions, decode["pred"], reps)
    scored.update({"run_id": run_id, "s1": s1, "s2": s2, "why": why})
    return scored, pe.run_metrics(scored, decode["decode_seconds"])


def latest_metrics(run_id: str):
    import csv

    if not RUNS_CSV.exists():
        return None
    with RUNS_CSV.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["run_id"] == run_id]
    return json.loads(rows[-1]["metrics_json"]) if rows else None


def run_examples(key: str) -> list[dict]:
    from src.demo_pipeline import load_pipeline

    pipeline = load_pipeline(key)
    out = []
    for example in EXAMPLES:
        (result,) = pipeline.analyse(example.text)
        out.append({
            "test_index": example.test_index, "corpus_ade": example.ade,
            "is_ade": result.is_ade, "confidence": round(result.confidence, 4),
            "entities": [f"{e.label} {e.text}" for e in result.entities],
        })
    return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(what: str, probe, timeout: float = 90.0):
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            value = probe()
            if value:
                return value
        except OSError:                      # not listening yet
            pass
        time.sleep(0.05)
    raise SystemExit(f"timed out after {timeout:.0f} s waiting for {what}")


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=2) as response:
        return json.loads(response.read())


def responds(url: str) -> bool:
    # A short timeout on purpose. On Windows a connect to a local port nobody listens on
    # is not refused; it times out, so a 1 s timeout would sample readiness once a second
    # and add up to a second to every launch timed. A listening local port accepts in
    # well under a millisecond.
    with urllib.request.urlopen(url, timeout=0.1) as response:
        return response.status == 200


def stop(process) -> None:
    """Stop a process and everything it spawned - browsers run helper processes."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


class DevTools:
    """The two Chrome DevTools Protocol calls the timing needs, over one page's socket."""

    def __init__(self, url: str):
        from websockets.sync.client import connect   # installed with streamlit

        self.socket = connect(url, proxy=None, max_size=None)
        self.last_id = 0

    def call(self, method: str, **params) -> dict:
        self.last_id += 1
        self.socket.send(json.dumps({"id": self.last_id, "method": method, "params": params}))
        while True:                          # skip events until our reply arrives
            message = json.loads(self.socket.recv(timeout=30))
            if message.get("id") == self.last_id:
                return message.get("result", {})

    def evaluate(self, expression: str):
        reply = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        return reply.get("result", {}).get("value")


def find_browser(explicit: str | None) -> str | None:
    for candidate in [explicit] if explicit else BROWSERS:
        if Path(candidate).is_file():
            return candidate
        if shutil.which(candidate):
            return shutil.which(candidate)
    return None


def time_cold_starts(browser: str, launches: int) -> dict:
    """Seconds from launching `streamlit run` to the first verdict in an open browser."""
    verdict = "!!document.body && document.body.innerText.includes('gate confidence')"
    seconds, server, tools = [], None, None
    cdp_port = free_port()
    with tempfile.TemporaryDirectory(prefix="demo-browser-", ignore_cleanup_errors=True) as profile:
        browser_process = subprocess.Popen(
            [browser, "--headless=new", f"--remote-debugging-port={cdp_port}",
             f"--user-data-dir={profile}", "--no-first-run", "--window-size=1400,1100",
             "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            targets = wait_for("the browser's DevTools endpoint",
                               lambda: get_json(f"http://127.0.0.1:{cdp_port}/json/list"))
            tools = DevTools(next(t for t in targets if t["type"] == "page")["webSocketDebuggerUrl"])
            time.sleep(3)                    # the browser's own start-up is not the demo's

            for _ in range(launches):
                tools.call("Page.navigate", url="about:blank")
                port = free_port()
                started = time.perf_counter()
                server = subprocess.Popen(
                    [sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "true",
                     "--server.port", str(port), "--browser.gatherUsageStats", "false"],
                    cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    env={**os.environ, "PYTHONUTF8": "1"})
                wait_for("the Streamlit health check",
                         lambda: responds(f"http://127.0.0.1:{port}/_stcore/health"))
                tools.call("Page.navigate", url=f"http://localhost:{port}/")
                wait_for("the first verdict on the page", lambda: tools.evaluate(verdict))
                seconds.append(round(time.perf_counter() - started, 2))
                stop(server)
                server = None
        finally:
            if tools:
                tools.socket.close()
            if server:
                stop(server)
            stop(browser_process)

    return {"browser": Path(browser).name, "launch_seconds": seconds,
            "median_seconds": round(statistics.median(seconds), 2), "max_seconds": max(seconds)}


def main(argv=None) -> int:
    args = parse_args(argv)

    checks, failures = {}, []

    # Timed first, on an idle machine. The checks below hold the CPU at full load for
    # minutes (--bert: about five), and a laptop that has just done that runs a short
    # burst of work - which is what a cold start is - far slower than a rested one.
    browser = find_browser(args.browser)
    if browser is None:
        checks["cold_start"] = {"measured": False,
                                "reason": "no Chromium-based browser found - pass --browser"}
        print("cold start: NOT MEASURED - no Chromium-based browser found (pass --browser)")
    else:
        cold = time_cold_starts(browser, args.launches)
        over = sum(s > COLD_START_LIMIT for s in cold["launch_seconds"])
        checks["cold_start"] = {"measured": True, **cold, "limit_seconds": COLD_START_LIMIT,
                                "launches_over_limit": over}
        print(f"cold start, launch to first verdict in {cold['browser']}: median "
              f"{cold['median_seconds']} s, max {cold['max_seconds']} s; {over} of "
              f"{len(cold['launch_seconds'])} launches over {COLD_START_LIMIT:.0f} s "
              f"({', '.join(map(str, cold['launch_seconds']))})")
        if cold["median_seconds"] > COLD_START_LIMIT:
            failures.append(f"median cold start {cold['median_seconds']} s exceeds "
                            f"{COLD_START_LIMIT:.0f} s")

    import pipeline_eval as pe

    texts, labels, words, gold = pe.load_test()
    checks["pairs"] = {}

    for key, pair in PAIRS.items():
        if key != "bilstm" and not args.bert:
            continue
        gate = reproduce_gate(pair.gate_run, texts, labels)
        exact = gate["identical_decisions"] == gate["sentences"]
        print(f"{pair.label}: gate run {pair.gate_run} reproduces {gate['identical_decisions']}/"
              f"{gate['sentences']} saved decisions (max |P(ADE) difference| "
              f"{gate['max_abs_score_difference']:.2e}) | load {gate['load_seconds']} s, "
              f"decode {gate['decode_seconds']} s")
        if not exact:
            failures.append(f"gate run {pair.gate_run} differs from its saved decisions")

        examples = run_examples(key)
        for e in examples:
            print(f"  example {e['test_index']:>4}: corpus {'ADE' if e['corpus_ade'] else 'not ADE'}"
                  f" | {'ADE' if e['is_ade'] else 'not ADE'} ({e['confidence']:.0%}) "
                  f"{'; '.join(e['entities'])}")
        checks["pairs"][key] = {"gate": gate, "examples": examples}

    scored, metrics = score_12c(texts, labels, words, gold, args.bootstrap)
    sp = scored["scores"]["pipeline"]
    print(f"run 12c (run 6 + run 10): oracle {scored['scores']['oracle']['entity_f1_strict']:.4f}"
          f" -> pipeline {sp['entity_f1_strict']:.4f} strict entity-F1")
    checks["run_12c"] = metrics

    import torch

    checks.update({"torch": torch.__version__, "failures": failures})
    OUT.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")

    if not args.no_log:
        roundtrip = json.loads(json.dumps(metrics, sort_keys=True, default=str))
        if latest_metrics("12c") == roundtrip:
            print("run 12c: latest runs.csv row already has these metrics - not appended")
        else:
            import transformers

            from src.utils import log_run

            log_run(run_id="12c", stage="1+2", model="run6+run10", metrics=metrics,
                    params=pe.run_params(scored, args.bootstrap, torch.__version__,
                                         transformers.__version__),
                    seed=pe.SEED, device_count=0,
                    notes=f"Step 7.2 run 12c; {RUN_12C[3]}; inference only.")
            print("appended run 12c to results/runs.csv")

    for failure in failures:
        print(f"FAILED: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
