"""ADE-Sentinel demo (Phase 7): verdict, confidence and highlighted spans for any text.

    .venv\\Scripts\\python -m streamlit run app\\streamlit_app.py

Layout only - every model decision is made in `src/demo_pipeline.py`, which
`scripts/check_demo.py` checks against the logged runs and times.
"""

from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.demo_pipeline import (  # noqa: E402
    DEFAULT_PAIR, EXAMPLES, PAIRS, load_pipeline, logged_scores,
)

DISCLAIMER = "Research demonstration on published literature; not a clinical or diagnostic tool."
MAX_CHARS = 5000
# RGB triples, drawn translucent so the marks read on light and dark themes alike.
COLOURS = {"DRUG": "37, 99, 235", "EFFECT": "217, 70, 0"}

st.set_page_config(page_title="ADE-Sentinel", page_icon="💊", layout="centered")


@st.cache_resource(show_spinner="Loading models...")
def pipeline_for(key: str):
    return load_pipeline(key)


def chip(label: str, text: str) -> str:
    rgb = COLOURS.get(label, "110, 110, 110")
    return (f'<mark style="background: rgba({rgb}, 0.16); border-bottom: 2px solid rgb({rgb}); '
            f'color: inherit; padding: 0.05em 0.2em; border-radius: 0.25em;">{text}'
            f'<sup style="font-size: 0.6em; font-weight: 700; letter-spacing: 0.03em; '
            f'margin-left: 0.3em; color: rgb({rgb});">{label}</sup></mark>')


def highlight(result) -> str:
    """The sentence as HTML with its entities marked. All input text is escaped."""
    parts, cursor = [], 0
    for entity in result.entities:          # strict entities: ordered, never overlapping
        parts.append(html.escape(result.text[cursor:entity.start]))
        parts.append(chip(entity.label, html.escape(entity.text)))
        cursor = entity.end
    parts.append(html.escape(result.text[cursor:]))
    return "".join(parts)


def score(value) -> str:
    return f"{value:.3f}" if value is not None else "not logged"


def pick_example():
    index = st.session_state.example
    if index is not None:
        st.session_state.text = EXAMPLES[index].text


if "text" not in st.session_state:
    st.session_state.text = EXAMPLES[0].text
    st.session_state.example = 0

# ---- sidebar: which models, and how good they measured ------------------------------------
with st.sidebar:
    st.header("Models")
    key = st.radio("Model pair", list(PAIRS), index=list(PAIRS).index(DEFAULT_PAIR),
                   format_func=lambda k: PAIRS[k].label, label_visibility="collapsed")
    pair = PAIRS[key]
    st.caption(pair.description)

    scores = logged_scores(pair)
    st.markdown(
        "| Measured on the test split | Score |\n|---|---|\n"
        f"| Stage 1 gate, run {pair.gate_run} - macro-F1 | {score(scores['stage1'])} |\n"
        f"| Stage 2 tagger, run {pair.tagger_run} - strict entity-F1 on ADE sentences "
        f"| {score(scores['stage2'])} |\n"
        f"| Both chained, run {pair.pipeline_run} - strict entity-F1 on all sentences "
        f"| {score(scores['pipeline'])} |")
    st.caption("Scores are read from `results/runs.csv`. Confidence is the gate's score for its "
               "verdict, not a calibrated probability.")

# ---- main ----------------------------------------------------------------------------------
st.title("ADE-Sentinel")
st.markdown("Finds sentences that report an **adverse drug event**, then marks the "
            "**drug** and the **effect** in each one.")
st.warning(DISCLAIMER, icon="⚠️")

st.pills("Try an example - all five are held-out test sentences", options=range(len(EXAMPLES)),
         format_func=lambda i: EXAMPLES[i].button, key="example", on_change=pick_example)
text = st.text_area("Text", key="text", height=130, max_chars=MAX_CHARS,
                    help="One or more sentences. Each sentence is judged on its own.")

chosen = st.session_state.example
if chosen is not None and text.strip() == EXAMPLES[chosen].text:
    example = EXAMPLES[chosen]
    st.caption(f"Test sentence {example.test_index} - corpus label "
               f"**{'ADE' if example.ade else 'not ADE'}**. {example.note}")

if not text.strip():
    st.info("Enter a sentence to analyse.")
    st.stop()

pipeline = pipeline_for(pair.key)
started = time.perf_counter()
results = pipeline.analyse(text)
elapsed_ms = (time.perf_counter() - started) * 1000

for result in results:
    with st.container(border=True):
        badge = ":red-badge[ADE]" if result.is_ade else ":gray-badge[not ADE]"
        st.markdown(f"{badge} &nbsp; gate confidence {result.confidence:.0%}")
        st.markdown(f'<div style="font-size: 1.05rem; line-height: 2;">{highlight(result)}</div>',
                    unsafe_allow_html=True)
        if not result.is_ade:
            st.caption("Stage 2 did not run: the gate judged that this sentence does not report "
                       "an ADE.")
        elif not result.entities:
            st.caption("The gate accepted this sentence, but Stage 2 marked no drug or effect.")
        if result.truncated:
            st.caption("This sentence is longer than a model's input limit, so its end went unread.")

count = f"{len(results)} sentence{'s' if len(results) != 1 else ''}"
st.markdown(f'<div style="font-size: 0.85rem; opacity: 0.75;">{chip("DRUG", "drug")} &nbsp; '
            f'{chip("EFFECT", "effect")} &nbsp; - {count} analysed in {elapsed_ms:.0f} ms '
            f'with {html.escape(pair.label)}</div>', unsafe_allow_html=True)
