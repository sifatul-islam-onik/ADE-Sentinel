"""The ADE-Sentinel demo: does this sentence report a medicine causing harm?

    .venv\\Scripts\\python -m streamlit run app\\streamlit_app.py

This file is layout only. Every model decision is made in `src/pipeline.py`.

The styling is deliberately large and high-contrast: the demo is read on a
projector and by people who find small, low-contrast text hard going. The theme
is pinned to light in `.streamlit/config.toml` so it does not follow the
viewer's system dark mode.
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

from src.pipeline import EXAMPLES, load_pipeline, logged_scores  # noqa: E402

DISCLAIMER = ("This is a research demonstration on published medical papers. "
              "It is not a medical tool and must not be used for any health decision.")
MAX_CHARS = 5000

# Chosen for contrast on white: both pass WCAG AA for normal text.
DRUG = "#1d4ed8"        # blue
EFFECT = "#c2410c"      # orange-red

st.set_page_config(page_title="ADE-Sentinel", page_icon="💊", layout="centered")

# ---- typography -----------------------------------------------------------------------------
st.markdown(f"""
<style>
  html, body, [class*="css"] {{ font-size: 18px; }}
  .block-container {{ padding-top: 2.5rem; max-width: 50rem; }}

  h1 {{ font-size: 2.4rem !important; font-weight: 800 !important; color: #0f172a; }}
  h2, h3 {{ color: #0f172a; }}

  p, li, label, .stMarkdown {{ font-size: 1.05rem; line-height: 1.7; color: #1f2937; }}

  /* The text box people type into */
  .stTextArea textarea {{
      font-size: 1.15rem !important; line-height: 1.6 !important;
      color: #111827 !important; background: #ffffff !important;
      border: 2px solid #cbd5e1 !important; border-radius: 0.6rem !important;
  }}
  .stTextArea textarea:focus {{ border-color: {DRUG} !important; }}

  /* Example buttons - bigger tap targets */
  button[kind="pills"], button[data-testid="stBaseButton-pills"] {{
      font-size: 1rem !important; padding: 0.5rem 1rem !important;
  }}

  section[data-testid="stSidebar"] {{ background: #f8fafc; }}
  section[data-testid="stSidebar"] * {{ font-size: 1rem; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Starting up, one moment...")
def get_pipeline():
    return load_pipeline()


def mark(colour: str, text: str) -> str:
    """One highlighted phrase inside the sentence."""
    return (f'<mark style="background: {colour}22; '
            f'border-bottom: 3px solid {colour}; color: #111827; '
            f'padding: 0.1em 0.15em; border-radius: 0.2em;">{text}</mark>')


def highlight(result) -> str:
    """The sentence as HTML with its findings marked. All input text is escaped."""
    parts, cursor = [], 0
    for entity in result.entities:      # strict entities: ordered, never overlapping
        parts.append(html.escape(result.text[cursor:entity.start]))
        colour = DRUG if entity.label == "DRUG" else EFFECT
        parts.append(mark(colour, html.escape(entity.text)))
        cursor = entity.end
    parts.append(html.escape(result.text[cursor:]))
    return "".join(parts)


def findings(result, label: str) -> list[str]:
    return [e.text for e in result.entities if e.label == label]


def pick_example():
    index = st.session_state.example
    if index is not None:
        st.session_state.text = EXAMPLES[index].text


if "text" not in st.session_state:
    st.session_state.text = EXAMPLES[0].text
    st.session_state.example = 0

# ---- sidebar --------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### How well does it work?")
    st.markdown("Measured on **3,133 sentences it had never seen** during training.")

    scores = logged_scores()

    def show(label: str, value, note: str) -> None:
        shown = f"{value:.2f}" if value is not None else "—"
        st.markdown(
            f'<div style="margin: 0.9rem 0 1.2rem 0;">'
            f'<div style="font-size:0.95rem; color:#475569;">{label}</div>'
            f'<div style="font-size:2rem; font-weight:800; color:#0f172a; '
            f'line-height:1.2;">{shown}</div>'
            f'<div style="font-size:0.85rem; color:#64748b;">{note}</div></div>',
            unsafe_allow_html=True)

    show("Spotting which sentences report a side effect", scores["stage1"],
         "out of a best possible 1.00")
    show("Finding the exact words", scores["stage2"],
         "when handed sentences that really do report one")
    show("Both jobs, end to end", scores["pipeline"],
         "when it has to decide for itself first")

    st.markdown(
        '<div style="font-size:0.85rem; color:#64748b; line-height:1.5;">'
        'These are <b>F1 scores</b>, not "percent correct" — they balance how often '
        'it finds the right answer against how often it raises a false alarm. '
        'Read from <code>results/runs.csv</code>.</div>',
        unsafe_allow_html=True)

# ---- main -----------------------------------------------------------------------------------
st.title("Does this sentence report a side effect?")
st.markdown(
    "Paste a sentence from a medical report below. This tool reads it and answers "
    "one question: **does it say a medicine caused something harmful?** "
    "If it does, the tool also highlights which medicine and which harm.")

st.info(DISCLAIMER, icon="⚠️")

st.markdown("**Try one of these real examples, or type your own:**")
st.pills("Examples", options=range(len(EXAMPLES)),
         format_func=lambda i: EXAMPLES[i].button, key="example",
         on_change=pick_example, label_visibility="collapsed")

text = st.text_area("Your sentence", key="text", height=140, max_chars=MAX_CHARS,
                    help="One or more sentences. Each one is judged on its own.")

chosen = st.session_state.example
if chosen is not None and text.strip() == EXAMPLES[chosen].text:
    example = EXAMPLES[chosen]
    answer = "a side effect" if example.ade else "no side effect"
    st.caption(f"This is a real sentence from a medical paper. The correct answer is "
               f"**{answer}**. {example.note}")

if not text.strip():
    st.warning("Type or paste a sentence above to get an answer.")
    st.stop()

pipeline = get_pipeline()
started = time.perf_counter()
results = pipeline.analyse(text)
elapsed_ms = (time.perf_counter() - started) * 1000

st.markdown("---")

for result in results:
    with st.container(border=True):
        if result.is_ade:
            st.markdown(
                f'<div style="font-size:1.35rem; font-weight:800; color:#b91c1c; '
                f'margin-bottom:0.2rem;">Yes — this reports a side effect</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="font-size:1.35rem; font-weight:800; color:#334155; '
                f'margin-bottom:0.2rem;">No side effect reported here</div>',
                unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:0.95rem; color:#64748b; margin-bottom:1rem;">'
            f'The tool is <b>{result.confidence:.0%} sure</b> of this answer.</div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:1.2rem; line-height:2.1; color:#111827; '
            f'margin-bottom:0.6rem;">{highlight(result)}</div>',
            unsafe_allow_html=True)

        drugs, effects = findings(result, "DRUG"), findings(result, "EFFECT")
        if drugs or effects:
            rows = ""
            if drugs:
                rows += (f'<div style="margin-top:0.5rem;"><span style="color:{DRUG}; '
                         f'font-weight:700;">Medicine:</span> '
                         f'{html.escape(", ".join(drugs))}</div>')
            if effects:
                rows += (f'<div style="margin-top:0.3rem;"><span style="color:{EFFECT}; '
                         f'font-weight:700;">Harm it caused:</span> '
                         f'{html.escape(", ".join(effects))}</div>')
            st.markdown(f'<div style="font-size:1.05rem; border-top:1px solid #e2e8f0; '
                        f'padding-top:0.7rem;">{rows}</div>', unsafe_allow_html=True)

        elif result.is_ade:
            st.caption("The tool thinks this reports a side effect, but could not pin down "
                       "which words name the medicine and the harm.")
        else:
            st.caption("Nothing is highlighted because the tool decided there is no side "
                       "effect to find here.")

        if result.truncated:
            st.caption("This sentence is longer than the tool can read, so its ending "
                       "was ignored.")

count = f"{len(results)} sentence{'s' if len(results) != 1 else ''}"
st.markdown(
    f'<div style="font-size:0.9rem; color:#64748b; margin-top:1rem;">'
    f'{count} read in {elapsed_ms:.0f} milliseconds. '
    f'<span style="color:{DRUG}; font-weight:700;">Blue</span> marks a medicine, '
    f'<span style="color:{EFFECT}; font-weight:700;">orange</span> marks the harm '
    f'it caused.</div>',
    unsafe_allow_html=True)
