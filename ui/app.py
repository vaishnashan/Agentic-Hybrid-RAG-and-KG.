"""
Streamlit UI for the agent — two tabs: Ask (the actual demo) and About the Dataset
(what the corpus is, where it came from).

Calls the FastAPI /ask endpoint over HTTP rather than importing utils.agent4
directly — keeps the UI as a genuine separate service.

Run locally:
    streamlit run utils/ui/app.py

Set API_BASE_URL and API_KEY as env vars (or edit the defaults below) to point
at your running FastAPI instance.

Design notes: palette is white / ash-gray / near-black / a single "shiny" blue
accent, per brief. The name "Prism" and the blue-light motif are a deliberate
match to what the system actually does — a question gets split into retrieval
paths (dense/sparse, sub-questions) and recomposed into one answer, same as a
prism splitting and recombining light. Swap PROJECT_TITLE below if you'd rather
use a different name — nothing else depends on it.

Confidence/retries are NOT shown — since self_critic.py was removed from the
pipeline, FinalAnswer.confidence is hardcoded to 1.0 and .retries is always 0,
so displaying them would be showing fake data. The strategy actually used
(vector-only vs hybrid) is real per-answer data and is shown instead.
"""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")

PROJECT_TITLE = "Prism"
PROJECT_TAGLINE = (
    "A research agent that splits every question across hybrid retrieval "
    "(dense + sparse) and a knowledge graph, then recomposes the pieces into "
    "one grounded answer."
)

CORPUS_REPO_URL = "https://github.com/masamasa59/ai-agent-papers"

st.set_page_config(page_title=f"{PROJECT_TITLE} — Agentic RAG + Knowledge Graph", layout="wide")

# ---------------------------------------------------------------------------
# Styling — fonts + palette (white / ash / black / blue), injected once.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --c-white: #FFFFFF;
            --c-bg: #FAFAFB;
            --c-ash: #E4E4E8;
            --c-ash-dark: #6B6B72;
            --c-black: #0B0B0D;
            --c-blue: #2563EB;
            --c-blue-light: #5B9BFF;
            --c-blue-glow: rgba(37, 99, 235, 0.35);
        }

        .stApp { background-color: var(--c-bg); }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .prism-hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 2.6rem;
            letter-spacing: -0.02em;
            color: var(--c-black);
            margin-bottom: 0.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .prism-hero-title .glint {
            background: linear-gradient(120deg, var(--c-blue) 0%, var(--c-blue-light) 45%, var(--c-blue) 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .prism-hero-tagline {
            font-family: 'Inter', sans-serif;
            color: var(--c-ash-dark);
            font-size: 1.02rem;
            max-width: 680px;
            line-height: 1.5;
            margin-bottom: 1.1rem;
        }

        .pill-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.6rem; }
        .pill {
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            font-weight: 500;
            padding: 0.32rem 0.85rem;
            border-radius: 999px;
            background: var(--c-white);
            color: var(--c-ash-dark);
            border: 1px solid var(--c-ash);
        }
        .pill.blue {
            background: linear-gradient(135deg, var(--c-blue), var(--c-blue-light));
            color: var(--c-white);
            border: none;
            box-shadow: 0 2px 10px var(--c-blue-glow);
        }

        .answer-card {
            background: var(--c-white);
            border: 1px solid var(--c-ash);
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 1px 3px rgba(11, 11, 13, 0.04);
            margin-top: 0.6rem;
        }
        .answer-card h4 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.95rem;
            color: var(--c-ash-dark);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.6rem;
        }
        .answer-text {
            font-family: 'Inter', sans-serif;
            font-size: 1.02rem;
            color: var(--c-black);
            line-height: 1.6;
        }

        .strategy-badge {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            padding: 0.3rem 0.7rem;
            border-radius: 8px;
            margin-top: 0.9rem;
        }
        .strategy-badge.hybrid {
            background: linear-gradient(135deg, var(--c-blue), var(--c-blue-light));
            color: var(--c-white);
            box-shadow: 0 2px 10px var(--c-blue-glow);
        }
        .strategy-badge.vector {
            background: var(--c-ash);
            color: var(--c-black);
        }

        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, var(--c-blue) 0%, var(--c-blue-light) 100%);
            border: none;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            box-shadow: 0 4px 14px var(--c-blue-glow);
            transition: box-shadow 0.2s ease, transform 0.15s ease;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            box-shadow: 0 6px 20px var(--c-blue-glow);
            transform: translateY(-1px);
        }

        code, .stCodeBlock, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="prism-hero-title">{PROJECT_TITLE}<span class="glint"> · Agentic RAG</span></div>
    <div class="prism-hero-tagline">{PROJECT_TAGLINE}</div>
    <div class="pill-row">
        <div class="pill blue">Hybrid Retrieval</div>
        <div class="pill blue">Knowledge Graph</div>
        <div class="pill blue">RAG</div>
        <div class="pill">Single-hop &amp; Multi-hop</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_ask, tab_about = st.tabs(["Ask", "About the Dataset"])

# ---------------------------------------------------------------------------
# Tab 1 — Ask
# ---------------------------------------------------------------------------
with tab_ask:
    question = st.text_input(
        "Ask a question about the paper corpus:",
        placeholder="e.g. What is SkillOpt? · or · Compare Gorilla and MRKL Systems.",
    )
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and question.strip():
        with st.spinner("Routing → retrieving → reasoning..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={"question": question},
                    headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
                    timeout=60,
                )
                if response.status_code == 429:
                    st.error("Rate limit hit — please wait a moment and try again.")
                elif response.status_code == 401:
                    st.error("Auth failed — check API_KEY matches the API's configured key.")
                else:
                    response.raise_for_status()
                    result = response.json()

                    strategy = result.get("strategy_used", "")
                    is_hybrid = strategy == "hybrid_both"
                    badge_class = "hybrid" if is_hybrid else "vector"
                    badge_label = "Hybrid (dense + sparse)" if is_hybrid else "Vector-only (dense)"

                    sources = result.get("sources", [])

                    st.markdown(
                        f"""
                        <div class="answer-card">
                            <h4>Answer</h4>
                            <div class="answer-text">{result['answer']}</div>
                            <span class="strategy-badge {badge_class}">{badge_label}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if sources:
                        with st.expander(f"Sources ({len(sources)} chunks)"):
                            for source_id in sources:
                                st.code(source_id)

            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
    elif ask_clicked:
        st.warning("Type a question first.")

# ---------------------------------------------------------------------------
# Tab 2 — About the Dataset
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
        <div class="answer-card">
            <h4>Corpus</h4>
            <div class="answer-text">
                This agent answers questions over 30 full-text research papers on AI
                agent capabilities, sourced from
                <a href="{CORPUS_REPO_URL}" target="_blank">masamasa59/ai-agent-papers</a>
                on GitHub — full PDFs (23–60 pages each), not just abstracts, so
                retrieval works at the section level and can reason across papers,
                not just within one.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="answer-card">
                <h4>Tool-use</h4>
                <div class="answer-text">19 papers on how agents discover, call, and
                chain external tools.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="answer-card">
                <h4>Knowledge</h4>
                <div class="answer-text">11 papers on how agents access and reason
                over external or internal knowledge.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        """
        <div class="answer-card">
            <h4>How to use this</h4>
            <div class="answer-text">
                Ask a direct lookup question (<em>single-hop</em>) — e.g. "What is
                MRKL Systems?" — and the agent searches the paper corpus directly.
                Ask a question that connects two or more things (<em>multi-hop</em>)
                — e.g. "Compare Gorilla and MRKL Systems" — and the agent breaks it
                into sub-questions, retrieves for each separately, and composes one
                combined answer. Every answer also checks a knowledge graph of
                papers and concepts alongside retrieval.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )