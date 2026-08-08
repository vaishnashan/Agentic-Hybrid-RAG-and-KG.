"""
Streamlit UI for the agent — two tabs: Ask (the actual demo) and About the
Knowledge Base (what the corpus is, how NOVA works, where it came from).

Calls the FastAPI /ask endpoint over HTTP rather than importing utils.agent4
directly — keeps the UI as a genuine separate service.

Run locally:
    streamlit run utils/ui/app.py

Set API_BASE_URL and API_KEY as env vars (or edit the defaults below) to point
at your running FastAPI instance.

Design notes (v2): dark ash/charcoal base with a warm copper-brown accent and
a white-blue "shine" on the title, per updated brief. NOVA (the name) renders
in a shining white-to-blue animated gradient at the top; the tagline renders
in a shining copper gradient directly beneath it. Both use a slow animated
sheen (background-position keyframe). A low-opacity SVG "node graph" pattern
(small colored circles connected by thin lines, evoking the Neo4j knowledge
graph) sits behind the hero and the Ask panel so the input area isn't a blank
void. The Ask panel itself is wrapped in an ash card with a small icon badge
and static example chips instead of a bare text box.

Confidence/retries are NOT shown — since self_critic.py was removed from the
pipeline, FinalAnswer.confidence is hardcoded to 1.0 and .retries is always 0,
so displaying them would be showing fake data. The strategy actually used
(vector-only vs hybrid) is real per-answer data and is shown instead. Likewise,
"what the user receives" in the About tab lists only fields the API actually
returns today (answer, strategy, source chunk IDs, source count) — the KG is
described as part of how the pipeline works, not as a field the API returns,
since used_graph_facts isn't currently exposed on FinalAnswer.
"""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")

PROJECT_TITLE = "NOVA"
PROJECT_SUBTITLE = "Node Orchestrated Vector and Knowledge Assistant"
PROJECT_DESCRIPTION = (
    "NOVA is an agentic AI application that combines hybrid retrieval, "
    "Retrieval-Augmented Generation, and a Neo4j knowledge graph to answer "
    "single-hop and multi-hop questions over a curated research-paper corpus. "
    "It retrieves relevant content using semantic vector search and BM25 "
    "keyword search, connects related concepts through the knowledge graph, "
    "and generates grounded answers with supporting sources."
)

CORPUS_REPO_URL = "https://github.com/masamasa59/ai-agent-papers"

st.set_page_config(page_title=f"{PROJECT_TITLE} — Agentic RAG + Knowledge Graph", layout="wide")

# ---------------------------------------------------------------------------
# Styling — fonts + palette (white / ash / black / blue), injected once.
# ---------------------------------------------------------------------------
NODE_GRAPH_SVG = (
    "%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%22260%22%20height%3D%22260%22%20viewBox%3D%220%200%20260%20260%22%3E%0A"
    "%3Cg%20fill%3D%22none%22%20stroke-width%3D%221.2%22%3E%0A"
    "%3Cline%20x1%3D%2230%22%20y1%3D%2240%22%20x2%3D%22110%22%20y2%3D%2290%22%20stroke%3D%22%23B4753E%22%20stroke-opacity%3D%220.35%22/%3E%0A"
    "%3Cline%20x1%3D%22110%22%20y1%3D%2290%22%20x2%3D%22200%22%20y2%3D%2250%22%20stroke%3D%22%236FA8FF%22%20stroke-opacity%3D%220.30%22/%3E%0A"
    "%3Cline%20x1%3D%22110%22%20y1%3D%2290%22%20x2%3D%2290%22%20y2%3D%22180%22%20stroke%3D%22%234FBFA8%22%20stroke-opacity%3D%220.28%22/%3E%0A"
    "%3Cline%20x1%3D%2290%22%20y1%3D%22180%22%20x2%3D%22200%22%20y2%3D%22210%22%20stroke%3D%22%23B4753E%22%20stroke-opacity%3D%220.28%22/%3E%0A"
    "%3Cline%20x1%3D%22200%22%20y1%3D%2250%22%20x2%3D%22230%22%20y2%3D%22140%22%20stroke%3D%22%236FA8FF%22%20stroke-opacity%3D%220.25%22/%3E%0A"
    "%3Cline%20x1%3D%2220%22%20y1%3D%22150%22%20x2%3D%2290%22%20y2%3D%22180%22%20stroke%3D%22%234FBFA8%22%20stroke-opacity%3D%220.22%22/%3E%0A"
    "%3C/g%3E%0A"
    "%3Ccircle%20cx%3D%2230%22%20cy%3D%2240%22%20r%3D%224%22%20fill%3D%22%23B4753E%22%20fill-opacity%3D%220.55%22/%3E%0A"
    "%3Ccircle%20cx%3D%22110%22%20cy%3D%2290%22%20r%3D%225.5%22%20fill%3D%22%236FA8FF%22%20fill-opacity%3D%220.55%22/%3E%0A"
    "%3Ccircle%20cx%3D%22200%22%20cy%3D%2250%22%20r%3D%224%22%20fill%3D%22%234FBFA8%22%20fill-opacity%3D%220.5%22/%3E%0A"
    "%3Ccircle%20cx%3D%2290%22%20cy%3D%22180%22%20r%3D%224.5%22%20fill%3D%22%23B4753E%22%20fill-opacity%3D%220.5%22/%3E%0A"
    "%3Ccircle%20cx%3D%22200%22%20cy%3D%22210%22%20r%3D%224%22%20fill%3D%22%236FA8FF%22%20fill-opacity%3D%220.5%22/%3E%0A"
    "%3Ccircle%20cx%3D%22230%22%20cy%3D%22140%22%20r%3D%223.5%22%20fill%3D%22%234FBFA8%22%20fill-opacity%3D%220.45%22/%3E%0A"
    "%3Ccircle%20cx%3D%2220%22%20cy%3D%22150%22%20r%3D%223%22%20fill%3D%22%236FA8FF%22%20fill-opacity%3D%220.4%22/%3E%0A"
    "%3C/svg%3E"
)

st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --c-white: #F5F3F1;
            --c-bg-1: #17151A;
            --c-bg-2: #241F1C;
            --c-ash: #322C29;
            --c-ash-line: #453E39;
            --c-ash-dark: #B7AFA7;
            --c-ash-dim: #8B837B;
            --c-black: #0B0B0D;
            --c-brown: #B4753E;
            --c-brown-light: #E3A066;
            --c-brown-glow: rgba(180, 117, 62, 0.35);
            --c-blue: #6FA8FF;
            --c-blue-light: #CFE4FF;
            --c-blue-glow: rgba(111, 168, 255, 0.35);
            --c-teal: #4FBFA8;
        }}

        .stApp {{
            background:
                radial-gradient(1100px 550px at 12% -8%, rgba(111,168,255,0.10), transparent 60%),
                radial-gradient(900px 500px at 100% 0%, rgba(180,117,62,0.14), transparent 55%),
                linear-gradient(160deg, var(--c-bg-1) 0%, var(--c-bg-2) 55%, #1B1714 100%);
        }}
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: var(--c-white); }}

        div.block-container {{ padding-top: 2.6rem !important; }}

        @keyframes shine-sweep {{
            0%   {{ background-position: 0% 50%; }}
            100% {{ background-position: 200% 50%; }}
        }}
        @keyframes drift {{
            0%   {{ transform: translateY(0px); }}
            50%  {{ transform: translateY(-6px); }}
            100% {{ transform: translateY(0px); }}
        }}

        /* --- decorative knowledge-graph node pattern, behind the hero --- */
        .node-field {{
            position: absolute;
            top: -20px; right: -30px;
            width: 340px; height: 340px;
            background-image: url("data:image/svg+xml,{NODE_GRAPH_SVG}");
            background-repeat: no-repeat;
            background-size: contain;
            pointer-events: none;
            z-index: 0;
            animation: drift 7s ease-in-out infinite;
        }}
        .node-field.field-2 {{
            top: 140px; right: 420px;
            width: 220px; height: 220px;
            opacity: 0.6;
            animation-delay: 1.5s;
        }}

        .hero-wrap {{ position: relative; }}

        .badge-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.5rem; height: 1.5rem;
            border-radius: 50%;
            margin-right: 0.5rem;
            vertical-align: middle;
            background: linear-gradient(135deg, var(--c-brown), var(--c-blue));
            box-shadow: 0 0 10px var(--c-brown-glow);
        }}

        .nova-title {{
            position: relative;
            z-index: 1;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 3.2rem;
            letter-spacing: -0.02em;
            line-height: 1.25;
            padding-top: 0.15rem;
            margin: 0 0 0.15rem 0;
            background: linear-gradient(110deg, var(--c-white) 15%, var(--c-blue) 45%, var(--c-blue-light) 60%, var(--c-white) 85%);
            background-size: 220% auto;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: shine-sweep 4s linear infinite;
            filter: drop-shadow(0 0 18px rgba(111,168,255,0.25));
        }}

        .nova-subtitle {{
            position: relative;
            z-index: 1;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 1.15rem;
            letter-spacing: -0.005em;
            margin: 0 0 0.6rem 0;
            background: linear-gradient(110deg, var(--c-brown) 20%, var(--c-brown-light) 50%, var(--c-brown) 80%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: shine-sweep 4s linear infinite;
        }}

        .nova-description {{
            position: relative;
            z-index: 1;
            font-family: 'Inter', sans-serif;
            color: var(--c-ash-dark);
            font-size: 0.98rem;
            max-width: 780px;
            line-height: 1.55;
            margin-bottom: 1.3rem;
        }}

        .pill-row {{ position: relative; z-index: 1; display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.4rem; }}
        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            font-weight: 500;
            padding: 0.32rem 0.85rem 0.32rem 0.6rem;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--c-brown), var(--c-brown-light));
            color: #241708;
            border: none;
            box-shadow: 0 2px 10px var(--c-brown-glow);
        }}
        .pill .dot {{
            width: 7px; height: 7px; border-radius: 50%;
            background: var(--c-white);
            box-shadow: 0 0 6px rgba(255,255,255,0.8);
        }}

        .answer-card {{
            position: relative;
            background: linear-gradient(155deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
            border: 1px solid var(--c-ash-line);
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 6px 24px rgba(0,0,0,0.28);
            margin-top: 0.6rem;
            margin-bottom: 0.9rem;
        }}
        .answer-card h4 {{
            display: flex;
            align-items: center;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.95rem;
            color: var(--c-brown-light);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.6rem;
        }}
        .answer-text {{
            font-family: 'Inter', sans-serif;
            font-size: 1.0rem;
            color: var(--c-white);
            line-height: 1.6;
        }}
        .answer-text a {{ color: var(--c-blue); }}
        .answer-text ul {{ margin: 0.3rem 0 0.3rem 1.1rem; padding: 0; }}
        .answer-text li {{ margin-bottom: 0.25rem; }}

        .strategy-badge {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            padding: 0.3rem 0.7rem;
            border-radius: 8px;
            margin-top: 0.9rem;
        }}
        .strategy-badge.hybrid {{
            background: linear-gradient(135deg, var(--c-blue), var(--c-blue-light));
            color: #0B0B0D;
            box-shadow: 0 2px 10px var(--c-blue-glow);
        }}
        .strategy-badge.vector {{
            background: linear-gradient(135deg, var(--c-brown), var(--c-brown-light));
            color: #241708;
        }}

        /* --- Ask panel: an ash card wrapping the input, instead of bare white space --- */
        .ask-panel {{
            position: relative;
            overflow: hidden;
            background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
            border: 1px solid var(--c-ash-line);
            border-radius: 18px;
            padding: 1.6rem 1.8rem 1.3rem 1.8rem;
            margin-top: 1rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.30);
        }}
        .ask-panel-label {{
            position: relative;
            z-index: 1;
            display: flex;
            align-items: center;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 1.0rem;
            color: var(--c-white);
            margin-bottom: 0.9rem;
        }}
        .ask-panel .node-field {{ opacity: 0.55; }}

        .chip-row {{ position: relative; z-index: 1; display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.9rem; }}
        .example-chip {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            color: var(--c-ash-dark);
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--c-ash-line);
            border-radius: 999px;
            padding: 0.3rem 0.75rem;
        }}

        div[data-testid="stTextInput"] {{ position: relative; z-index: 1; }}
        div[data-testid="stTextInput"] label {{
            color: var(--c-ash-dark) !important;
            font-family: 'Inter', sans-serif;
        }}
        div[data-testid="stTextInput"] input {{
            background: rgba(0,0,0,0.28) !important;
            border: 1px solid var(--c-ash-line) !important;
            border-radius: 10px !important;
            color: var(--c-white) !important;
            caret-color: var(--c-brown-light);
        }}
        div[data-testid="stTextInput"] input:focus {{
            border: 1px solid var(--c-brown) !important;
            box-shadow: 0 0 0 3px var(--c-brown-glow) !important;
        }}
        div[data-testid="stTextInput"] input::placeholder {{ color: var(--c-ash-dim) !important; }}

        div[data-testid="stButton"] button[kind="primary"] {{
            position: relative;
            z-index: 1;
            background: linear-gradient(135deg, var(--c-brown) 0%, var(--c-blue) 130%);
            border: none;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: #12100E;
            box-shadow: 0 4px 14px var(--c-brown-glow);
            transition: box-shadow 0.2s ease, transform 0.15s ease;
        }}
        div[data-testid="stButton"] button[kind="primary"]:hover {{
            box-shadow: 0 6px 22px var(--c-blue-glow);
            transform: translateY(-1px);
        }}

        .stTabs [data-baseweb="tab"] {{ color: var(--c-ash-dark); }}
        .stTabs [aria-selected="true"] {{ color: var(--c-blue-light) !important; }}
        div[data-testid="stExpander"] {{
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--c-ash-line);
            border-radius: 12px;
        }}

        code, .stCodeBlock, pre {{
            font-family: 'JetBrains Mono', monospace !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero — NOVA (shining black) / expanded name (shining blue) / tagline / description
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-wrap">
        <div class="node-field"></div>
        <div class="node-field field-2"></div>
        <div class="nova-title">{PROJECT_TITLE}</div>
        <div class="nova-subtitle">{PROJECT_SUBTITLE}</div>
        <div class="nova-description">{PROJECT_DESCRIPTION}</div>
        <div class="pill-row">
            <div class="pill"><span class="dot"></span>Hybrid Retrieval</div>
            <div class="pill"><span class="dot"></span>Knowledge Graph</div>
            <div class="pill"><span class="dot"></span>Agent AI</div>
            <div class="pill"><span class="dot"></span>Single-hop &amp; Multi-hop</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_ask, tab_about = st.tabs(["Ask", "About the Knowledge Base"])

# ---------------------------------------------------------------------------
# Tab 1 — Ask
# ---------------------------------------------------------------------------
with tab_ask:
    st.markdown(
        """
        <div class="ask-panel">
            <div class="node-field field-2" style="top:-40px; right:-20px; width:200px; height:200px;"></div>
            <div class="ask-panel-label">
                <span class="badge-icon"></span> Ask NOVA about the corpus
            </div>
        """,
        unsafe_allow_html=True,
    )
    question = st.text_input(
        "Ask a question from the corpus:",
        placeholder="e.g. What is SkillOpt? · or · Compare Gorilla and MRKL Systems.",
    )
    ask_clicked = st.button("Ask", type="primary")
    st.markdown(
        """
            <div class="chip-row">
                <span class="example-chip">What is MRKL Systems?</span>
                <span class="example-chip">Compare Gorilla and MRKL Systems</span>
                <span class="example-chip">How are agent skills connected to tool use?</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ask_clicked and question.strip():
        with st.spinner("Routing → retrieving → reasoning..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={"question": question},
                    headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
                    timeout=180,  # multi-hop questions make several LLM calls (planner + per-sub-question reasoning + synthesis) — 60s was too tight even with models pre-baked at build time
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
                            <h4><span class="badge-icon"></span>Answer</h4>
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
# Tab 2 — About the Knowledge Base
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
        <div class="answer-card">
            <h4><span class="badge-icon"></span>Corpus</h4>
            <div class="answer-text">
                NOVA is built over a curated corpus of <strong>30 full-text research
                papers on AI agent capabilities</strong>, sourced from the
                <a href="{CORPUS_REPO_URL}" target="_blank">masamasa59/ai-agent-papers</a>
                collection on GitHub. Unlike systems that index only titles or
                abstracts, NOVA processes the complete PDF documents, with papers
                ranging from approximately 23 to 60 pages — so it can retrieve from
                specific sections (introduction, related work, methods, experiments,
                datasets, results, conclusions) rather than only the abstract.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="answer-card">
            <h4><span class="badge-icon"></span>How NOVA works</h4>
            <div class="answer-text">
                NOVA combines semantic vector retrieval, BM25 keyword retrieval,
                Retrieval-Augmented Generation, a Neo4j knowledge graph, and
                LangGraph-based agent orchestration. When a question comes in,
                NOVA first decides whether it's a simple lookup or whether it
                needs information pulled from multiple sources. Direct questions
                retrieve the most relevant paper sections and answer from those
                directly. More complex questions get divided into smaller
                sub-questions, each retrieved separately, then combined into one
                final answer — with a knowledge graph of papers, concepts,
                methods, datasets, and tasks checked alongside retrieval to catch
                connections keyword matching alone would miss.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="answer-card">
            <h4><span class="badge-icon"></span>What you receive</h4>
            <div class="answer-text">
                For each question, NOVA returns a generated answer, the retrieval
                strategy used (vector-only or hybrid), and the supporting source
                chunk IDs with a count — so every answer stays traceable back to
                the corpus it came from, rather than being a black box.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )