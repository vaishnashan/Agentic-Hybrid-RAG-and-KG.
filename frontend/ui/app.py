"""Professional Streamlit frontend for NOVA.

The UI talks to the FastAPI backend through /ask and intentionally keeps the
frontend independent from the retrieval/agent code.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")

PROJECT_TITLE = "NOVA"
PROJECT_SUBTITLE = "Node Orchestrated Vector and Knowledge Assistant"
PROJECT_DESCRIPTION = (
    "An agentic research assistant that combines dense vector search, BM25, "
    "reranking, and a Neo4j knowledge graph to answer single-hop and multi-hop "
    "questions with grounded evidence from a curated AI-agent research corpus."
)
CORPUS_REPO_URL = "https://github.com/masamasa59/ai-agent-papers"

st.set_page_config(
    page_title=f"{PROJECT_TITLE} — Agentic RAG + Knowledge Graph",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# A self-contained SVG graph pattern. Keeping it in CSS avoids missing-asset
# failures in local/Docker/Render deployments.
NODE_GRAPH_SVG = (
    "%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%22320%22%20height%3D%22320%22%20viewBox%3D%220%200%20320%20320%22%3E"
    "%3Cg%20fill%3D%22none%22%20stroke-width%3D%221.15%22%3E"
    "%3Cline%20x1%3D%2240%22%20y1%3D%2260%22%20x2%3D%22138%22%20y2%3D%22112%22%20stroke%3D%22%23C78952%22%20stroke-opacity%3D%220.18%22/%3E"
    "%3Cline%20x1%3D%22138%22%20y1%3D%22112%22%20x2%3D%22246%22%20y2%3D%2262%22%20stroke%3D%22%2378A8FF%22%20stroke-opacity%3D%220.17%22/%3E"
    "%3Cline%20x1%3D%22138%22%20y1%3D%22112%22%20x2%3D%22116%22%20y2%3D%22228%22%20stroke%3D%22%2354BFA7%22%20stroke-opacity%3D%220.14%22/%3E"
    "%3Cline%20x1%3D%22116%22%20y1%3D%22228%22%20x2%3D%22248%22%20y2%3D%22262%22%20stroke%3D%22%23C78952%22%20stroke-opacity%3D%220.14%22/%3E"
    "%3C/g%3E"
    "%3Ccircle%20cx%3D%2240%22%20cy%3D%2260%22%20r%3D%225%22%20fill%3D%22%23C78952%22%20fill-opacity%3D%220.27%22/%3E"
    "%3Ccircle%20cx%3D%22138%22%20cy%3D%22112%22%20r%3D%227%22%20fill%3D%22%2378A8FF%22%20fill-opacity%3D%220.28%22/%3E"
    "%3Ccircle%20cx%3D%22246%22%20cy%3D%2262%22%20r%3D%225%22%20fill%3D%22%2354BFA7%22%20fill-opacity%3D%220.24%22/%3E"
    "%3Ccircle%20cx%3D%22116%22%20cy%3D%22228%22%20r%3D%225.5%22%20fill%3D%22%23C78952%22%20fill-opacity%3D%220.24%22/%3E"
    "%3Ccircle%20cx%3D%22248%22%20cy%3D%22262%22%20r%3D%225%22%20fill%3D%22%2378A8FF%22%20fill-opacity%3D%220.23%22/%3E"
    "%3C/svg%3E"
)

st.markdown(
    f"""
    <style>
    :root {{
        --bg: #151311;
        --bg-soft: #1D1916;
        --panel: rgba(34, 29, 25, 0.88);
        --panel-2: rgba(42, 35, 30, 0.86);
        --border: rgba(218, 187, 158, 0.16);
        --border-strong: rgba(218, 187, 158, 0.28);
        --text: #F4EFEA;
        --muted: #BDB4AB;
        --muted-2: #8F867E;
        --copper: #C78952;
        --copper-light: #E6AE79;
        --blue: #78A8FF;
        --blue-light: #C9DEFF;
        --teal: #54BFA7;
    }}

    /* Strong fallback against Streamlit's light theme / version-specific CSS. */
    html, body, #root, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }}

    .stApp {{
        background-image:
            radial-gradient(900px 520px at 8% -4%, rgba(120,168,255,0.11), transparent 58%),
            radial-gradient(820px 520px at 96% 2%, rgba(199,137,82,0.14), transparent 55%),
            url("data:image/svg+xml,{NODE_GRAPH_SVG}"),
            linear-gradient(155deg, #141210 0%, #1D1916 54%, #171412 100%) !important;
        background-repeat: no-repeat, no-repeat, repeat, no-repeat !important;
        background-size: auto, auto, 360px 360px, cover !important;
        background-attachment: fixed !important;
    }}

    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu {{
        opacity: 0.82;
    }}

    .block-container {{
        max-width: 1240px !important;
        padding-top: 2.1rem !important;
        padding-bottom: 4rem !important;
    }}

    html, body, p, li, label, div {{
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    @keyframes nova-shine {{
        0% {{ background-position: 0% 50%; }}
        100% {{ background-position: 210% 50%; }}
    }}

    .nova-hero {{
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(300px, 0.55fr);
        gap: 2.1rem;
        align-items: stretch;
        padding: 2.2rem 2.25rem;
        border: 1px solid var(--border);
        border-radius: 28px;
        background: linear-gradient(145deg, rgba(37,31,27,0.93), rgba(25,22,19,0.82));
        box-shadow: 0 22px 70px rgba(0,0,0,0.34);
        backdrop-filter: blur(16px);
        overflow: hidden;
        position: relative;
        margin-bottom: 1.55rem;
    }}

    .nova-hero::after {{
        content: "";
        position: absolute;
        width: 420px;
        height: 420px;
        right: -160px;
        top: -210px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(199,137,82,0.18), transparent 68%);
        pointer-events: none;
    }}

    .eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        color: var(--copper-light);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }}

    .eyebrow-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--copper), var(--blue));
        box-shadow: 0 0 12px rgba(199,137,82,0.55);
    }}

    .nova-title {{
        margin: 0;
        font-size: clamp(5rem, 10vw, 9rem);
        font-weight: 800;
        letter-spacing: -0.065em;
        line-height: 0.95;
        background: linear-gradient(
            110deg,
            #FFFFFF 18%,
            var(--blue-light) 42%,
            var(--blue) 55%,
            #FFFFFF 82%
        );
        background-size: 220% auto;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }}

    .nova-subtitle {{
        margin-top: 0.8rem;
        color: var(--copper-light);
        font-size: clamp(1.05rem, 2vw, 1.32rem);
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    .nova-description {{
        max-width: 760px;
        margin-top: 0.75rem;
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.7;
    }}

    .pill-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.58rem;
        margin-top: 1.25rem;
    }}

    .pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.46rem;
        padding: 0.48rem 0.78rem;
        border-radius: 999px;
        border: 1px solid rgba(230,174,121,0.22);
        color: #E8DDD3;
        background: rgba(199,137,82,0.08);
        font-size: 0.82rem;
        font-weight: 600;
    }}

    .pill-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--copper-light);
    }}

    .flow-card {{
        position: relative;
        z-index: 1;
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.15rem;
        background: linear-gradient(155deg, rgba(255,255,255,0.045), rgba(255,255,255,0.018));
        min-height: 100%;
    }}

    .flow-title {{
        color: var(--muted-2);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }}

    .flow-step {{
        display: flex;
        align-items: center;
        gap: 0.68rem;
        color: #EAE4DE;
        padding: 0.72rem 0.78rem;
        border-radius: 13px;
        background: rgba(8,8,8,0.16);
        border: 1px solid rgba(255,255,255,0.045);
        margin-bottom: 0.55rem;
        font-size: 0.9rem;
        font-weight: 600;
    }}

    .step-no {{
        display: inline-grid;
        place-items: center;
        width: 26px;
        height: 26px;
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(199,137,82,0.28), rgba(120,168,255,0.20));
        color: var(--blue-light);
        font-size: 0.73rem;
        font-weight: 800;
    }}

    .flow-meta {{
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid var(--border);
        color: var(--muted-2);
        font-size: 0.75rem;
    }}

    .section-kicker {{
        color: var(--copper-light);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }}

    .section-title {{
        color: var(--text);
        font-size: clamp(1.7rem, 3vw, 2.35rem);
        font-weight: 780;
        letter-spacing: -0.035em;
        line-height: 1.12;
        margin-bottom: 0.32rem;
    }}

    .section-copy {{
        color: var(--muted);
        max-width: 760px;
        line-height: 1.6;
        margin-bottom: 1.1rem;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1.35rem;
        border-bottom: 1px solid var(--border) !important;
        margin-bottom: 1.1rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #AEA59D !important;
        font-size: 0.97rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 0.15rem !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--text) !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        height: 3px !important;
        background: linear-gradient(90deg, var(--copper), var(--blue)) !important;
        border-radius: 999px !important;
    }}

    /* Large question input */
    div[data-testid="stTextArea"] label {{
        color: #DED6CF !important;
        font-size: 1.03rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.45rem !important;
    }}

    div[data-testid="stTextArea"] textarea {{
        min-height: 142px !important;
        border-radius: 18px !important;
        border: 1px solid var(--border-strong) !important;
        background: linear-gradient(145deg, rgba(10,9,8,0.56), rgba(27,23,20,0.74)) !important;
        color: var(--text) !important;
        font-size: 1.18rem !important;
        line-height: 1.58 !important;
        padding: 1.15rem 1.2rem !important;
        caret-color: var(--copper-light) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.025), 0 14px 32px rgba(0,0,0,0.16) !important;
    }}

    div[data-testid="stTextArea"] textarea:focus {{
        border-color: rgba(230,174,121,0.78) !important;
        box-shadow: 0 0 0 4px rgba(199,137,82,0.12), 0 18px 42px rgba(0,0,0,0.20) !important;
    }}

    div[data-testid="stTextArea"] textarea::placeholder {{
        color: #766F69 !important;
    }}

    div[data-testid="stButton"] button {{
        border-radius: 12px !important;
        min-height: 46px !important;
        font-weight: 750 !important;
        transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease !important;
    }}

    div[data-testid="stButton"] button[kind="primary"] {{
        background: linear-gradient(135deg, var(--copper) 0%, #D99A62 48%, var(--blue) 150%) !important;
        color: #17110D !important;
        border: 0 !important;
        box-shadow: 0 9px 24px rgba(199,137,82,0.23) !important;
        font-size: 1rem !important;
    }}

    div[data-testid="stButton"] button[kind="primary"]:hover {{
        transform: translateY(-1px);
        box-shadow: 0 12px 30px rgba(199,137,82,0.30) !important;
    }}

    div[data-testid="stButton"] button[kind="secondary"] {{
        background: rgba(255,255,255,0.025) !important;
        color: #CFC6BE !important;
        border: 1px solid var(--border) !important;
        font-size: 0.86rem !important;
        text-align: left !important;
    }}

    div[data-testid="stButton"] button[kind="secondary"]:hover {{
        background: rgba(199,137,82,0.08) !important;
        color: #FFFFFF !important;
        border-color: rgba(230,174,121,0.35) !important;
        transform: translateY(-1px);
    }}

    .prompt-label {{
        margin-top: 0.4rem;
        margin-bottom: 0.5rem;
        color: var(--muted-2);
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}

    /* Answer + About cards */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: var(--border) !important;
        border-radius: 18px !important;
        background: linear-gradient(155deg, rgba(38,32,28,0.80), rgba(27,23,20,0.70)) !important;
        box-shadow: 0 12px 38px rgba(0,0,0,0.18) !important;
    }}

    .answer-label {{
        color: var(--copper-light);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.15rem;
    }}

    .strategy-badge {{
        display: inline-flex;
        margin-top: 0.45rem;
        padding: 0.38rem 0.62rem;
        border-radius: 9px;
        border: 1px solid var(--border);
        background: rgba(120,168,255,0.08);
        color: var(--blue-light);
        font-size: 0.76rem;
        font-weight: 750;
    }}

    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {{
        color: #DDD6CF;
        line-height: 1.65;
    }}

    div[data-testid="stExpander"] {{
        border-color: var(--border) !important;
        background: rgba(255,255,255,0.018) !important;
        border-radius: 14px !important;
    }}

    .status-note {{
        color: var(--muted-2);
        font-size: 0.8rem;
        margin-top: 0.45rem;
    }}

    @media (max-width: 860px) {{
        .nova-hero {{ grid-template-columns: 1fr; padding: 1.55rem; border-radius: 22px; }}
        .flow-card {{ display: none; }}
        .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _fill_question(text: str) -> None:
    st.session_state.question_input = text


def _api_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def _ask_backend(question: str) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/ask",
        json={"question": question},
        headers=_api_headers(),
        timeout=180,
    )

    if response.status_code == 429:
        raise RuntimeError("The backend is rate-limited right now. Please wait a moment and try again.")
    if response.status_code == 401:
        raise RuntimeError("Authentication failed. Check that the frontend and backend API keys match.")

    response.raise_for_status()
    return response.json()


st.markdown(
    f"""
    <section class="nova-hero">
        <div>
            <div class="eyebrow"><span class="eyebrow-dot"></span>Agentic Research Intelligence</div>
            <h1 class="nova-title">{PROJECT_TITLE}</h1>
            <div class="nova-subtitle">{PROJECT_SUBTITLE}</div>
            <div class="nova-description">{PROJECT_DESCRIPTION}</div>
            <div class="pill-row">
                <span class="pill"><span class="pill-dot"></span>Hybrid Retrieval</span>
                <span class="pill"><span class="pill-dot"></span>Knowledge Graph</span>
                <span class="pill"><span class="pill-dot"></span>Agent Orchestration</span>
                <span class="pill"><span class="pill-dot"></span>Grounded Answers</span>
            </div>
        </div>
        <aside class="flow-card">
            <div class="flow-title">NOVA query flow</div>
            <div class="flow-step"><span class="step-no">01</span>Plan & classify the question</div>
            <div class="flow-step"><span class="step-no">02</span>Route retrieval strategy</div>
            <div class="flow-step"><span class="step-no">03</span>Retrieve + rerank evidence</div>
            <div class="flow-step"><span class="step-no">04</span>Generate grounded answer</div>
            <div class="flow-meta"><span>30 research papers</span><span>Dense + BM25 + KG</span></div>
        </aside>
    </section>
    """,
    unsafe_allow_html=True,
)

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

ask_tab, about_tab = st.tabs(["Ask NOVA", "About the Knowledge Base"])

with ask_tab:
    st.markdown(
        """
        <div class="section-kicker">Research assistant</div>
        <div class="section-title">Ask a question in natural language.</div>
        <div class="section-copy">NOVA will decide whether the query needs vector-only retrieval or the full hybrid path, then return an evidence-grounded answer.</div>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Your question",
        placeholder="Example: Compare Gorilla and MRKL Systems, and explain how their tool-use approaches differ.",
        key="question_input",
        height=150,
    )

    action_col, note_col = st.columns([1.15, 4.85])
    with action_col:
        ask_clicked = st.button("Ask NOVA", type="primary", use_container_width=True)
    with note_col:
        st.markdown(
            '<div class="status-note">Single-hop and multi-hop questions are supported. Multi-hop answers can take longer because NOVA performs additional reasoning and retrieval steps.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="prompt-label">Suggested prompts</div>', unsafe_allow_html=True)
    examples = [
        "What is MRKL Systems?",
        "Compare Gorilla and MRKL Systems.",
        "How does the knowledge graph help answer multi-hop questions?",
    ]
    example_cols = st.columns(3)
    for col, prompt in zip(example_cols, examples):
        with col:
            st.button(
                prompt,
                key=f"example_{prompt}",
                on_click=_fill_question,
                args=(prompt,),
                use_container_width=True,
            )

    if ask_clicked:
        clean_question = question.strip()
        if not clean_question:
            st.warning("Enter a question before asking NOVA.")
        else:
            with st.spinner("Planning → routing → retrieving → reasoning..."):
                try:
                    result = _ask_backend(clean_question)
                except RuntimeError as exc:
                    st.error(str(exc))
                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not reach the NOVA backend: {exc}")
                except ValueError:
                    st.error("The backend returned a response that was not valid JSON.")
                else:
                    strategy = result.get("strategy_used", "unknown")
                    if strategy == "hybrid_both":
                        strategy_label = "Hybrid retrieval · dense + sparse"
                    elif strategy:
                        strategy_label = strategy.replace("_", " ").title()
                    else:
                        strategy_label = "Strategy not reported"

                    answer = result.get("answer", "No answer was returned by the backend.")
                    sources = result.get("sources", []) or []

                    st.markdown("---")
                    with st.container(border=True):
                        st.markdown('<div class="answer-label">NOVA answer</div>', unsafe_allow_html=True)
                        st.markdown(answer)
                        st.markdown(
                            f'<span class="strategy-badge">{strategy_label}</span>',
                            unsafe_allow_html=True,
                        )

                    if sources:
                        with st.expander(f"View supporting source chunks ({len(sources)})"):
                            for source_id in sources:
                                st.code(str(source_id), language=None)

with about_tab:
    st.markdown(
        """
        <div class="section-kicker">Knowledge base</div>
        <div class="section-title">Built for research-paper reasoning, not generic chat.</div>
        <div class="section-copy">The frontend is intentionally thin: it sends questions to the NOVA API while the backend performs planning, retrieval, reranking, graph lookup, and answer generation.</div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("### Corpus")
            st.markdown(
                f"NOVA is built over **30 full-text AI-agent research papers** from the "
                f"[masamasa59/ai-agent-papers]({CORPUS_REPO_URL}) collection. The system "
                "indexes full-document chunks rather than relying only on titles or abstracts."
            )

        with st.container(border=True):
            st.markdown("### Retrieval")
            st.markdown(
                "Dense semantic retrieval captures meaning, BM25 captures exact lexical matches, "
                "and reranking improves the final evidence order. Complex questions can also use "
                "knowledge-graph context."
            )

    with right:
        with st.container(border=True):
            st.markdown("### Agent flow")
            st.markdown(
                "The planner first decides whether the question is single-hop or multi-hop. "
                "The router then chooses the retrieval strategy before evidence is gathered and "
                "the final grounded response is generated."
            )

        with st.container(border=True):
            st.markdown("### What the UI receives")
            st.markdown(
                "The current API response exposes the generated answer, the retrieval strategy "
                "used, and supporting source chunk identifiers."
            )
