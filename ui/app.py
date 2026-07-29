"""
Streamlit UI for the agent — two tabs: Ask (the actual demo) and About the
Knowledge Base (what the corpus is, how NOVA works, where it came from).

Calls the FastAPI /ask endpoint over HTTP rather than importing utils.agent4
directly — keeps the UI as a genuine separate service.

Run locally:
    streamlit run utils/ui/app.py

Set API_BASE_URL and API_KEY as env vars (or edit the defaults below) to point
at your running FastAPI instance.

Design notes: palette is white / ash-gray / near-black / a single "shiny" blue
accent, per brief. NOVA (the name) renders in a shining near-black gradient at
the very top with minimal lead-in space; the expanded name renders in a
shining blue gradient directly beneath it. Both use a slow animated sheen
(background-position keyframe) rather than a static gradient, since "shining"
was requested literally, not just as a color choice.

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
            --c-black-light: #4B4B52;
            --c-blue: #2563EB;
            --c-blue-light: #5B9BFF;
            --c-blue-glow: rgba(37, 99, 235, 0.35);
        }

        .stApp { background-color: var(--c-bg); }
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* Enough top padding that Streamlit's own toolbar doesn't crowd NOVA,
           and enough line-height that the gradient-clipped text isn't clipped
           at the top of its own line box. */
        div.block-container { padding-top: 2.6rem !important; }

        @keyframes shine-sweep {
            0%   { background-position: 0% 50%; }
            100% { background-position: 200% 50%; }
        }

        .nova-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 3.2rem;
            letter-spacing: -0.02em;
            line-height: 1.25;
            padding-top: 0.15rem;
            margin: 0 0 0.15rem 0;
            background: linear-gradient(110deg, var(--c-black) 20%, var(--c-black-light) 50%, var(--c-black) 80%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: shine-sweep 4s linear infinite;
        }

        .nova-subtitle {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 1.15rem;
            letter-spacing: -0.005em;
            margin: 0 0 0.6rem 0;
            background: linear-gradient(110deg, var(--c-blue) 20%, var(--c-blue-light) 50%, var(--c-blue) 80%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: shine-sweep 4s linear infinite;
        }

        .nova-description {
            font-family: 'Inter', sans-serif;
            color: var(--c-ash-dark);
            font-size: 0.98rem;
            max-width: 780px;
            line-height: 1.55;
            margin-bottom: 1.3rem;
        }

        .pill-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.4rem; }
        .pill {
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            font-weight: 500;
            padding: 0.32rem 0.85rem;
            border-radius: 999px;
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
            margin-bottom: 0.9rem;
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
            font-size: 1.0rem;
            color: var(--c-black);
            line-height: 1.6;
        }
        .answer-text ul { margin: 0.3rem 0 0.3rem 1.1rem; padding: 0; }
        .answer-text li { margin-bottom: 0.25rem; }

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
# Hero — NOVA (shining black) / expanded name (shining blue) / tagline / description
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="nova-title">{PROJECT_TITLE}</div>
    <div class="nova-subtitle">{PROJECT_SUBTITLE}</div>
    <div class="nova-description">{PROJECT_DESCRIPTION}</div>
    <div class="pill-row">
        <div class="pill">Hybrid Retrieval</div>
        <div class="pill">Knowledge Graph</div>
        <div class="pill">Agent AI</div>
        <div class="pill">Single-hop &amp; Multi-hop</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_ask, tab_about = st.tabs(["Ask", "About the Knowledge Base"])

# ---------------------------------------------------------------------------
# Tab 1 — Ask
# ---------------------------------------------------------------------------
with tab_ask:
    question = st.text_input(
        "Ask a question from the corpus:",
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
# Tab 2 — About the Knowledge Base
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
        <div class="answer-card">
            <h4>Corpus</h4>
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

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="answer-card">
                <h4>Tool-use — 19 papers</h4>
                <div class="answer-text">
                    How agents discover available tools, select the right one for a
                    task, generate API calls, chain multiple tools together, and
                    learn tool-use strategies. Covers systems such as MRKL, Gorilla,
                    tool-augmented language models, and API-based agents.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="answer-card">
                <h4>Knowledge — 11 papers</h4>
                <div class="answer-text">
                    How agents access, organize, and reason over knowledge —
                    external retrieval, internal memory, knowledge graphs,
                    long-term agent memory, and structured multi-step reasoning
                    across sources.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="answer-card">
            <h4>How NOVA works</h4>
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

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            """
            <div class="answer-card">
                <h4>Semantic vector search</h4>
                <div class="answer-text">
                    Retrieves passages by meaning. A question like
                    <em>"How do agents interact with external services?"</em>
                    still matches sections using different wording — "tool
                    invocation," "API execution," "external tool calling" —
                    because their meanings are close, even if the exact words differ.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="answer-card">
                <h4>BM25 keyword search</h4>
                <div class="answer-text">
                    Retrieves passages by exact words and term importance —
                    especially useful for model names, system names, acronyms,
                    dataset names, and technical terms. A question naming
                    "MRKL," "Gorilla," or a specific benchmark benefits from
                    exact keyword matching that semantic search alone can blur.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="answer-card">
            <h4>Knowledge graph support</h4>
            <div class="answer-text">
                The knowledge graph stores papers, methods, concepts, datasets,
                and tasks as connected nodes and relationships, for example:
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(
        "Paper  → PROPOSES         → Method\n"
        "Paper  → MENTIONS_CONCEPT → Concept\n"
        "Paper  → EVALUATED_ON     → Dataset\n"
        "Paper  → SOLVES_TASK      → Task\n"
        "Method → SUPPORTS         → Concept",
        language="text",
    )
    st.markdown(
        """
        <div class="answer-text" style="margin-bottom: 1.2rem;">
            Instead of only finding chunks that contain the phrase "Agent Skills,"
            the knowledge graph can also surface papers related to that concept
            and the methods associated with it — especially useful for questions
            that require relationships or comparisons across papers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="answer-card">
            <h4>Ask a direct question (single-hop)</h4>
            <div class="answer-text">
                A direct question usually asks for one specific piece of
                information. NOVA searches the corpus, retrieves the most
                relevant sections, reranks them, and generates a grounded answer.
                <ul>
                    <li>"What is MRKL Systems?"</li>
                    <li>"What is SkillOpt?"</li>
                    <li>"Which paper proposed Gorilla?"</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="answer-card">
            <h4>Ask a comparison or connection question (multi-hop)</h4>
            <div class="answer-text">
                A multi-hop question needs information from more than one paper,
                concept, or method:
                <ul>
                    <li>"Compare Gorilla and MRKL Systems."</li>
                    <li>"How are agent skills connected to tool use?"</li>
                    <li>"What is the relationship between self-evolution and reinforcement learning?"</li>
                </ul>
                For these, NOVA identifies the question as multi-hop, divides it
                into sub-questions, retrieves evidence for each one separately,
                checks the knowledge graph for related facts, answers each
                sub-question, and combines the results into one final response —
                reasoning across multiple papers rather than a single passage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="answer-card">
            <h4>What you receive</h4>
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