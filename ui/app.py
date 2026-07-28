"""
Streamlit UI — Day 24-25: "ask questions, view sources, view confidence, and a
dashboard panel showing quality/latency/cost trends."

Calls the FastAPI /ask endpoint over HTTP rather than importing utils.agent4
directly — keeps the UI as a genuine separate service (matches the "API, UI,
Neo4j, vector DB, Redis" separate-containers picture in Day 24-25/docker-compose).

Run locally:
    streamlit run utils/ui/app.py

Set API_BASE_URL and API_KEY as env vars (or edit the defaults below) to point
at your running FastAPI instance.
"""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")

st.set_page_config(page_title="Agentic RAG + Knowledge Graph Demo", layout="wide")
st.title("Agentic RAG + Knowledge Graph — Live Demo")
st.caption(f"Talking to API at: {API_BASE_URL}")

with st.sidebar:
    st.header("About")
    st.write(
        "Ask a question about the paper corpus. The agent routes between "
        "dense+sparse hybrid retrieval and a Neo4j knowledge graph, self-checks "
        "its own answer, and retries with a different strategy if unconfident."
    )
    st.write("Try a simple lookup, e.g. *'What is MRKL Systems?'*, or a multi-hop "
             "question, e.g. *'Compare Gorilla and MRKL Systems.'*")

question = st.text_input("Ask a question:", placeholder="What is SkillOpt?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Retrieving → reasoning → answering..."):
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

                st.subheader("Answer")
                st.write(result["answer"])

                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence", f"{result['confidence']:.2f}")
                col2.metric("Strategy used", result["strategy_used"])
                col3.metric("Retries", result["retries"])

                if result.get("sources"):
                    with st.expander(f"Sources ({len(result['sources'])} chunks)"):
                        for source_id in result["sources"]:
                            st.code(source_id)

        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")

st.divider()
st.caption(
    "See the monitoring dashboard for quality/latency/cost trends: "
    "`streamlit run utils/observability/dashboard_metrics.py`"
)