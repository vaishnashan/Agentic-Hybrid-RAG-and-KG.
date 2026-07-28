# Agentic RAG + Knowledge Graph over Research Papers

Multi-agent Q&A system over a corpus of research papers. Combines hybrid
(dense + sparse) retrieval with a Neo4j knowledge graph for multi-hop
reasoning, orchestrated by a resilient LangGraph agent, evaluated with RAGAS,
traced with Langfuse, and deployed behind a rate-limited public API.

**Live demo: [add your Hugging Face Space URL here once deployed]**

## Repo map

| Folder | What it does |
|---|---|
| `utils/ingestion1/` | Load PDFs, chunk, embed, build dense + sparse indexes |
| `utils/retrieval2/` | Dense + sparse retrieval, RRF merge, cross-encoder rerank |
| `utils/knowledge_graph3/` | Schema, LLM-based triple extraction, Neo4j client, graph algorithms |
| `utils/agent4/` | LangGraph planner → router → retriever → reasoner → self-critic, plus cache/circuit-breaker/retries/guardrails |
| `utils/observability5/` | Langfuse tracing + monitoring dashboard |
| `utils/evaluation6/` | RAGAS scoring, regression gate, chaos testing |
| `utils/api7/` | FastAPI app — `/ask`, `/health`, `/metrics`, API-key auth, rate limiting |
| `utils/ui8/` | Streamlit demo UI |

## Setup

```bash
git clone <your-repo-url>
cd <repo>
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

cp .env.example .env           # then fill in your real keys
```

## Running everything, in order

```bash
# 1. Ingest + index the corpus
python -m utils.ingestion1.embed_and_index

# 2. Build the knowledge graph
python -m utils.knowledge_graph3.neo4j_client       # sanity check: connection works
python -m utils.knowledge_graph3.extractor          # builds Papers + Concepts/Methods
python -m utils.knowledge_graph3.test_queries       # sanity check: graph answers queries

# 3. Test the agent end-to-end
python -m utils.agent4.graph_definition

# 4. Evaluate
python -m utils.evaluation6.run_golden_set
python -m utils.evaluation6.run_ragas
python -m utils.evaluation6.regression_check

# 5. Run the API
uvicorn utils.api7.main:app --reload --port 8000

# 6. Run the UI (separate terminal, API must be running)
streamlit run utils/ui8/app.py

# 7. Monitoring dashboard (separate terminal)
streamlit run utils/observability5/dashboard_metrics.py
```

## Running with Docker Compose

```bash
docker compose up --build
```
Brings up the API (port 8000) and UI (port 8501) as separate containers.
Neo4j, Chroma, and Redis are all managed cloud services (Aura / Chroma Cloud /
Upstash) — nothing to containerize for them, just credentials in `.env`.

## Calling the API directly

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ask \
  -H "X-API-Key: <your API_KEY from .env>" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is SkillOpt?"}'
```

## Architecture

```
question → guardrails (input_validation) → cache (Upstash, fail-open)
  → LangGraph agent:
      plan → route → retrieve (hybrid + KG) → reason → critique
      (critique can trigger one retry with a different strategy)
  → final answer (cached if confident)
```

Resilience: circuit breakers + retries around every external call (Groq,
Neo4j, Chroma), graceful degradation to hybrid-retrieval-only if the
knowledge graph is unreachable (`fallback.py`), self-critique retry loop.

## Evaluation results
See `utils/evaluation6/reports/` for the latest RAGAS scores and
`evaluation_report_template.md` for the full write-up (fill in after
Day 19-21 load/chaos testing).

## Known limitations
- Multi-hop decomposition currently handles only the first sub-question
  end-to-end.
- Cache is exact-match only, not embedding-similarity based.