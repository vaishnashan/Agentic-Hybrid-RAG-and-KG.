# Agentic RAG + Knowledge Graph over a research-paper corpus

## Architecture

```
backend/
├── main.py                    # FastAPI app entrypoint
├── requirements.txt
├── Dockerfile
├── .env.example                # every env var the project reads, documented
├── data/
│   ├── raw/                    # paper_metadata.jsonl + PDFs (input)
│   └── processed/               # papers.jsonl, chunks.jsonl, bm25_index.pkl (generated)
└── utils/
    ├── storage/                 # WRITES the indexes — run offline, not per-request
    │   ├── config.py             # single source of truth for paths/env vars/model names
    │   ├── loader.py             # PDFs + metadata folder -> papers.jsonl
    │   ├── chunker.py            # papers.jsonl -> chunks.jsonl
    │   ├── dense_index.py        # chunks -> Chroma Cloud (local sentence-transformers)
    │   ├── bm25_index.py         # chunks -> bm25_index.pkl (local disk)
    │   ├── schema.py             # knowledge-graph node/relation schema
    │   ├── neo4j_client.py       # thin wrapper around the Neo4j driver (Aura)
    │   ├── graph_index.py        # papers -> triples (GPT-OSS via Groq) -> Neo4j Aura
    │   └── pipeline.py           # <-- run this to build/rebuild everything
    ├── retrieval/                # READS the indexes — runs per query
    │   ├── query_embedder.py     # question -> embedding, via a cloud API (not local)
    │   ├── dense_retriever.py    # search Chroma Cloud
    │   ├── sparse_retriever.py   # search the local BM25 pickle
    │   ├── hybrid_merge.py       # reciprocal rank fusion of dense + sparse
    │   └── reranker.py           # cross-encoder rerank of the merged candidates
    ├── agent/                    # orchestrates retrieval + graph + LLM into an answer
    │   ├── planner.py            # decides single-hop vs multi-hop (Groq, Llama-3.3-70B)
    │   ├── router.py             # picks vector-only vs hybrid retrieval per sub-question
    │   ├── reasoner.py           # drafts + composes the final answer (Groq, Llama-3.3-70B)
    │   ├── fallback.py           # soft-fails to [] if the graph is unreachable
    │   ├── graph_definition.py   # LangGraph state machine wiring it all together
    │   ├── cache.py, circuit_breaker.py, retry_policy.py, input_validation.py
    │   └── api/                  # FastAPI routes (auth, rate limit, /ask, /health)
    └── observability/
        └── tracing.py            # Langfuse tracing, fails open to a local JSONL log
```

### Why storage/ and retrieval/ are separate packages
`storage/` is what you run **offline**, whenever the PDF corpus changes —
it's the only place that writes to Chroma Cloud, the BM25 pickle, or Neo4j.
`retrieval/` is what the live API calls on **every request** — it only
reads what `storage/` already built. Splitting them this way means the
deployed API never needs the heavier build-time dependencies (batch
embedding, PDF parsing, triple extraction) in its hot path.

## One-time cloud setup

1. **Groq** — API key at https://console.groq.com (used for triple
   extraction and agent reasoning).
2. **Chroma Cloud** — free database at https://trychroma.com (dense vector
   index).
3. **Neo4j AuraDB Free** — free instance at https://console.neo4j.io. Pick
   **AuraDB Free** specifically (permanent, no card required) — not the
   14-day AuraDB Professional trial, which looks similar but expires.
4. **Hugging Face** — access token at https://huggingface.co/settings/tokens
   (used for query-time embeddings via the Inference API).
5. Copy `.env.example` to `.env` and fill in the values from steps 1-4.

## Running the storage pipeline (build/rebuild the indexes)

Run this any time you add papers to `data/raw/` or change chunking/
embedding parameters. Run it **from the directory containing `backend/`**
(so `utils` resolves as a package), or `cd backend` first — either way, from
inside `backend/`:

```bash
python -m utils.storage.pipeline
```

Useful flags:

```bash
# Already have papers.jsonl / chunks.jsonl and just want to rebuild indexes:
python -m utils.storage.pipeline --skip-load --skip-chunk

# Only rebuild the dense + sparse indexes, skip the graph step (e.g. no
# GROQ_API_KEY set yet):
python -m utils.storage.pipeline --only dense,bm25
```

This runs, in order: **load** (PDFs → text) → **chunk** (text → chunks) →
**store**, which builds all three indexes from those chunks/papers:
dense (Chroma Cloud), sparse (BM25, local disk), and graph (Neo4j Aura, via
GPT-OSS triple extraction on Groq).

## Running the API

```bash
uvicorn main:app --reload --port 8000
```

or via Docker:

```bash
docker build -t agentic-rag-api .
docker run --env-file .env -p 8000:8000 agentic-rag-api
```

`POST /ask` (header `X-API-Key: <API_KEY from .env>`, body
`{"question": "..."}`) runs the full plan → retrieve → reason → compose
pipeline and returns a `FinalAnswer`. `GET /health` needs no auth.

## Deployment notes (no VM required)

- The dense index lives in **Chroma Cloud**, not on local disk — nothing to
  persist on the API host.
- The BM25 index is a small pickle file baked into the deploy image itself
  (see `Dockerfile`'s `COPY . .` — it deliberately includes
  `data/processed/`).
- The knowledge graph lives in **Neo4j Aura**, not on the API host.
- Query-time embeddings call the **Hugging Face Inference API** instead of
  loading sentence-transformers in-process — the only local model the
  deployed API loads is the small cross-encoder reranker (set
  `RERANKER_ENABLED=false` to drop that too, on very constrained hosts).

Together this means the API container only needs to run FastAPI + a small
reranker model — light enough for Render/Railway/Fly-style free tiers
without a dedicated VM.
