# Resilience & Fallback Design

## What happens when each component fails

| Component | Failure mode | Behavior |
|---|---|---|
| Neo4j (knowledge graph) | Connection error / timeout / empty result | `src/agent/fallback.py::safe_graph_query` catches the exception, logs a warning, and returns `[]`. The router treats `[]` as "degrade to vector-only" rather than "no relationships exist" — the request still completes, just without graph context. |
| LLM call | Timeout / malformed JSON output | Wrapped by `src/resilience/circuit_breaker.py::llm_breaker`. After repeated failures the breaker opens and fails fast (no hanging retries against a dead endpoint) rather than degrading — TODO: decide if you want a fallback LLM provider here. |
| Vector DB (Chroma/Qdrant) | Connection error | Wrapped by `vector_db_breaker`. If dense search fails, TODO: fall back to sparse (BM25)-only retrieval rather than failing the whole request. |
| Redis (cache) | Unavailable | Cache reads/writes should fail open (treat as cache miss) rather than failing the request — TODO: wrap `src/resilience/cache.py` calls in try/except that logs and returns `None` on error. |

## Why circuit breakers AND retries
- **Retries (tenacity)** handle transient blips — a single dropped connection, a momentary timeout.
- **Circuit breakers (pybreaker)** handle sustained outages — once a dependency has failed
  `fail_max` times, the breaker "opens" and short-circuits further calls immediately for
  `reset_timeout` seconds, instead of every request paying the full retry-and-timeout cost
  against something that's actually down.

## How to demonstrate this to an interviewer
1. Show a normal request succeeding with graph + vector context.
2. Kill the Neo4j container (`docker stop <neo4j_container>`).
3. Show the same question still answering (vector-only), with `strategy_used` reflecting
   the degraded path.
4. Show the circuit breaker log line / Langfuse trace marking the KG call as failed.

## Golden-set regression gate
`src/evaluation/regression_check.py` reads the latest RAGAS report and fails (non-zero
exit) if any metric drops below `RAGAS_REGRESSION_THRESHOLD` (see `.env.example`). This
is wired into `.github/workflows/ci_cd.yml` as the `evaluation-gate` job — a quality
regression blocks `build-and-deploy` from running.
