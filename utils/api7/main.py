"""
FastAPI app — Day 22-23: "Expose the agent via FastAPI endpoints (ask, health,
metrics); add API-key auth and rate limiting."

Run locally:
    uvicorn utils.api7.main:app --reload --port 8000

Then:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/ask \\
      -H "X-API-Key: <your API_KEY>" -H "Content-Type: application/json" \\
      -d '{"question": "What is SkillOpt?"}'
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from utils.api7.rate_limit import limiter
from utils.api7.routes import health
from utils.api7.routes import metrics
from utils.api7.routes import ask

app = FastAPI(
    title="Agentic RAG + Knowledge Graph API",
    description="Hybrid retrieval + knowledge-graph agent over a research-paper corpus.",
    version="1.0.0",
)

# CORS open by default so the Streamlit/Gradio UI (a separate origin) can call
# this API — tighten allow_origins to your actual UI's URL once deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(ask.router)