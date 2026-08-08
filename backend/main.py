
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from utils.agent.api.rate_limit import limiter
from utils.agent.api.routes import ask, health

app = FastAPI(
    title="Agentic RAG + Knowledge Graph API",
    description="Hybrid retrieval + knowledge-graph agent over a research-paper corpus.",
    version="1.0.0",
)

# CORS open by default so a separate-origin UI (Streamlit/Gradio/etc.) can call
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
app.include_router(ask.router)