"""
POST /ask — the one endpoint that actually runs the agent. Calls ask() from
graph_definition.py directly — same function you've been testing from the CLI,
just now reachable over HTTP. Requires the X-API-Key header and is rate-limited.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from codebase.backend.utils.agent4.graph_definition import ask
from codebase.backend.utils.agent4.schemas import FinalAnswer

from codebase.backend.utils.api7.auth import require_api_key
from codebase.backend.utils.api7.rate_limit import limiter

router = APIRouter()


class AskRequest(BaseModel):
    question: str


@router.post("/ask", response_model=FinalAnswer)
@limiter.limit("30/minute")  # mirrors RATE_LIMIT; slowapi needs the decorator on the route itself
async def ask_endpoint(request: Request, body: AskRequest, _: bool = Depends(require_api_key)):
    # ask() is a heavy synchronous call (embeddings, cross-encoder inference, Neo4j,
    # Groq HTTP calls). Calling it directly here would block the single-threaded
    # event loop for the whole request, freezing every other request (including
    # /health) until it finishes. run_in_threadpool offloads it to a worker thread
    # so the event loop stays free to serve concurrent requests.
    return await run_in_threadpool(ask, body.question)