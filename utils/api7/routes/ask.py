"""
POST /ask — the one endpoint that actually runs the agent. Calls ask() from
graph_definition.py directly — same function you've been testing from the CLI,
just now reachable over HTTP. Requires the X-API-Key header and is rate-limited.
"""
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from utils.agent4.graph_definition import ask
from utils.agent4.schemas import FinalAnswer

from utils.api7.auth import require_api_key
from utils.api7.rate_limit import limiter

router = APIRouter()


class AskRequest(BaseModel):
    question: str


@router.post("/ask", response_model=FinalAnswer)
@limiter.limit("30/minute")  # mirrors RATE_LIMIT; slowapi needs the decorator on the route itself
async def ask_endpoint(request: Request, body: AskRequest, _: bool = Depends(require_api_key)):
    return ask(body.question)