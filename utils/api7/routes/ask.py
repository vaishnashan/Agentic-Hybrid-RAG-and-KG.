"""
POST /ask — the main endpoint the UI and Locust hit.

Blocking version first (matches your current ask()); a streaming variant using
reasoner.stream_reason() is sketched below for Day 24-25's UI streaming, but
wiring full end-to-end SSE/websocket streaming through the LangGraph nodes is
a bigger change than a one-file drop-in — see the note at the bottom.
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


# ---------------------------------------------------------------------------
# Streaming sketch for Day 24-25 (not wired up yet — reasoner.stream_reason()
# exists and works standalone, but plugging it into the LangGraph flow means
# the plan/route/retrieve steps run first and only the *reasoning* step
# streams token-by-token; everything before that is still one blocking chunk).
#
# from fastapi.responses import StreamingResponse
# from utils.agent4.planner import plan
# from utils.agent4.router import route
# from utils.retrieval2.hybrid_merge import hybrid_search
# from utils.retrieval2.dense_retriever import dense_search
# from utils.retrieval2.reranker import rerank
# from utils.agent4.fallback import safe_graph_query
# from utils.agent4.reasoner import stream_reason
#
# @router.post("/ask/stream")
# async def ask_stream_endpoint(request: Request, body: AskRequest, _: bool = Depends(require_api_key)):
#     async def event_generator():
#         yield f"data: {{\"stage\": \"planning\"}}\n\n"
#         planner_output = plan(body.question)
#         sub_q = planner_output.sub_questions[0]
#         route_decision = route(sub_q.text, requires_graph_hint=sub_q.requires_graph)
#
#         yield f"data: {{\"stage\": \"retrieving\", \"strategy\": \"{route_decision.strategy}\"}}\n\n"
#         candidates = (dense_search(body.question, top_k=20) if route_decision.strategy == "vector_only"
#                       else hybrid_search(body.question, top_k=20))
#         chunks = rerank(body.question, candidates, top_k=5)
#         graph_facts = safe_graph_query(body.question) if route_decision.strategy != "vector_only" else []
#
#         yield f"data: {{\"stage\": \"reasoning\"}}\n\n"
#         for token in stream_reason(body.question, chunks, graph_facts):
#             yield f"data: {{\"token\": {json.dumps(token)}}}\n\n"
#         yield "data: [DONE]\n\n"
#
#     return StreamingResponse(event_generator(), media_type="text/event-stream")
# ---------------------------------------------------------------------------