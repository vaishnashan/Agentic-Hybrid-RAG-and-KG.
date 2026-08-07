"""
Wires planner -> router -> retriever -> reasoner -> compose into a LangGraph
StateGraph. This is the actual "agent" — the piece that turns retrieval + the
knowledge graph into one system that answers a question end to end.

Each node's real work is wrapped in a tracing.RequestTrace span (plan/retrieve/
reason/compose), so every ask() call produces a full trace — visible in Langfuse
if configured, and always appended to the local traces.jsonl regardless (tracing
fails open, same pattern as cache.py). trace.finish() is called on every exit path
(validation reject, cache hit, full run) so the root span always closes and flushes.

No self-critic / retry loop: the composed answer from the LLM is treated as final.
(self_critic.py still exists in the codebase but is no longer wired in here — delete
it if you don't plan to reintroduce it later.)

Multi-hop handling: the planner may decompose a question into several sub-questions.
Each sub-question is routed and retrieved independently (a "compare X and Y" question
often needs different strategies for X than for Y), reasoned over independently, and
then all sub-answers are composed into one final answer.
"""
from typing import TypedDict, List, Optional, Any

from langgraph.graph import StateGraph, END

from codebase.backend.utils.agent.planner import plan
from codebase.backend.utils.agent.router import route
from codebase.backend.utils.agent.reasoner import reason, compose_multi_hop_answer
from codebase.backend.utils.agent.fallback import safe_graph_query
from codebase.backend.utils.agent.schemas import FinalAnswer, SubQuestion
from codebase.backend.utils.storage.retrieval.dense_retriever import dense_search
from codebase.backend.utils.storage.retrieval.hybrid_merge import hybrid_search
from codebase.backend.utils.storage.retrieval.reranker import rerank
from codebase.backend.utils.agent.cache import get_cached_answer, set_cached_answer
from codebase.backend.utils.agent.input_validation import validate_input
from codebase.backend.utils.observability.tracing import RequestTrace


class AgentState(TypedDict):
    question: str
    trace: Any  # RequestTrace — not a Pydantic/dataclass field, just carried through state
    sub_questions: List[SubQuestion]
    current_sub_index: int
    sub_answers: List[dict]  # {"sub_question", "answer", "used_context_ids", "graph_facts", "strategy"}
    strategy: str
    context_chunks: List
    graph_facts: List[str]
    draft_answer: Optional[str]
    used_context_ids: List[str]
    used_graph_facts: List[str]
    final_answer: Optional[FinalAnswer]


def node_plan(state: AgentState) -> AgentState:
    print("\n[PLAN] Analyzing question...")
    with state["trace"].span("plan", question=state["question"]):
        planner_output = plan(state["question"])
    print(f"[PLAN] is_multi_hop={planner_output.is_multi_hop}, "
          f"sub_questions={[sq.text for sq in planner_output.sub_questions]}")

    state["sub_questions"] = planner_output.sub_questions
    state["current_sub_index"] = 0
    state["sub_answers"] = []
    return state


def node_retrieve(state: AgentState) -> AgentState:
    idx = state["current_sub_index"]
    sub_q = state["sub_questions"][idx]
    print(f"\n[RETRIEVE] Sub-question {idx + 1}/{len(state['sub_questions'])}: {sub_q.text}")

    route_decision = route(sub_q.text, requires_graph_hint=sub_q.requires_graph)
    strategy = route_decision.strategy
    print(f"[ROUTE] strategy={strategy}")
    print(f"[ROUTE] reason: {route_decision.reason}")
    state["strategy"] = strategy

    with state["trace"].span("retrieve", sub_question=sub_q.text, sub_question_index=idx, strategy=strategy):
        if strategy == "vector_only":
            candidates = dense_search(sub_q.text, top_k=20)
            print(f"[RETRIEVE] dense_search returned {len(candidates)} candidates")
        else:
            candidates = hybrid_search(sub_q.text, top_k=20)
            print(f"[RETRIEVE] hybrid_search returned {len(candidates)} candidates")

        reranked = rerank(sub_q.text, candidates, top_k=5)
        state["context_chunks"] = reranked
        print(f"[RETRIEVE] reranker kept top {len(reranked)} chunks")
        for c in reranked:
            print(f"    - {c.chunk_id} (score={c.score:.3f}) {c.metadata.get('title', '')[:50]}")

        # KG is attempted for every question regardless of strategy — safe_graph_query()
        # soft-fails to [] if the graph is unreachable or no known concept is mentioned.
        print("[RETRIEVE] Querying knowledge graph (optional — soft-fails to [] if unavailable)...")
        state["graph_facts"] = safe_graph_query(sub_q.text)
        print(f"[RETRIEVE] Graph returned {len(state['graph_facts'])} fact(s)")
        for f in state["graph_facts"]:
            print(f"    - {f}")

    return state


def node_reason(state: AgentState) -> AgentState:
    idx = state["current_sub_index"]
    sub_q = state["sub_questions"][idx]
    print(f"\n[REASON] Drafting answer for sub-question {idx + 1}/{len(state['sub_questions'])}...")

    with state["trace"].span("reason", sub_question=sub_q.text, sub_question_index=idx):
        result = reason(sub_q.text, state["context_chunks"], state["graph_facts"])
    print(f"[REASON] Draft answer: {result.draft_answer[:200]}")

    state["sub_answers"].append({
        "sub_question": sub_q.text,
        "answer": result.draft_answer,
        "used_context_ids": result.used_context_ids,
        "context_texts": [c.text for c in state["context_chunks"]],  # real text, for RAGAS faithfulness scoring
        "graph_facts": state["graph_facts"],
        "strategy": state["strategy"],
    })
    state["current_sub_index"] = idx + 1
    return state


def has_more_subquestions(state: AgentState) -> str:
    decision = "retrieve" if state["current_sub_index"] < len(state["sub_questions"]) else "compose"
    print(f"[ROUTER-EDGE] has_more_subquestions -> '{decision}'")
    return decision


def node_compose(state: AgentState) -> AgentState:
    print(f"\n[COMPOSE] Combining {len(state['sub_answers'])} sub-answer(s) into final answer...")

    with state["trace"].span("compose", n_sub_answers=len(state["sub_answers"])):
        combined = compose_multi_hop_answer(state["question"], state["sub_answers"])

    used_ids: List[str] = []
    used_facts: List[str] = []
    for sa in state["sub_answers"]:
        for cid in sa["used_context_ids"]:
            if cid not in used_ids:
                used_ids.append(cid)
        for fact in sa["graph_facts"]:
            if fact not in used_facts:
                used_facts.append(fact)

    state["draft_answer"] = combined
    state["used_context_ids"] = used_ids
    state["used_graph_facts"] = used_facts
    print(f"[COMPOSE] Combined answer: {combined[:200]}")

    # No self-critic step: the composed LLM answer is accepted as final directly.
    state["final_answer"] = FinalAnswer(
        question=state["question"],
        answer=combined,
        sources=used_ids,
        confidence=1.0,       # no critic scoring this anymore — not a measured value
        strategy_used=state["strategy"],
        retries=0,
    )
    return state


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("plan", node_plan)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("reason", node_reason)
    graph.add_node("compose", node_compose)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_conditional_edges("reason", has_more_subquestions, {"retrieve": "retrieve", "compose": "compose"})
    graph.add_edge("compose", END)

    return graph.compile()


# Compiled once at import time and reused across every request — building
# and compiling a fresh StateGraph on every single call (as before) is pure
# wasted work; the compiled graph is stateless and safe to share.
_compiled_agent_graph = build_agent_graph()


def run_pipeline(question: str, trace: Optional[RequestTrace] = None) -> AgentState:
    """
    Runs the full plan -> retrieve -> reason -> compose graph and returns the
    COMPLETE final state — not just the FinalAnswer. ask() uses this for the normal
    request path (cache-aware); evaluation scripts (run_golden_set.py, run_ragas.py)
    call this directly instead, because they need the retrieved context text and
    per-sub-question breakdown, not just the composed answer + chunk IDs.

    Always a fresh run — no cache check here. Evaluation should never be shortcut by
    a cached answer from a previous (possibly differently-configured) run.
    """
    if trace is None:
        trace = RequestTrace(question)

    app = _compiled_agent_graph
    initial_state: AgentState = {
        "question": question,
        "trace": trace,
        "sub_questions": [],
        "current_sub_index": 0,
        "sub_answers": [],
        "strategy": "vector_only",
        "context_chunks": [],
        "graph_facts": [],
        "draft_answer": None,
        "used_context_ids": [],
        "used_graph_facts": [],
        "final_answer": None,
    }
    return app.invoke(initial_state)


def ask(question: str) -> FinalAnswer:
    print("=" * 70)
    print(f"AGENT STARTED — Question: {question}")
    print("=" * 70)

    trace = RequestTrace(question)

    validation = validate_input(question)
    if not validation.is_valid:
        print(f"[GUARDRAILS] REJECTED — {validation.reason}")
        final = FinalAnswer(
            question=question,
            answer=f"Request rejected: {validation.reason}",
            sources=[],
            confidence=0.0,
            strategy_used="rejected",
            retries=0,
        )
        trace.finish(final.model_dump())
        return final
    print("[GUARDRAILS] Input passed validation.")

    cached = get_cached_answer(question)
    if cached is not None:
        print("[CACHE] HIT — returning cached answer, skipping retrieval + LLM calls.")
        final = FinalAnswer(**cached)
        trace.finish({**final.model_dump(), "cache_hit": True})
        return final
    print("[CACHE] MISS — running full pipeline.")

    result_state = run_pipeline(question, trace)
    final = result_state["final_answer"]

    print("\n" + "=" * 70)
    print("AGENT FINISHED")
    print("=" * 70)
    print(f"Answer     : {final.answer}")
    print(f"Strategy   : {final.strategy_used}")
    print(f"Sources    : {final.sources}")
    print("=" * 70)

    # No confidence gate anymore (that was the critic's job) — every completed
    # run gets cached. Cache write still fails open (not fatal) if Upstash is down.
    cache_success = set_cached_answer(question, final.model_dump())
    print(f"[CACHE] {'Stored' if cache_success else 'Store failed (Upstash unreachable) — not fatal'}")

    trace.finish(final.model_dump())
    return final


if __name__ == "__main__":
    ask("What is the relationship between Agent Skills and SkillOpt?")