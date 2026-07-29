"""
Wires planner -> router -> retriever -> reasoner -> compose -> self_critic (-> retry
loop) into a LangGraph StateGraph. This is the actual "agent" — the piece that turns
retrieval + the knowledge graph into one system that answers a question end to end.

Multi-hop handling: the planner may decompose a question into several sub-questions.
Each sub-question is routed and retrieved independently (a "compare X and Y" question
often needs different strategies for X than for Y), reasoned over independently, and
then all sub-answers are composed into one final answer before critique. A retry
resets the whole sub-question loop with a forced strategy rather than re-running a
single question.
"""
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END

from utils.agent4.planner import plan
from utils.agent4.router import route
from utils.agent4.reasoner import reason, compose_multi_hop_answer
from utils.agent4.self_critic import critique
from utils.agent4.fallback import safe_graph_query
from utils.agent4.schemas import FinalAnswer, ReasonerOutput, SubQuestion
from utils.retrieval2.dense_retriever import dense_search
from utils.retrieval2.hybrid_merge import hybrid_search
from utils.retrieval2.reranker import rerank
from utils.agent4.cache import get_cached_answer, set_cached_answer
from utils.agent4.input_validation import validate_input


class AgentState(TypedDict):
    question: str
    sub_questions: List[SubQuestion]
    current_sub_index: int
    sub_answers: List[dict]  # {"sub_question", "answer", "used_context_ids", "graph_facts", "strategy"}
    strategy: str
    force_strategy: Optional[str]  # set on retry to force one strategy across all sub-questions
    context_chunks: List
    graph_facts: List[str]
    draft_answer: Optional[str]
    used_context_ids: List[str]
    used_graph_facts: List[str]
    confidence: float
    retries: int
    final_answer: Optional[FinalAnswer]


def node_plan(state: AgentState) -> AgentState:
    print("\n[PLAN] Analyzing question...")
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

    if state.get("force_strategy"):
        # Retry path: apply the critic's chosen strategy to every sub-question,
        # rather than re-deriving a (possibly identical) route decision.
        strategy = state["force_strategy"]
        print(f"[ROUTE] strategy={strategy} (forced by retry)")
    else:
        route_decision = route(sub_q.text, requires_graph_hint=sub_q.requires_graph)
        strategy = route_decision.strategy
        print(f"[ROUTE] strategy={strategy}")
        print(f"[ROUTE] reason: {route_decision.reason}")

    state["strategy"] = strategy

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

    if strategy in ("hybrid_both", "graph_only"):
        print("[RETRIEVE] Querying knowledge graph...")
        state["graph_facts"] = safe_graph_query(sub_q.text)
        print(f"[RETRIEVE] Graph returned {len(state['graph_facts'])} fact(s)")
        for f in state["graph_facts"]:
            print(f"    - {f}")
    else:
        state["graph_facts"] = []
        print("[RETRIEVE] Skipping knowledge graph (strategy=vector_only)")

    return state


def node_reason(state: AgentState) -> AgentState:
    idx = state["current_sub_index"]
    sub_q = state["sub_questions"][idx]
    print(f"\n[REASON] Drafting answer for sub-question {idx + 1}/{len(state['sub_questions'])}...")

    result = reason(sub_q.text, state["context_chunks"], state["graph_facts"])
    print(f"[REASON] Draft answer: {result.draft_answer[:200]}")

    state["sub_answers"].append({
        "sub_question": sub_q.text,
        "answer": result.draft_answer,
        "used_context_ids": result.used_context_ids,
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
    print(f"\n[COMPOSE] Combining {len(state['sub_answers'])} sub-answer(s) into final draft...")

    combined = compose_multi_hop_answer(state["question"], state["sub_answers"])

    # Union used_context_ids / graph_facts across all sub-questions, preserving order,
    # de-duplicated — this is what the critic and the final answer's "sources" see.
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
    return state


def node_critique(state: AgentState) -> AgentState:
    print("\n[CRITIQUE] Evaluating draft answer...")
    reasoner_output = ReasonerOutput(
        draft_answer=state["draft_answer"],
        used_context_ids=state["used_context_ids"],
        used_graph_facts=state["used_graph_facts"],
    )
    verdict = critique(reasoner_output, state["strategy"], state["retries"])
    state["confidence"] = verdict.confidence_score

    print(f"[CRITIQUE] confident={verdict.confident}, confidence_score={verdict.confidence_score}")
    if verdict.issues:
        print(f"[CRITIQUE] issues: {verdict.issues}")

    if verdict.should_retry:
        state["retries"] += 1
        state["force_strategy"] = verdict.retry_strategy or state["strategy"]
        state["current_sub_index"] = 0
        state["sub_answers"] = []
        print(f"[CRITIQUE] RETRYING (attempt {state['retries']}) with forced strategy='{state['force_strategy']}' "
              f"across all sub-questions")
        return state

    print("[CRITIQUE] Accepting answer as final.")
    state["final_answer"] = FinalAnswer(
        question=state["question"],
        answer=state["draft_answer"],
        sources=state["used_context_ids"],
        confidence=state["confidence"],
        strategy_used=state["strategy"],
        retries=state["retries"],
    )
    return state


def should_retry(state: AgentState) -> str:
    decision = (
        "retrieve"
        if state.get("final_answer") is None and state["retries"] <= 1 and state["confidence"] < 0.5
        else "end"
    )
    print(f"[ROUTER-EDGE] should_retry -> '{decision}'")
    return decision


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("plan", node_plan)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("reason", node_reason)
    graph.add_node("compose", node_compose)
    graph.add_node("critique", node_critique)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_conditional_edges("reason", has_more_subquestions, {"retrieve": "retrieve", "compose": "compose"})
    graph.add_edge("compose", "critique")
    graph.add_conditional_edges("critique", should_retry, {"retrieve": "retrieve", "end": END})

    return graph.compile()


def ask(question: str) -> FinalAnswer:
    print("=" * 70)
    print(f"AGENT STARTED — Question: {question}")
    print("=" * 70)

    validation = validate_input(question)
    if not validation.is_valid:
        print(f"[GUARDRAILS] REJECTED — {validation.reason}")
        return FinalAnswer(
            question=question,
            answer=f"Request rejected: {validation.reason}",
            sources=[],
            confidence=0.0,
            strategy_used="rejected",
            retries=0,
        )
    print("[GUARDRAILS] Input passed validation.")

    cached = get_cached_answer(question)
    if cached is not None:
        print("[CACHE] HIT — returning cached answer, skipping retrieval + LLM calls.")
        return FinalAnswer(**cached)
    print("[CACHE] MISS — running full pipeline.")

    app = build_agent_graph()
    initial_state: AgentState = {
        "question": question,
        "sub_questions": [],
        "current_sub_index": 0,
        "sub_answers": [],
        "strategy": "vector_only",
        "force_strategy": None,
        "context_chunks": [],
        "graph_facts": [],
        "draft_answer": None,
        "used_context_ids": [],
        "used_graph_facts": [],
        "confidence": 0.0,
        "retries": 0,
        "final_answer": None,
    }
    result_state = app.invoke(initial_state)
    final = result_state["final_answer"]

    print("\n" + "=" * 70)
    print("AGENT FINISHED")
    print("=" * 70)
    print(f"Answer     : {final.answer}")
    print(f"Confidence : {final.confidence}")
    print(f"Strategy   : {final.strategy_used}")
    print(f"Retries    : {final.retries}")
    print(f"Sources    : {final.sources}")
    print("=" * 70)

    # Only cache confident answers — caching a low-confidence/degraded answer would
    # serve that same weak answer to everyone who asks a similar question later.
    if final.confidence >= 0.7:
        cache_success = set_cached_answer(question, final.model_dump())
        print(f"[CACHE] {'Stored' if cache_success else 'Store failed (Upstash unreachable) — not fatal'}")

    return final


if __name__ == "__main__":
    ask("What is SkillOpt?")