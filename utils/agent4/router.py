"""
Router node: given a (sub-)question, decides whether to hit hybrid retrieval, the
knowledge graph, or both. Stays rule-based (cheap, deterministic, fast) — an LLM
classifier would add latency/cost for a decision this simple already handles well.
"""
from utils.agent4.schemas import RouteDecision
from utils.knowledge_graph3.schema import SEED_CONCEPTS

# Relational/multi-hop signal words, plus: if the question directly names one of our
# known concepts, that's also a strong signal the KG can help (not just plain-text signals).
GRAPH_SIGNALS = [
    "relationship", "connect", "compare", "cite", "between", "both",
    "which papers", "across", "related to", "similar to",
]


def route(question: str, requires_graph_hint: bool = False) -> RouteDecision:
    q_lower = question.lower()

    matched_concepts = [c for c in SEED_CONCEPTS if c.lower() in q_lower]

    if requires_graph_hint or any(sig in q_lower for sig in GRAPH_SIGNALS) or len(matched_concepts) >= 1:
        reason = "Multi-hop / relational signal detected."
        if matched_concepts:
            reason += f" Matched known concept(s): {matched_concepts}."
        return RouteDecision(strategy="hybrid_both", reason=reason)

    return RouteDecision(strategy="vector_only", reason="Looks like a simple lookup question.")


if __name__ == "__main__":
    tests = [
        "What is SkillOpt?",
        "Which papers discuss Agent Skills?",
        "Compare MRKL and Gorilla's approach to tool use.",
    ]
    for q in tests:
        decision = route(q)
        print(f"Q: {q}")
        print(f"  strategy={decision.strategy} | {decision.reason}\n")
