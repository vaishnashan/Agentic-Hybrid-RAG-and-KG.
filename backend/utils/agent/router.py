"""
Router node: given a (sub-)question, decides whether to use vector-only or hybrid
retrieval. This is a strict, deterministic rule tied to hop-type — nothing else
overrides it:

    single-hop  -> vector_only
    multi-hop   -> hybrid_both

The knowledge graph is NOT decided here. It's attempted for every question
regardless of strategy (see node_retrieve in graph_definition.py) — safe_graph_query()
in fallback.py already soft-fails to [] if the graph is unreachable or no known
concept is mentioned, so treating it as "always try, silently optional" is safe and
doesn't need a routing decision of its own.

Note: this used to also check for relational keywords ("compare", "between", ...) or
a matched concept name and bump single-hop questions to hybrid_both on that basis.
That's been removed so the strategy is fully predictable from hop-type alone — if you
want that keyword signal back (e.g. a single-hop question that's still clearly
relational), it can be re-added as an explicit second input to route(), rather than
silently overriding the hop-type decision.
"""
from .schemas import RouteDecision


def route(question: str, requires_graph_hint: bool = False) -> RouteDecision:
    if requires_graph_hint:
        return RouteDecision(
            strategy="hybrid_both",
            reason="Multi-hop question — using hybrid (dense + sparse) retrieval.",
        )
    return RouteDecision(
        strategy="vector_only",
        reason="Single-hop question — vector-only retrieval is sufficient.",
    )


if __name__ == "__main__":
    tests = [
        ("What is SkillOpt?", False),
        ("Which papers discuss Agent Skills?", False),
        ("Compare MRKL and Gorilla's approach to tool use.", True),
    ]
    for q, is_multi_hop in tests:
        decision = route(q, requires_graph_hint=is_multi_hop)
        print(f"Q: {q} (multi_hop={is_multi_hop})")
        print(f"  strategy={decision.strategy} | {decision.reason}\n")