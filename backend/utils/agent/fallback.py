"""
Graceful-degradation logic: if the knowledge graph is unavailable (connection error,
timeout, empty result), degrade to hybrid-retrieval-only instead of failing the
request. Callers must treat an empty list as "the graph couldn't help / wasn't
reachable" — NOT as "no relationships exist".

Rewritten to match the ACTUAL Neo4jClient interface (papers_for_concept /
shared_concepts) — the original draft assumed a multi_hop_neighbors() method that
was never built; this version queries what's really in the graph.
"""
import logging

from codebase.backend.utils.storage.knowledge_graph3.falkordb_client import Neo4jClient
from codebase.backend.utils.storage.knowledge_graph3.schema import SEED_CONCEPTS

logger = logging.getLogger("fallback")


def match_concepts_in_question(question: str) -> list:
    """Finds which known concept names are directly mentioned in the question text."""
    q_lower = question.lower()
    return [c for c in SEED_CONCEPTS if c.lower() in q_lower]


def safe_graph_query(question: str) -> list:
    """
    Looks up papers related to any concept(s) mentioned in the question. Returns a
    list of human-readable fact strings for the reasoner, or [] if the graph is
    unreachable or no concepts matched — either way, the caller should degrade to
    hybrid-retrieval-only rather than fail.
    """
    matched_concepts = match_concepts_in_question(question)
    if not matched_concepts:
        return []

    try:
        client = Neo4jClient()
        client.verify_connectivity()

        facts = []
        for concept in matched_concepts:
            papers = client.papers_for_concept(concept)
            if papers:
                titles = ", ".join(p["title"] for p in papers[:5])
                facts.append(f"Papers mentioning '{concept}': {titles}")

        client.close()
        return facts

    except Exception as exc:
        logger.warning(f"Knowledge graph query failed, degrading to hybrid-retrieval-only: {exc}")
        return []


if __name__ == "__main__":
    facts = safe_graph_query("Which papers discuss Agent Skills?")
    for f in facts:
        print(f)
