"""
Quick manual test of the knowledge graph's retrieval queries — run this AFTER
extractor.py has successfully populated your Aura instance, to confirm the graph
actually answers real questions before wiring it into the agent.

Run from your project root:
    python -m utils.knowledge_graph.test_queries
"""
from utils.knowledge_graph3.neo4j_client import Neo4jClient


def test_papers_for_concept(client, concept_name: str):
    print(f"\n=== Papers mentioning concept: '{concept_name}' ===")
    results = client.papers_for_concept(concept_name)
    if not results:
        print("  (no papers found — check the concept name matches exactly, "
              "case-sensitive, e.g. 'Agent Skills' not 'agent skills')")
    for r in results:
        print(f"  {r['arxiv_id']} | {r['category']:10} | {r['title'][:60]}")
    print(f"  Total: {len(results)} papers")


def test_papers_using_method(client, method_name: str):
    print(f"\n=== Papers using/proposing method: '{method_name}' ===")
    results = client.papers_using_method(method_name)
    if not results:
        print("  (no papers found — check the method name matches exactly, "
              "e.g. 'GRPO' not 'grpo')")
    for r in results:
        print(f"  {r['arxiv_id']} | {r['category']:10} | {r['title'][:60]}")
    print(f"  Total: {len(results)} papers")


def test_shared_concepts(client, arxiv_id_a: str, arxiv_id_b: str):
    print(f"\n=== Concepts shared between {arxiv_id_a} and {arxiv_id_b} ===")
    results = client.shared_concepts(arxiv_id_a, arxiv_id_b)
    if not results:
        print("  (no shared concepts found — these two papers may not overlap, "
              "or check the arxiv_ids are correct)")
    for r in results:
        print(f"  - {r['shared_concept']}")
    print(f"  Total: {len(results)} shared concepts")


def test_multi_hop_neighbors(client, concept_name: str):
    print(f"\n=== Multi-hop neighbors of concept: '{concept_name}' ===")
    results = client.run(
        """
        MATCH (c:Concept {name: $name})<-[:MENTIONS_CONCEPT]-(p:Paper)
              -[:MENTIONS_CONCEPT]->(other:Concept)
        WHERE other.name <> $name
        RETURN DISTINCT other.name AS related_concept, count(p) AS shared_papers
        ORDER BY shared_papers DESC
        """,
        {"name": concept_name},
    )
    for r in results:
        print(f"  {r['related_concept']} (co-occurs in {r['shared_papers']} paper(s))")


def test_all_triples_for_paper(client, arxiv_id: str):
    """Sanity check on the richer triple extraction — everything a paper links
    to, across every relation type (PROPOSES, EVALUATED_ON, SOLVES_TASK, etc),
    not just MENTIONS_CONCEPT/USES_METHOD like the old two-list extractor."""
    print(f"\n=== All triples extracted from paper: {arxiv_id} ===")
    results = client.triples_for_paper(arxiv_id)
    if not results:
        print("  (no triples found — check the arxiv_id is correct and "
              "extractor.py has been run against this paper)")
    for r in results:
        labels = "/".join(l for l in r["object_labels"])
        print(f"  -[{r['relation']}]-> {r['object_key']} ({labels})")
    print(f"  Total: {len(results)} triples")


if __name__ == "__main__":
    client = Neo4jClient()
    client.verify_connectivity()
    print("Connected to Neo4j.\n")

    # Test 1: simple lookup — "which papers are about Agent Skills?"
    test_papers_for_concept(client, "Agent Skills")

    # Test 2: a narrower concept, to check specificity
    test_papers_for_concept(client, "Ontology Construction")

    # Test 3: multi-hop — what do these two known-related papers share?
    # (MRKL and Gorilla — both tool-use papers, from your golden eval set)
    test_shared_concepts(client, "2205.00445", "2305.15334")

    # Test 4: another known pair — two self-evolving-skills papers
    test_shared_concepts(client, "2604.01687", "2605.23904")

    # Test 5: what else relates to "Agent Skills" through shared papers?
    test_multi_hop_neighbors(client, "Agent Skills")

    # Test 6: does a named method resolve to the papers that propose/use it?
    # (swap in a method name you've actually seen printed during extractor.py's run)
    test_papers_using_method(client, "GRPO")

    # Test 7: the full richer picture for one paper — every relation type, not
    # just concepts/methods. Good smoke test that PROPOSES/EVALUATED_ON/SOLVES_TASK
    # triples are actually making it into the graph.
    test_all_triples_for_paper(client, "2205.00445")

    client.close()
    print("\nDone.")