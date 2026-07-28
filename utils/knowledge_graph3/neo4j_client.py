"""
Thin wrapper around the Neo4j Python driver — connects to your Aura instance and
runs the Cypher queries the rest of the KG pipeline needs. Keep raw Cypher out of
extractor.py/graph_algorithms.py; go through here.

Reads connection details from environment variables (set these in your .env,
matching the credentials .txt file Aura gave you when you created the instance):

    NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=<the generated password>
"""
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from neo4j import GraphDatabase

from utils.knowledge_graph3.schema import CONSTRAINTS_CYPHER, NODE_KEY_PROPERTY

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


class Neo4jClient:
    def __init__(self):
        if not NEO4J_URI or not NEO4J_PASSWORD:
            raise ValueError(
                "NEO4J_URI / NEO4J_PASSWORD not set. Add them to your .env file "
                "using the credentials from your Aura instance's downloaded .txt file."
            )
        self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    def close(self):
        self._driver.close()

    def verify_connectivity(self):
        self._driver.verify_connectivity()

    def run(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        params = params or {}
        with self._driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]

    def ensure_constraints(self):
        for stmt in CONSTRAINTS_CYPHER:
            self.run(stmt)

    def upsert_paper(self, arxiv_id: str, title: str, category: str, published_month: str, url: str):
        self.run(
            """
            MERGE (p:Paper {arxiv_id: $arxiv_id})
            SET p.title = $title, p.category = $category,
                p.published_month = $published_month, p.url = $url
            """,
            {
                "arxiv_id": arxiv_id, "title": title, "category": category,
                "published_month": published_month, "url": url,
            },
        )

    def link_triple(
        self,
        subject_label: str, subject_key: str,
        relation: str,
        object_label: str, object_key: str,
    ):
        """
        Generic schema-controlled triple writer — this is what extractor.py calls
        for every (subject, relation, object) it produces, whatever the labels are
        (Paper->Method, Method->Concept, Concept->Concept, etc).

        subject_label/object_label/relation are NEVER taken from raw LLM output —
        extractor.py validates them against schema.RELATION_SHAPES first, so by the
        time they reach this f-string they're one of a small fixed whitelist, not
        arbitrary user input. Cypher can't parameterize labels or relationship
        types, which is why this needs f-strings at all; the validation upstream
        is what keeps that safe.

        The subject node is expected to already exist (e.g. Paper nodes are
        written via upsert_paper before any triples for that paper are linked).
        The object node is created on the fly via MERGE if it doesn't exist yet —
        this is what lets a Concept/Method/Dataset/Task node come into existence
        the first time any paper mentions it.
        """
        if subject_label not in NODE_KEY_PROPERTY or object_label not in NODE_KEY_PROPERTY:
            raise ValueError(f"Unknown node label: {subject_label} or {object_label}")

        subject_prop = NODE_KEY_PROPERTY[subject_label]
        object_prop = NODE_KEY_PROPERTY[object_label]

        query = f"""
        MATCH (a:{subject_label} {{{subject_prop}: $subject_key}})
        MERGE (b:{object_label} {{{object_prop}: $object_key}})
        MERGE (a)-[:{relation}]->(b)
        """
        self.run(query, {"subject_key": subject_key, "object_key": object_key})

    def triples_for_paper(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """All outgoing triples from one paper, across every relation type — a
        quick way to see everything the extractor pulled out of a given paper."""
        return self.run(
            """
            MATCH (p:Paper {arxiv_id: $arxiv_id})-[r]->(n)
            RETURN type(r) AS relation, labels(n) AS object_labels,
                   coalesce(n.name, n.arxiv_id) AS object_key
            """,
            {"arxiv_id": arxiv_id},
        )

    def papers_for_concept(self, concept_name: str) -> List[Dict[str, Any]]:
        return self.run(
            """
            MATCH (p:Paper)-[:MENTIONS_CONCEPT]->(c:Concept {name: $concept_name})
            RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.category AS category
            """,
            {"concept_name": concept_name},
        )

    def papers_using_method(self, method_name: str) -> List[Dict[str, Any]]:
        return self.run(
            """
            MATCH (p:Paper)-[:USES_METHOD|PROPOSES]->(m:Method {name: $method_name})
            RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.category AS category
            """,
            {"method_name": method_name},
        )

    def shared_concepts(self, arxiv_id_a: str, arxiv_id_b: str) -> List[Dict[str, Any]]:
        """Concepts two specific papers both mention — the core multi-hop query."""
        return self.run(
            """
            MATCH (a:Paper {arxiv_id: $a})-[:MENTIONS_CONCEPT]->(c:Concept)
                  <-[:MENTIONS_CONCEPT]-(b:Paper {arxiv_id: $b})
            RETURN c.name AS shared_concept
            """,
            {"a": arxiv_id_a, "b": arxiv_id_b},
        )

    def export_graph_for_algorithms(self) -> Dict[str, List]:
        """
        Pulls all nodes + MENTIONS_CONCEPT edges out as plain Python data, for
        graph_algorithms.py to load into NetworkX. Aura Free doesn't support the
        Neo4j GDS plugin, so algorithms run in Python instead of inside the DB.
        """
        nodes = self.run(
            """
            MATCH (n) RETURN
                CASE WHEN n:Paper THEN n.arxiv_id ELSE n.name END AS id,
                labels(n) AS labels
            """
        )
        edges = self.run(
            """
            MATCH (p:Paper)-[:MENTIONS_CONCEPT]->(c:Concept)
            RETURN p.arxiv_id AS source, c.name AS target
            """
        )
        return {"nodes": nodes, "edges": edges}

    def write_back_scores(self, scores: Dict[str, float], property_name: str):
        """Writes computed scores (e.g. PageRank) back onto Concept nodes as a property."""
        for name, score in scores.items():
            self.run(
                f"MATCH (c:Concept {{name: $name}}) SET c.{property_name} = $score",
                {"name": name, "score": score},
            )

    def write_back_communities(self, communities: Dict[str, int]):
        """Writes computed community IDs back onto Concept nodes."""
        for name, community_id in communities.items():
            self.run(
                "MATCH (c:Concept {name: $name}) SET c.community_id = $community_id",
                {"name": name, "community_id": community_id},
            )


if __name__ == "__main__":
    client = Neo4jClient()
    client.verify_connectivity()
    print("Connected to Neo4j successfully.")
    client.ensure_constraints()
    print("Constraints ensured.")
    client.close()