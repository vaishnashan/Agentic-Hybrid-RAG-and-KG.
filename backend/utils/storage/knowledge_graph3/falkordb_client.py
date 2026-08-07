"""
Thin wrapper around the FalkorDB Python client — connects to your FalkorDB
CLOUD instance and runs the Cypher queries the rest of the KG pipeline needs.
Keep raw Cypher out of extractor.py/graph_algorithms.py; go through here.

WHY FALKORDB CLOUD INSTEAD OF NEO4J AURA: same idea as Aura (managed, no infra
to run yourself) but FalkorDB's Free Tier gives you a fully functional instance
with 100MB RAM on AWS or GCP, no credit card required to sign up. Plenty of
headroom for a corpus of a few dozen papers / a few thousand triples.

SETUP (one-time):
  1. Sign up free at https://app.falkordb.cloud/signup (no credit card needed).
  2. Create a Free Tier instance (choose AWS or GCP region).
  3. From the instance's connection panel, copy the host, port, and password.

FalkorDB speaks a subset of openCypher, so MATCH/MERGE/WHERE/WITH/UNWIND/CREATE
and path patterns all work the same as they did against Neo4j. The two real
differences from the old neo4j_client.py:
  1. No native uniqueness CONSTRAINTs (see schema.py) — uniqueness here is
     enforced purely by always MERGE-ing on the node's key property, never CREATE.
  2. Query results come back as raw rows + a header, not dict-like Records — this
     client converts them to list[dict] internally so every method below keeps
     returning plain dicts, exactly like the old Neo4j version. Callers (extractor.py,
     test_queries.py) don't need to change how they read results.

Reads connection details from environment variables (set these in your .env,
using the values from your FalkorDB Cloud instance's connection panel):

    FALKORDB_HOST=xxxxxxxx.cloud.falkordb.com
    FALKORDB_PORT=6379            # cloud instances often use a non-default port too — check your panel
    FALKORDB_PASSWORD=<the password shown in your instance's connection details>
    FALKORDB_GRAPH_NAME=papers_kg
    FALKORDB_USE_TLS=true         # FalkorDB Cloud requires TLS; local Docker doesn't
"""
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from falkordb import FalkorDB

from codebase.backend.utils.storage.knowledge_graph3.schema import (
    FALKORDB_INDEX_CYPHER,
    NODE_KEY_PROPERTY,
)

load_dotenv()

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD", "") or None
FALKORDB_GRAPH_NAME = os.getenv("FALKORDB_GRAPH_NAME", "papers_kg")
FALKORDB_USE_TLS = os.getenv("FALKORDB_USE_TLS", "true").strip().lower() in ("1", "true", "yes")


class FalkorGraphClient:
    def __init__(self):
        if not FALKORDB_HOST:
            raise ValueError(
                "FALKORDB_HOST not set. Add FALKORDB_HOST / FALKORDB_PORT / "
                "FALKORDB_PASSWORD to your .env using the connection details from "
                "your FalkorDB Cloud instance (app.falkordb.cloud)."
            )

        connection_kwargs = {
            "host": FALKORDB_HOST,
            "port": FALKORDB_PORT,
            "password": FALKORDB_PASSWORD,
        }
        # FalkorDB Cloud instances require TLS; local Docker instances typically don't.
        if FALKORDB_USE_TLS:
            connection_kwargs["ssl"] = True

        self._db = FalkorDB(**connection_kwargs)
        self._graph = self._db.select_graph(FALKORDB_GRAPH_NAME)

    def close(self):
        # falkordb-py doesn't require an explicit close of the graph handle;
        # kept as a no-op method so callers (extractor.py) don't need an if-check.
        pass

    def verify_connectivity(self):
        """Cheapest possible round-trip to confirm the FalkorDB Cloud instance is reachable."""
        self._graph.query("RETURN 1")

    @staticmethod
    def _rows_to_dicts(result) -> List[Dict[str, Any]]:
        """
        Converts a falkordb QueryResult (header + result_set rows) into the same
        list[dict] shape neo4j_client.py used to return, so every downstream
        method (and test_queries.py) keeps working unchanged.
        """
        if not result.result_set:
            return []

        # header entries can be plain column-name strings or (type, name) tuples
        # depending on falkordb-py version — normalize to plain names.
        columns = []
        for col in result.header:
            columns.append(col[1] if isinstance(col, (list, tuple)) else col)

        return [dict(zip(columns, row)) for row in result.result_set]

    def run(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        params = params or {}
        result = self._graph.query(query, params)
        return self._rows_to_dicts(result)

    def ensure_constraints(self):
        """
        Creates lookup indexes on each node label's key property. Not a
        uniqueness guarantee (see schema.py docstring) — just speeds up the
        MATCH/MERGE calls below. Safe to call repeatedly: FalkorDB raises if an
        identically-named index already exists, so duplicates are swallowed.
        """
        for stmt in FALKORDB_INDEX_CYPHER:
            try:
                self.run(stmt)
            except Exception as exc:
                if "already indexed" not in str(exc).lower():
                    raise

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
        the first time any paper mentions it. Both MERGEs (object node, and the
        relationship itself) are what stands in for Neo4j's uniqueness
        CONSTRAINTs here — as long as every write path goes through this method,
        duplicate nodes/edges can't be created.
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
        graph_algorithms.py to load into NetworkX. FalkorDB Cloud doesn't ship
        the Neo4j GDS plugin either, so algorithms still run in Python, not
        inside the DB — same pattern as before.
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
    client = FalkorGraphClient()
    client.verify_connectivity()
    print(f"Connected to FalkorDB Cloud successfully (graph: '{FALKORDB_GRAPH_NAME}').")
    client.ensure_constraints()
    print("Indexes ensured.")
    client.close()