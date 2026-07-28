"""
Knowledge graph schema for the AI-agent-capability-papers corpus.

This is the single source of truth for what a "valid triple" is allowed to look
like. extractor.py checks every LLM-produced triple against RELATION_SHAPES
before it's written to Neo4j — anything that doesn't match a listed
(subject_type, relation, object_type) combination is dropped rather than stored,
so the graph can't silently fill up with relation types nobody agreed on
(e.g. "USES" vs "USES_TECHNIQUE" vs "APPLIES" all meaning the same thing).

Node labels:
  Paper(arxiv_id, title, category, published_month, url)
  Concept(name)     -- broad theme, e.g. "Agent Skills", "Self-Evolution", "Tool Use"
  Method(name)      -- specific named system/technique, e.g. "SkillOpt", "GRPO"
  Dataset(name)     -- benchmark or dataset a paper evaluates on
  Task(name)        -- a task/problem a paper addresses

Relationship types (see RELATION_SHAPES below for exactly which node-type pairs
each relation is allowed to connect):
  PROPOSES           (:Paper)-[:PROPOSES]->(:Method)
  MENTIONS_CONCEPT   (:Paper)-[:MENTIONS_CONCEPT]->(:Concept)
  USES_METHOD        (:Paper)-[:USES_METHOD]->(:Method)
  EVALUATED_ON       (:Paper)-[:EVALUATED_ON]->(:Dataset)
  SOLVES_TASK        (:Paper)-[:SOLVES_TASK]->(:Task)
  SUPPORTS           (:Method)-[:SUPPORTS]->(:Concept)
  USES               (:Method)-[:USES]->(:Method) or (:Method)-[:USES]->(:Concept)
  CO_OCCURS_WITH     (:Concept)-[:CO_OCCURS_WITH]->(:Concept)   -- derived, not LLM-extracted

TODO: if you extend the corpus later and want richer entities (Author, Venue),
add the label here, a constraint below, and a relation shape for it.
"""

NODE_LABELS = ["Paper", "Concept", "Method", "Dataset", "Task"]

# Which property uniquely identifies a node of each label. Paper is keyed by its
# arxiv_id (stable external ID); everything else is keyed by its literal name,
# since there's no better natural key for a concept/method/dataset/task.
NODE_KEY_PROPERTY = {
    "Paper": "arxiv_id",
    "Concept": "name",
    "Method": "name",
    "Dataset": "name",
    "Task": "name",
}

# The allowed (subject_type, object_type) pairs for each relation. A triple is
# only valid if its relation is a key here AND its (subject_type, object_type)
# appears in that relation's list. This is what stops the LLM's freeform output
# from turning into an unbounded set of near-duplicate relation types.
RELATION_SHAPES = {
    "PROPOSES": [("Paper", "Method")],
    "MENTIONS_CONCEPT": [("Paper", "Concept")],
    "USES_METHOD": [("Paper", "Method")],
    "EVALUATED_ON": [("Paper", "Dataset")],
    "SOLVES_TASK": [("Paper", "Task")],
    "SUPPORTS": [("Method", "Concept")],
    "USES": [("Method", "Method"), ("Method", "Concept")],
    "CO_OCCURS_WITH": [("Concept", "Concept")],
}

RELATIONSHIP_TYPES = list(RELATION_SHAPES.keys())

CONSTRAINTS_CYPHER = [
    "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT method_name IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT task_name IF NOT EXISTS FOR (t:Task) REQUIRE t.name IS UNIQUE",
]

# Seed vocabulary — passed to the LLM as PREFERRED concept names to reuse when a
# match genuinely fits, so the graph doesn't fragment into near-duplicate concept
# nodes ("self-evolution" vs "self-evolving skills" vs "skill self-evolution").
# The LLM is still allowed to introduce a new concept name if none of these fit.
SEED_CONCEPTS = [
    "Agent Skills", "Self-Evolution", "Tool Use", "Reinforcement Learning",
    "Knowledge Graph Construction", "Ontology Construction", "Multi-Agent Systems",
    "Skill Benchmarking", "Red Teaming / Safety", "Compositional Reasoning",
    "Neuro-Symbolic Systems", "Causal Modeling", "Long-Context Agents",
    "Reproducibility", "Scientific Agents", "AI Sustainability & Efficiency",
]

# Kept as a free, no-API-key fallback if the LLM call fails for a given paper
# (see extractor.py's _extract_with_keywords) — same vocabulary as the original
# keyword-only version, tested against all 30 real papers with zero unmatched papers.
CONCEPT_KEYWORDS = {
    "Agent Skills": [r"agent skill", r"\bskills?\b"],
    "Self-Evolution": [r"self-evolv", r"self evolv", r"co-evolutionary"],
    "Tool Use": [r"tool.use", r"tool.augment", r"api call", r"massive apis"],
    "Reinforcement Learning": [r"reinforcement learning", r"\brl\b"],
    "Knowledge Graph Construction": [r"knowledge graph"],
    "Ontology Construction": [r"ontolog"],
    "Multi-Agent Systems": [r"multi-agent"],
    "Skill Benchmarking": [r"benchmark"],
    "Red Teaming / Safety": [r"red team", r"\bsafety\b", r"\baudit"],
    "Compositional Reasoning": [r"compositional reasoning"],
    "Neuro-Symbolic Systems": [r"neuro-symbolic", r"neuro symbolic", r"discrete reasoning"],
    "Causal Modeling": [r"causal"],
    "Long-Context Agents": [r"long-context", r"long context"],
    "Reproducibility": [r"reproducib"],
    "Scientific Agents": [r"scientific discovery", r"lab notebook"],
    "AI Sustainability & Efficiency": [
        r"sustainab", r"energetic burden", r"inference cost", r"compute cost",
    ],
}


def is_valid_triple(subject_type: str, relation: str, object_type: str) -> bool:
    """The single validation check extractor.py runs on every LLM-produced triple."""
    allowed_pairs = RELATION_SHAPES.get(relation)
    if allowed_pairs is None:
        return False
    return (subject_type, object_type) in allowed_pairs