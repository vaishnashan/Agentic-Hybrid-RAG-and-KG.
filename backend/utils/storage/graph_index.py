"""
Builds the knowledge graph: reads data/processed/papers.jsonl and, for each
paper, asks an LLM (Groq, GPT-OSS) to extract schema-controlled (subject,
relation, object) triples, then loads them into Neo4j Aura after validating
each one against schema.py.

Why triples instead of two flat lists: a flat "concepts" + "methods" list
can only ever express (Paper)-[:MENTIONS_CONCEPT]->(Concept) and
(Paper)-[:USES_METHOD]->(Method). It can't say "this method PROPOSES that
method" or "this paper EVALUATED_ON this dataset" — richer relationships
that make the graph useful for real multi-hop questions. Asking the LLM for
triples directly lets it express any of the relation shapes defined in
schema.py, in one call.

Why validation against schema.py, not just "trust the LLM": an LLM asked for
freeform relation names will drift — "USES", "USES_TECHNIQUE", "APPLIES",
"BASED_ON" all meaning roughly the same thing across different papers/runs.
Every triple the LLM returns is checked against schema.RELATION_SHAPES
before it's written; anything that doesn't match a known (subject_type,
relation, object_type) shape is dropped (and logged), never silently stored.

Falls back to the old keyword-matching approach (schema.CONCEPT_KEYWORDS)
for any paper where the LLM call fails or returns no valid triples — so a
flaky API call never crashes the whole build, it just degrades that one
paper's extraction to Paper-MENTIONS_CONCEPT-Concept triples only.

Requires GROQ_API_KEY in your .env. Uses openai/gpt-oss-20b by default (see
storage/config.py — override with GROQ_EXTRACTION_MODEL).

GRAPH BACKEND: writes to Neo4j Aura (free tier, no credit card required) —
see neo4j_client.py for connection details.
"""
import json
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .config import GROQ_API_KEY, GROQ_EXTRACTION_MODEL, GROQ_URL, PAPERS_PATH
from .neo4j_client import Neo4jGraphClient
from .schema import CONCEPT_KEYWORDS, NODE_LABELS, RELATION_SHAPES, SEED_CONCEPTS, is_valid_triple

# How much of full_text gets sent to the LLM per paper. Wide enough to also
# catch the intro/methods section, where dataset names and task framing
# usually show up (abstracts rarely mention them).
TEXT_WINDOW_CHARS = 6000

# Literal placeholder the LLM uses instead of guessing the paper's arxiv_id
# (it only sees the title/text, never the ID) — swapped back to the real
# arxiv_id in _validate_and_normalize_triples before anything reaches Neo4j.
THIS_PAPER = "THIS_PAPER"

_compiled_keyword_patterns = {
    concept: [re.compile(p, re.IGNORECASE) for p in patterns]
    for concept, patterns in CONCEPT_KEYWORDS.items()
}

_relation_shape_lines = "\n".join(
    f"  {relation}: " + " or ".join(f"({s} -> {o})" for s, o in pairs)
    for relation, pairs in RELATION_SHAPES.items()
    if relation != "CO_OCCURS_WITH"  # derived after the fact, never LLM-extracted
)

EXTRACTION_PROMPT = """You extract structured knowledge-graph triples from a research paper's title and text.

A triple is (subject, subject_type, relation, object, object_type).

Allowed node types: {node_labels}

Allowed relations and which node types they must connect (subject_type -> object_type):
{relation_shapes}

Preferred concept vocabulary (reuse one of these EXACTLY if it genuinely fits — do not
invent a near-duplicate of one of these, e.g. do not write "Self Evolving Skills" if
"Self-Evolution" already covers it):
{seed_concepts}

Rules:
- If the subject of a triple is this paper itself, use the exact literal string
  "{this_paper}" as the subject (not the title, not an abbreviation).
- Only produce triples whose (subject_type, relation, object_type) matches one of the
  allowed shapes above EXACTLY. Do not invent new relation names.
- Extract 4-10 triples total. Prioritize triples that are explicitly stated, not implied.
- Method/Dataset/Task names: use the paper's own name for the thing if it has one
  (e.g. "GRPO", "GSM8K", "tool-use planning"). Do not pad names with filler words.
- Leave a category empty (omit it) rather than force a low-confidence triple.

Return ONLY a JSON object with this exact shape, nothing else — no preamble, no markdown fences:
{{"triples": [{{"subject": "...", "subject_type": "...", "relation": "...", "object": "...", "object_type": "..."}}]}}

Title: {title}

Paper text:
{text}
"""


def load_papers(path: Path = PAPERS_PATH) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"No processed papers found at {path}. Run loader.py first "
            f"(python -m utils.storage.loader)."
        )
    papers = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers


def _parse_llm_json(text: str) -> dict:
    """Robust to markdown fences and stray preamble/postamble text around the JSON."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def _normalize_name(name: str) -> str:
    """Collapses stray whitespace; leaves casing alone since acronyms (GRPO) matter."""
    return re.sub(r"\s+", " ", name).strip()


def _validate_and_normalize_triples(raw_triples: List[dict], arxiv_id: str) -> List[dict]:
    """
    Drops any triple that doesn't match schema.RELATION_SHAPES, and swaps the
    THIS_PAPER placeholder for the paper's real arxiv_id. This is the one
    place that stands between "whatever the LLM said" and "what actually
    gets written to Neo4j" — nothing skips this check.
    """
    valid: List[dict] = []
    seen = set()  # de-dupe identical triples within the same paper

    for t in raw_triples:
        try:
            subject = _normalize_name(str(t["subject"]))
            subject_type = str(t["subject_type"]).strip()
            relation = str(t["relation"]).strip().upper()
            obj = _normalize_name(str(t["object"]))
            object_type = str(t["object_type"]).strip()
        except (KeyError, TypeError):
            continue  # malformed entry, skip rather than crash the whole paper

        if subject_type not in NODE_LABELS or object_type not in NODE_LABELS:
            continue
        if not is_valid_triple(subject_type, relation, object_type):
            continue
        if not subject or not obj:
            continue

        if subject == THIS_PAPER:
            subject = arxiv_id
        if obj == THIS_PAPER:
            obj = arxiv_id

        key = (subject.lower(), relation, obj.lower())
        if key in seen:
            continue
        seen.add(key)

        valid.append({
            "subject": subject, "subject_type": subject_type,
            "relation": relation, "object": obj, "object_type": object_type,
        })

    return valid


def _extract_with_llm(title: str, text: str, arxiv_id: str, retries: int = 2) -> Optional[List[dict]]:
    if not GROQ_API_KEY:
        return None

    prompt = EXTRACTION_PROMPT.format(
        node_labels=", ".join(NODE_LABELS),
        relation_shapes=_relation_shape_lines,
        seed_concepts=", ".join(SEED_CONCEPTS),
        this_paper=THIS_PAPER,
        title=title,
        text=text,
    )

    for attempt in range(retries + 1):
        try:
            response = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_EXTRACTION_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
                timeout=30,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = _parse_llm_json(content)
            raw_triples = parsed.get("triples", [])
            return _validate_and_normalize_triples(raw_triples, arxiv_id)

        except Exception as exc:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            print(f"    LLM extraction failed after {retries + 1} attempts: {exc}")
            return None


def _extract_with_keywords(title: str, text: str, arxiv_id: str) -> List[dict]:
    """Free, deterministic fallback — no API key or network call needed. Produces
    only Paper-MENTIONS_CONCEPT-Concept triples, same vocabulary as before."""
    full_text = title + " " + text
    triples = []
    for concept, patterns in _compiled_keyword_patterns.items():
        if any(p.search(full_text) for p in patterns):
            triples.append({
                "subject": arxiv_id, "subject_type": "Paper",
                "relation": "MENTIONS_CONCEPT",
                "object": concept, "object_type": "Concept",
            })
    return triples


def extract_triples(paper: dict) -> List[dict]:
    """Public entry point: LLM extraction with keyword fallback for one paper."""
    title = paper["title"]
    text = paper["full_text"][:TEXT_WINDOW_CHARS]
    arxiv_id = paper["arxiv_id"]

    triples = _extract_with_llm(title, text, arxiv_id)
    if triples:
        return triples
    return _extract_with_keywords(title, text, arxiv_id)


def build_knowledge_graph(papers: Optional[List[dict]] = None) -> Dict[str, int]:
    print("=" * 70)
    print("KNOWLEDGE GRAPH BUILDER STARTED")
    print("=" * 70)

    using_llm = bool(GROQ_API_KEY)
    print(
        f"Extraction mode: "
        f"{'LLM (Groq, ' + GROQ_EXTRACTION_MODEL + ')' if using_llm else 'keyword fallback only (no GROQ_API_KEY set)'}\n"
    )

    if papers is None:
        papers = load_papers()
    print(f"Loaded {len(papers)} papers\n")

    client = Neo4jGraphClient()
    client.verify_connectivity()
    print("Connected to Neo4j AuraDB.")
    client.ensure_constraints()
    print("Constraints ensured.\n")

    paper_concepts: Dict[str, List[str]] = {}
    llm_failures = 0
    total_triples = 0

    for index, paper in enumerate(papers, start=1):
        arxiv_id = paper["arxiv_id"]
        text = paper["full_text"][:TEXT_WINDOW_CHARS]

        triples = _extract_with_llm(paper["title"], text, arxiv_id) if using_llm else None
        if not triples:
            if using_llm:
                llm_failures += 1
            triples = _extract_with_keywords(paper["title"], text, arxiv_id)

        # Paper node itself always gets written, even if it produced zero triples.
        client.upsert_paper(
            arxiv_id=arxiv_id,
            title=paper["title"],
            category=paper["category"],
            published_month=paper.get("published_month") or "",
            url=paper.get("arxiv_url", ""),
        )

        for t in triples:
            client.link_triple(
                t["subject_type"], t["subject"], t["relation"], t["object_type"], t["object"],
            )
        total_triples += len(triples)

        paper_concepts[arxiv_id] = [
            t["object"] for t in triples
            if t["relation"] == "MENTIONS_CONCEPT" and t["object_type"] == "Concept"
        ]

        summary = ", ".join(f"{t['subject']}-[{t['relation']}]->{t['object']}" for t in triples[:4])
        print(f"[{index}/{len(papers)}] {arxiv_id} -> {len(triples)} triples ({summary}{' ...' if len(triples) > 4 else ''})")

    print("\nLinking co-occurring concepts...")
    co_occurrence_pairs = set()
    for concepts in paper_concepts.values():
        for a, b in combinations(sorted(set(concepts)), 2):
            co_occurrence_pairs.add((a, b))
    for a, b in co_occurrence_pairs:
        client.link_triple("Concept", a, "CO_OCCURS_WITH", "Concept", b)

    client.close()

    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH BUILD COMPLETE")
    print("=" * 70)
    print(f"Papers loaded          : {len(papers)}")
    print(f"Total triples written  : {total_triples}")
    print(f"Co-occurrence edges    : {len(co_occurrence_pairs)}")
    if using_llm:
        print(f"Papers that fell back to keyword extraction (LLM failed/empty): {llm_failures}")
    print("=" * 70)

    return {"papers": len(papers), "triples": total_triples, "co_occurrence_edges": len(co_occurrence_pairs)}


def run_graph_index(papers: Optional[List[dict]] = None) -> Dict[str, int]:
    """Entry point used by pipeline.py."""
    return build_knowledge_graph(papers)


if __name__ == "__main__":
    build_knowledge_graph()
