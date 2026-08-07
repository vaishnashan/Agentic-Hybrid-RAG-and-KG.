"""
Planner node: decides whether a question is multi-hop and, if so, decomposes it into
sub-questions. Uses Groq (same pattern as extractor.py) with a heuristic fallback if
the LLM call fails — a flaky API call degrades the plan quality, it never crashes
the request.

Raw LLM output is validated against PlannerLLMResponse (Pydantic) before use — if the
LLM returns a malformed shape (wrong types, missing fields), that fails loudly and
visibly here and falls back to the heuristic, rather than silently coercing bad data
with .get() defaults.
"""
import json
import logging
import os
import re
from typing import List, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from codebase.backend.utils.agent.schemas import PlannerOutput, SubQuestion

load_dotenv()
logger = logging.getLogger("planner")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MULTI_HOP_SIGNALS = [
    "compare", "relationship between", "how does", "and", "versus", "vs",
    "connect", "both", "which papers", "across",
]

PLANNER_PROMPT = """Decide whether this question needs information from a SINGLE source
or needs connecting information ACROSS multiple sources (multi-hop).

Return ONLY a JSON object, no preamble, no markdown fences:
{{"is_multi_hop": true/false, "sub_questions": ["..."]}}

Rules:
- If single-hop: sub_questions should be a list containing just the original question.
- If multi-hop: break it into 2-3 sub-questions that, if each were answered, would let
  you compose the full answer. Keep sub-questions concise.

Question: {question}
"""


class PlannerLLMResponse(BaseModel):
    """Strict shape the LLM's raw JSON must match — validation failure here is
    treated as a malformed-output event, not silently patched over."""
    is_multi_hop: bool
    sub_questions: List[str]


def _parse_llm_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def _plan_with_llm(question: str) -> Optional[dict]:
    if not GROQ_API_KEY:
        return None
    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": PLANNER_PROMPT.format(question=question)}],
                "temperature": 0.1,
            },
            timeout=20,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        raw_parsed = _parse_llm_json(content)

        # Validate the LLM's raw output shape explicitly — fails loudly and visibly
        # if malformed, rather than silently defaulting via .get().
        validated = PlannerLLMResponse(**raw_parsed)

        sub_questions = [s.strip() for s in validated.sub_questions if s.strip()] or [question]
        return {"is_multi_hop": validated.is_multi_hop, "sub_questions": sub_questions}

    except ValidationError as exc:
        logger.warning(f"Planner LLM returned malformed output, falling back to heuristic: {exc}")
        return None
    except Exception as exc:
        logger.warning(f"Planner LLM call failed, falling back to heuristic: {exc}")
        return None


def _plan_with_heuristic(question: str) -> dict:
    """Free, deterministic fallback — no API key or network call needed.
    Same intent as _plan_with_llm but rule-based: flags multi-hop via keyword
    signals, and always returns the original question as a sub-question so
    downstream nodes never see an empty list."""
    q_lower = question.lower()
    is_multi_hop = any(signal in q_lower for signal in MULTI_HOP_SIGNALS)
    return {"is_multi_hop": is_multi_hop, "sub_questions": [question]}

def plan(question: str) -> PlannerOutput:
    result = _plan_with_llm(question)
    if result is None:
        result = _plan_with_heuristic(question)

    sub_questions = [
        SubQuestion(text=q, requires_graph=result["is_multi_hop"])
        for q in result["sub_questions"]
    ]

    return PlannerOutput(
        original_question=question,
        sub_questions=sub_questions,
        is_multi_hop=result["is_multi_hop"],
    )


if __name__ == "__main__":
    for q in [
        "What is SkillOpt?",
        "Compare how MRKL and Gorilla connect LLMs to external tools.",
    ]:
        result = plan(q)
        print(f"Q: {q}")
        print(f"  multi_hop={result.is_multi_hop}, sub_questions={[s.text for s in result.sub_questions]}\n")