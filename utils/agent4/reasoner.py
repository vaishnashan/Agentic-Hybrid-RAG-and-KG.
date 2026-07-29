"""
Reasoner node: combines retrieved chunks + graph facts into a draft answer, using
Groq (same client pattern as extractor.py/planner.py). Falls back to a clearly-
labeled low-confidence placeholder if the LLM call fails — self_critic.py is
designed to catch that placeholder and trigger a retry with a different strategy.

Wrapped with retry_policy.with_retries() for transient blips and
circuit_breaker.llm_breaker for sustained Groq outages — see resilience/ for why
these are two different mechanisms.
"""
import json
import os
from typing import List

import requests
from dotenv import load_dotenv
from pybreaker import CircuitBreakerError

from utils.agent4.schemas import ReasonerOutput
from utils.retrieval2.sparse_retriever import RetrievedChunk
from utils.agent4.circuit_breaker import llm_breaker
from utils.agent4.retry_policy import with_retries

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

REASONER_PROMPT = """Answer the question using ONLY the context below. If the context
is insufficient to answer confidently, say so explicitly rather than guessing.

Question: {question}

Retrieved context:
{context}

Knowledge graph facts:
{graph_facts}

Answer:"""


def _call_llm(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    @with_retries(max_attempts=2)
    def _single_call():
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=30,  # explicit timeout — never hang indefinitely on a stuck request
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    try:
        return llm_breaker.call(_single_call)
    except CircuitBreakerError as exc:
        raise RuntimeError(f"LLM circuit breaker open (Groq has failed repeatedly recently): {exc}")


COMPOSE_PROMPT = """You answered several sub-questions in order to address one original
question. Combine the sub-answers below into ONE coherent, well-organized answer to
the original question — synthesize them, don't just concatenate. If any sub-answer
indicates missing or insufficient information, reflect that honestly rather than
papering over it.

Original question: {question}

Sub-question answers:
{sub_answers}

Combined answer:"""


def compose_multi_hop_answer(question: str, sub_answers: List[dict]) -> str:
    """
    Synthesizes per-sub-question draft answers into one final answer for the
    original (multi-hop) question.

    If there's only one sub-answer (the single-hop case, or a multi-hop plan that
    degenerated to one sub-question), this is a pure pass-through — no extra LLM
    call, no added latency/cost for the common case.

    Uses the same circuit-breaker + retry-wrapped _call_llm() as reason(), and the
    same fail-soft pattern: if synthesis fails, fall back to a clearly-labeled
    concatenation of the individual sub-answers rather than crashing the request.
    """
    if len(sub_answers) == 1:
        return sub_answers[0]["answer"]

    formatted = "\n\n".join(
        f"Sub-question {i + 1}: {sa['sub_question']}\nAnswer: {sa['answer']}"
        for i, sa in enumerate(sub_answers)
    )
    prompt = COMPOSE_PROMPT.format(question=question, sub_answers=formatted)

    try:
        return _call_llm(prompt)
    except Exception as exc:
        fallback = "\n\n".join(f"- {sa['sub_question']}: {sa['answer']}" for sa in sub_answers)
        return f"[Synthesis LLM call failed: {exc}] Combined sub-answers:\n{fallback}"


def reason(question: str, context_chunks: List[RetrievedChunk], graph_facts: List[str]) -> ReasonerOutput:
    context_str = "\n---\n".join(c.text for c in context_chunks) or "(no context retrieved)"
    graph_str = "\n".join(graph_facts) or "(no graph facts retrieved)"

    prompt = REASONER_PROMPT.format(question=question, context=context_str, graph_facts=graph_str)

    try:
        draft = _call_llm(prompt)
    except Exception as exc:
        # Deliberately contains "insufficient" so self_critic.py's LOW_CONFIDENCE_SIGNALS
        # catches this and triggers a retry with a different strategy, rather than
        # silently returning a broken answer as if it were a normal one.
        draft = f"[LLM call failed: {exc}] Context was insufficient to generate an answer."

    return ReasonerOutput(
        draft_answer=draft,
        used_context_ids=[c.chunk_id for c in context_chunks],
        used_graph_facts=graph_facts,
    )


def stream_reason(question: str, context_chunks: List[RetrievedChunk], graph_facts: List[str]):
    """
    Streaming version of reason() — yields answer text incrementally as Groq generates
    it, instead of waiting for the full response. This is the backend capability a
    future UI (Week 4) plugs into to show partial progress rather than a blocking wait.

    Yields plain text chunks (strings). The caller is responsible for concatenating
    them into the full answer and building the final ReasonerOutput once the stream
    ends (see the __main__ example below).
    """
    if not GROQ_API_KEY:
        yield "[No GROQ_API_KEY set — streaming unavailable, falling back to non-streaming reason()]"
        result = reason(question, context_chunks, graph_facts)
        yield result.draft_answer
        return

    context_str = "\n---\n".join(c.text for c in context_chunks) or "(no context retrieved)"
    graph_str = "\n".join(graph_facts) or "(no graph facts retrieved)"
    prompt = REASONER_PROMPT.format(question=question, context=context_str, graph_facts=graph_str)

    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "stream": True,
            },
            timeout=30,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            payload = decoded[len("data: "):]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    except Exception as exc:
        yield f"\n[Streaming failed: {exc}]"