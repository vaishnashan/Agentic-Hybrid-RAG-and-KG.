"""
Self-critic node: checks the reasoner's draft answer for confidence/faithfulness and
decides whether to retry with a different retrieval strategy.

Confidence scoring is a deliberate, documented formula built only from signals the
critic actually has access to (ReasonerOutput + strategy + retry count) — not three
hardcoded buckets. See _compute_confidence() for the exact weighting.

Kept heuristic for now (Day 8-9 scope) — an LLM-as-judge upgrade is explicitly
planned for Week 2, Day 11-14, once structured-output enforcement is in place
everywhere. Doing that now would be building ahead of the plan's own sequencing.
"""
from utils.agent4.schemas import CriticVerdict, ReasonerOutput

LOW_CONFIDENCE_SIGNALS = [
    "placeholder", "insufficient", "cannot determine", "not clear from the context",
    "llm call failed",
]

# Tunable weights for _compute_confidence — kept as named constants (not inline
# magic numbers) so the scoring logic is auditable and easy to adjust in one place.
BASE_CONFIDENT = 0.6
BASE_WEAK = 0.2
CONTEXT_BONUS_PER_CHUNK = 0.05
CONTEXT_BONUS_MAX_CHUNKS = 5       # bonus caps out at 5+ used chunks (+0.25 max)
GRAPH_BONUS_PER_FACT = 0.03
GRAPH_BONUS_MAX_FACTS = 3          # bonus caps out at 3+ used graph facts (+0.09 max)
RETRY_CONFIDENCE_CAP = 0.65        # a best-effort answer after a retry is never "high confidence"
CONFIDENT_THRESHOLD = 0.7          # matches the caching gate in graph_definition.py


def _compute_confidence(
    reasoner_output: ReasonerOutput, has_weak_signal: bool, retries_so_far: int
) -> float:
    """
    Computes a confidence score in [0, 0.99] from three signals:
      1. Whether the draft answer itself shows a low-confidence phrase, or used no
         context at all (dominant factor — sets the base).
      2. How much retrieved context was actually used (more grounding = more trust).
      3. How many corroborating knowledge-graph facts were used.
    A second-attempt (retried) answer is capped below the "confident" threshold,
    since giving it a high score would mean pretending a best-effort fallback is as
    trustworthy as a first-try success.
    """
    base = BASE_WEAK if has_weak_signal else BASE_CONFIDENT

    n_context = len(reasoner_output.used_context_ids)
    context_bonus = min(n_context, CONTEXT_BONUS_MAX_CHUNKS) * CONTEXT_BONUS_PER_CHUNK

    n_graph = len(reasoner_output.used_graph_facts)
    graph_bonus = min(n_graph, GRAPH_BONUS_MAX_FACTS) * GRAPH_BONUS_PER_FACT

    score = base + context_bonus + graph_bonus

    if retries_so_far >= 1:
        score = min(score, RETRY_CONFIDENCE_CAP)

    return round(min(max(score, 0.0), 0.99), 2)


def critique(reasoner_output: ReasonerOutput, current_strategy: str, retries_so_far: int) -> CriticVerdict:
    answer_lower = reasoner_output.draft_answer.lower()
    has_weak_signal = (
        any(sig in answer_lower for sig in LOW_CONFIDENCE_SIGNALS)
        or not reasoner_output.used_context_ids
    )

    confidence_score = _compute_confidence(reasoner_output, has_weak_signal, retries_so_far)

    if not has_weak_signal:
        return CriticVerdict(
            confident=confidence_score >= CONFIDENT_THRESHOLD,
            confidence_score=confidence_score,
            issues=[],
            should_retry=False,
        )

    if retries_so_far >= 1:
        return CriticVerdict(
            confident=False,
            confidence_score=confidence_score,
            issues=["Low confidence after retry; returning best-effort answer."],
            should_retry=False,
        )

    next_strategy = "hybrid_both" if current_strategy == "vector_only" else "vector_only"
    return CriticVerdict(
        confident=False,
        confidence_score=confidence_score,
        issues=["Weak or unsupported draft answer detected."],
        should_retry=True,
        retry_strategy=next_strategy,
    )


if __name__ == "__main__":
    strong = ReasonerOutput(
        draft_answer="SkillOpt is a text-space optimizer for agent skills, described in the paper.",
        used_context_ids=["c1", "c2", "c3", "c4", "c5", "c6"],
        used_graph_facts=["Papers mentioning 'SkillOpt': Paper A, Paper B"],
    )
    weak = ReasonerOutput(
        draft_answer="[LLM call failed] Context was insufficient to generate an answer.",
        used_context_ids=[],
        used_graph_facts=[],
    )

    v1 = critique(strong, "hybrid_both", retries_so_far=0)
    print(f"Strong answer, 6 chunks + 1 graph fact -> confidence={v1.confidence_score}, confident={v1.confident}")

    v2 = critique(weak, "vector_only", retries_so_far=0)
    print(f"Weak answer, first attempt -> confidence={v2.confidence_score}, should_retry={v2.should_retry}, retry_strategy={v2.retry_strategy}")

    v3 = critique(weak, "hybrid_both", retries_so_far=1)
    print(f"Weak answer, already retried -> confidence={v3.confidence_score}, should_retry={v3.should_retry}")

    assert v1.confidence_score > v2.confidence_score
    assert v3.confidence_score <= RETRY_CONFIDENCE_CAP
    print("PASS: confidence reflects context/graph support and retry status, not fixed buckets")