"""
Self-critic node: checks the reasoner's draft answer for confidence/faithfulness and
decides whether to retry with a different retrieval strategy.

Kept heuristic for now (Day 8-9 scope) — an LLM-as-judge upgrade is explicitly
planned for Week 2, Day 11-14, once structured-output enforcement is in place
everywhere. Doing that now would be building ahead of the plan's own sequencing.
"""
from utils.agent4.schemas import CriticVerdict, ReasonerOutput

LOW_CONFIDENCE_SIGNALS = [
    "placeholder", "insufficient", "cannot determine", "not clear from the context",
    "llm call failed",
]


def critique(reasoner_output: ReasonerOutput, current_strategy: str, retries_so_far: int) -> CriticVerdict:
    answer_lower = reasoner_output.draft_answer.lower()
    looks_weak = any(sig in answer_lower for sig in LOW_CONFIDENCE_SIGNALS) or not reasoner_output.used_context_ids

    if not looks_weak:
        return CriticVerdict(
            confident=True,
            confidence_score=0.85,
            issues=[],
            should_retry=False
        )
    if retries_so_far >= 1:
        return CriticVerdict(
            confident=False,
            confidence_score=0.3,
            issues=["Low confidence after retry; returning best-effort answer."],
            should_retry=False,
        )

    next_strategy = "hybrid_both" if current_strategy == "vector_only" else "vector_only"
    return CriticVerdict(
        confident=False,
        confidence_score=0.4,
        issues=["Weak or unsupported draft answer detected."],
        should_retry=True,
        retry_strategy=next_strategy,
    )
