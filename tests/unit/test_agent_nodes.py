"""Unit tests for individual agent nodes (planner, router, self-critic) in isolation."""
from utils.agent.planner import plan
from utils.agent.router import route
from utils.agent.self_critic import critique
from utils.agent.schemas import ReasonerOutput


def test_planner_flags_multi_hop_question():
    result = plan("Compare transformers and RNNs for sequence modeling")
    assert result.is_multi_hop is True


def test_planner_flags_simple_lookup_question():
    result = plan("What is a transformer?")
    assert result.is_multi_hop is False


def test_router_selects_hybrid_for_relational_question():
    decision = route("What is the relationship between BERT and GPT?", requires_graph_hint=True)
    assert decision.strategy == "hybrid_both"


def test_router_selects_vector_only_for_simple_question():
    decision = route("What is few-shot learning?")
    assert decision.strategy == "vector_only"


def test_self_critic_flags_weak_answer_for_retry():
    weak = ReasonerOutput(draft_answer="The context is insufficient to answer.", used_context_ids=[])
    verdict = critique(weak, current_strategy="vector_only", retries_so_far=0)
    assert verdict.should_retry is True
    assert verdict.retry_strategy == "hybrid_both"


def test_self_critic_stops_retrying_after_one_attempt():
    weak = ReasonerOutput(draft_answer="cannot determine from context", used_context_ids=[])
    verdict = critique(weak, current_strategy="hybrid_both", retries_so_far=1)
    assert verdict.should_retry is False
