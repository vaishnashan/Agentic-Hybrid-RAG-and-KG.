"""
Pydantic schemas for every agent node's input/output — enforced so malformed LLM
output fails fast and visibly instead of silently corrupting downstream steps.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    text: str
    requires_graph: bool = False


class PlannerOutput(BaseModel):
    original_question: str
    sub_questions: List[SubQuestion]
    is_multi_hop: bool


class RouteDecision(BaseModel):
    strategy: Literal["vector_only", "graph_only", "hybrid_both"]
    reason: str


class ReasonerOutput(BaseModel):
    draft_answer: str
    used_context_ids: List[str]
    used_graph_facts: List[str] = Field(default_factory=list)


class CriticVerdict(BaseModel):
    confident: bool
    confidence_score: float
    issues: List[str] = Field(default_factory=list)
    should_retry: bool
    retry_strategy: Optional[Literal["vector_only", "graph_only", "hybrid_both"]] = None


class FinalAnswer(BaseModel):
    question: str
    answer: str
    sources: List[str]
    confidence: float
    strategy_used: str
    retries: int = 0
