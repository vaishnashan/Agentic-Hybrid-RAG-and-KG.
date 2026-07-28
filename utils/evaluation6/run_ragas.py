"""
Runs RAGAS (faithfulness, answer relevance, context precision) against the golden
evaluation set and stores results as a versioned baseline for regression_check.py.

FIX vs the original draft: RAGAS's "contexts" field must be the actual retrieved
chunk texts the reasoner saw — not the answer text. Feeding it the answer instead
of the context makes "faithfulness" and "context precision" meaningless (you'd be
checking whether the answer is faithful to itself). This requires FinalAnswer to
carry the retrieved context text through, which is a small addition to schemas.py
and graph_definition.py — see the two small diffs noted at the bottom of this file.

Expects data/golden_eval_set.json shaped like:
[
  {"question": "...", "ground_truth": "...", "expected_multi_hop": false},
  ...
]
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from utils.ingestion1.loader import PROJECT_ROOT
from utils.agent4.graph_definition import ask

GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "golden_eval_set.json"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def load_golden_set():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_agent_over_golden_set(golden_set):
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden_set:
        result = ask(item["question"])
        questions.append(item["question"])
        answers.append(result.answer)
        # Real retrieved context, not the answer — see FinalAnswer.retrieved_context
        # (added in schemas.py) and graph_definition.py's node_critique, which now
        # populates it from state["context_chunks"] before building FinalAnswer.
        context_texts = result.retrieved_context or ["(no context was retrieved)"]
        contexts.append(context_texts)
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def run_ragas_eval():
    golden_set = load_golden_set()
    print(f"Loaded {len(golden_set)} golden-set questions from {GOLDEN_SET_PATH}")

    dataset = run_agent_over_golden_set(golden_set)

    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    scores = {k: float(v) for k, v in result.items()}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORTS_DIR / f"ragas_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(scores, f, indent=2)

    # Also write/overwrite a stable "latest" pointer for regression_check.py
    with open(REPORTS_DIR / "ragas_latest.json", "w") as f:
        json.dump(scores, f, indent=2)

    print(f"RAGAS scores: {scores}")
    print(f"Saved report to {report_path}")
    return scores


if __name__ == "__main__":
    run_ragas_eval()

# ---------------------------------------------------------------------------
# Two small companion diffs needed for `result.retrieved_context` to exist:
#
# schemas.py — add one field to FinalAnswer:
#
#     class FinalAnswer(BaseModel):
#         question: str
#         answer: str
#         sources: List[str]
#         confidence: float
#         strategy_used: str
#         retries: int = 0
#         retrieved_context: List[str] = Field(default_factory=list)   # <-- add this
#
# graph_definition.py — in node_critique(), when building FinalAnswer, add:
#
#     state["final_answer"] = FinalAnswer(
#         question=state["question"],
#         answer=state["draft_answer"],
#         sources=state["used_context_ids"],
#         confidence=state["confidence"],
#         strategy_used=state["strategy"],
#         retries=state["retries"],
#         retrieved_context=[c.text for c in state["context_chunks"]],  # <-- add this
#     )
# ---------------------------------------------------------------------------