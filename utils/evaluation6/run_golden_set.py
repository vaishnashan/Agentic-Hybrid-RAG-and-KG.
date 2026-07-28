"""
Runs the full agent pipeline against every question in the golden evaluation set
(data/golden_eval_set.json) and reports results — this is the Week 2 Day 11-14
requirement: "run the full pipeline against the Week 1 golden set and fix obvious
failures."

This does NOT do automated scoring against ground_truth (that's RAGAS's job, Week 3)
— it runs each question for real, surfaces confidence/retries/strategy per question,
and flags anything that looks broken (exceptions, rejected input, very low confidence,
or an empty answer) so you can manually spot-check and fix obvious failures now,
before the more rigorous Week 3 evaluation.
"""
import json
import time
from pathlib import Path

from utils.ingestion1.loader import PROJECT_ROOT
from utils.agent4.graph_definition import ask

GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "golden_eval_set.json"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "golden_set_results.json"

LOW_CONFIDENCE_THRESHOLD = 0.5


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list:
    if not path.exists():
        raise FileNotFoundError(f"Golden eval set not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_golden_set():
    questions = load_golden_set()
    print("=" * 70)
    print(f"RUNNING GOLDEN SET: {len(questions)} questions")
    print("=" * 70)

    results = []
    flagged = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_multi_hop = item.get("expected_multi_hop", False)

        print(f"\n[{index}/{len(questions)}] {question}")
        start = time.perf_counter()

        try:
            answer = ask(question)
            elapsed = time.perf_counter() - start

            issues = []
            if answer.strategy_used == "rejected":
                issues.append("Input was rejected by guardrails")
            if not answer.answer or not answer.answer.strip():
                issues.append("Empty answer")
            if answer.confidence < LOW_CONFIDENCE_THRESHOLD:
                issues.append(f"Low confidence ({answer.confidence})")
            actual_multi_hop_strategy = answer.strategy_used in ("hybrid_both", "graph_only")
            if expected_multi_hop and not actual_multi_hop_strategy:
                issues.append(
                    f"Expected multi-hop routing but got strategy='{answer.strategy_used}'"
                )

            result_record = {
                "question": question,
                "expected_multi_hop": expected_multi_hop,
                "answer": answer.answer,
                "confidence": answer.confidence,
                "strategy_used": answer.strategy_used,
                "retries": answer.retries,
                "sources": answer.sources,
                "elapsed_seconds": round(elapsed, 2),
                "issues": issues,
            }
            results.append(result_record)

            status = "FLAGGED" if issues else "OK"
            print(f"  [{status}] confidence={answer.confidence}, strategy={answer.strategy_used}, "
                  f"retries={answer.retries}, time={elapsed:.1f}s")
            if issues:
                flagged.append(result_record)
                for issue in issues:
                    print(f"    - {issue}")

        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(f"  [CRASHED] {exc}")
            result_record = {
                "question": question,
                "expected_multi_hop": expected_multi_hop,
                "error": str(exc),
                "elapsed_seconds": round(elapsed, 2),
                "issues": [f"Pipeline crashed: {exc}"],
            }
            results.append(result_record)
            flagged.append(result_record)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("GOLDEN SET RUN COMPLETEted ")
    print("=" * 70)
    print(f"Total questions : {len(questions)}")
    print(f"Flagged/failed  : {len(flagged)}")
    print(f"Clean           : {len(questions) - len(flagged)}")
    print(f"Results saved to: {RESULTS_PATH}")

    if flagged:
        print("\nFlagged questions (worth reviewing before moving to Week 3):")
        for r in flagged:
            print(f"  - {r['question']}")
            for issue in r.get("issues", []):
                print(f"      {issue}")

    print("=" * 70)
    return results


if __name__ == "__main__":
    run_golden_set()
