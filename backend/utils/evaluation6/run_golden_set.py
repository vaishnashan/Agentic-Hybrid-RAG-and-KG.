"""
Golden-set evaluation: runs a small, hand-curated set of question -> expected-signal
pairs through the full agent pipeline and checks each answer against cheap,
deterministic expectations — keyword presence in the answer, and (optionally)
whether the planner/router made the hop/strategy decision you expect.

This is NOT semantic quality scoring (see run_ragas.py for that) — it's a fast
smoke test you can run after any prompt/code change to catch obvious regressions
("did single-hop routing just start returning hybrid_both for everything?")
before reaching for the heavier, slower RAGAS pass.

The golden set (golden_set.json) is hand-written and hand-checked, never
LLM-generated — the whole point is a human-verified ground truth to compare
against, not another model's opinion. EDIT golden_set.json to match your actual
corpus; the shipped starter set uses names mentioned in early testing (SkillOpt,
Agent Skills, MRKL, Gorilla) and needs to be replaced with real Q&A pairs from
your indexed papers.

Always bypasses the cache (calls run_pipeline() directly, not ask()) — eval should
never be shortcut by a stale cached answer from a previous run.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from codebase.backend.utils.agent4.graph_definition import run_pipeline

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def load_golden_set() -> list:
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_item(item: dict, final_answer, sub_questions: list) -> dict:
    answer_lower = final_answer.answer.lower()

    keyword_hits, keyword_misses = [], []
    for kw in item.get("expected_answer_contains", []):
        (keyword_hits if kw.lower() in answer_lower else keyword_misses).append(kw)
    keywords_passed = len(keyword_misses) == 0

    multi_hop_actual = len(sub_questions) > 1
    multi_hop_expected = item.get("expected_multi_hop")
    multi_hop_passed = (multi_hop_expected is None) or (multi_hop_actual == multi_hop_expected)

    strategy_expected = item.get("expected_strategy")
    strategy_passed = (strategy_expected is None) or (final_answer.strategy_used == strategy_expected)

    return {
        "id": item["id"],
        "question": item["question"],
        "passed": keywords_passed and multi_hop_passed and strategy_passed,
        "keywords_passed": keywords_passed,
        "keyword_hits": keyword_hits,
        "keyword_misses": keyword_misses,
        "multi_hop_passed": multi_hop_passed,
        "multi_hop_expected": multi_hop_expected,
        "multi_hop_actual": multi_hop_actual,
        "strategy_passed": strategy_passed,
        "strategy_expected": strategy_expected,
        "strategy_actual": final_answer.strategy_used,
        "answer_preview": final_answer.answer[:200],
    }


def run_golden_set() -> dict:
    golden_set = load_golden_set()
    results = []

    for item in golden_set:
        print(f"\n[GOLDEN] Running: {item['id']} — {item['question']}")
        result_state = run_pipeline(item["question"])
        final = result_state["final_answer"]
        sub_questions = result_state["sub_questions"]

        check = _check_item(item, final, sub_questions)
        print(f"[GOLDEN] {'PASS' if check['passed'] else 'FAIL'} — {item['id']}")
        if check["keyword_misses"]:
            print(f"    missing keywords: {check['keyword_misses']}")
        if not check["multi_hop_passed"]:
            print(f"    multi_hop mismatch: expected={check['multi_hop_expected']}, actual={check['multi_hop_actual']}")
        if not check["strategy_passed"]:
            print(f"    strategy mismatch: expected={check['strategy_expected']}, actual={check['strategy_actual']}")
        results.append(check)

    n_passed = sum(1 for r in results if r["passed"])
    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_total": len(results),
        "n_passed": n_passed,
        "pass_rate": round(n_passed / len(results), 3) if results else 0.0,
        "results": results,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"golden_set_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"GOLDEN SET COMPLETE — {n_passed}/{len(results)} passed ({summary['pass_rate']*100:.1f}%)")
    print(f"Report written to: {report_path}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    summary = run_golden_set()
    # Non-zero exit code on any failure — usable as a CI gate later even without
    # a separate regression_check.py (that just diffs two of these reports).
    sys.exit(0 if summary["n_passed"] == summary["n_total"] else 1)