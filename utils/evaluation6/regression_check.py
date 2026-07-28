"""
CI gate: fails (non-zero exit) if any RAGAS metric in the latest run drops below the
configured threshold. Wired into .github/workflows/ci_cd.yml so a quality regression
blocks deployment rather than shipping silently.

FIX: the original draft imported `from src.config import settings`, but this
project has no `src/` package — everything lives under `utils/` (see your own
tree: utils/agent4, utils/evaluation, etc). That import would raise
ModuleNotFoundError immediately. Reads RAGAS_REGRESSION_THRESHOLD straight from
the environment instead, same pattern used everywhere else in this codebase
(planner.py, extractor.py, cache.py all read via os.getenv, not a settings object).
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
LATEST_REPORT_PATH = REPORTS_DIR / "ragas_latest.json"

RAGAS_REGRESSION_THRESHOLD = float(os.getenv("RAGAS_REGRESSION_THRESHOLD", "0.75"))


def check_regression() -> bool:
    if not LATEST_REPORT_PATH.exists():
        print(f"No RAGAS report found at {LATEST_REPORT_PATH}. "
              f"Run `python -m utils.evaluation.run_ragas` first.")
        return False

    with LATEST_REPORT_PATH.open("r") as f:
        scores = json.load(f)

    failures = {metric: score for metric, score in scores.items() if score < RAGAS_REGRESSION_THRESHOLD}

    if failures:
        print(f"REGRESSION DETECTED — below threshold {RAGAS_REGRESSION_THRESHOLD}: {failures}")
        return False

    print(f"All metrics >= threshold {RAGAS_REGRESSION_THRESHOLD}: {scores}")
    return True


if __name__ == "__main__":
    passed = check_regression()
    sys.exit(0 if passed else 1)