"""
Lightweight metrics endpoint — a JSON summary of the same data
observability/dashboard_metrics.py visualizes, useful for anything that wants
programmatic access (Grafana, a status page, or just curling it yourself).
Requires auth, unlike /health, since it exposes usage volume.
"""
import json
from pathlib import Path

from fastapi import APIRouter, Depends

from utils.api7.auth import require_api_key

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRACE_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "traces.jsonl"
RAGAS_LATEST_PATH = PROJECT_ROOT / "utils" / "evaluation6" / "reports" / "ragas_latest.json"


@router.get("/metrics")
async def metrics(_: bool = Depends(require_api_key)):
    request_count = 0
    latencies = []

    if TRACE_LOG_PATH.exists():
        with TRACE_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("type") == "request_complete":
                    request_count += 1
                    latencies.append(record.get("elapsed_ms", 0))

    ragas_scores = None
    if RAGAS_LATEST_PATH.exists():
        with RAGAS_LATEST_PATH.open("r") as f:
            ragas_scores = json.load(f)

    avg_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else None

    return {
        "requests_traced": request_count,
        "avg_latency_ms": avg_latency_ms,
        "latest_ragas_scores": ragas_scores,
    }