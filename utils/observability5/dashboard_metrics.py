"""
Monitoring dashboard (Streamlit panel) — Day 17-18: "quality trend, average
latency, and token cost over time."

Reads two on-disk sources, both written by pieces you already have:
  - data/processed/traces.jsonl        (written by observability/tracing.py)
  - utils/evaluation/reports/ragas_*.json  (written by run_ragas.py, one per run)

No new infra required to see this working locally — if you stand up Langfuse
later, this can be pointed at its query API instead, but this file works
standalone against the local JSONL log in the meantime.

Run with:
    streamlit run utils/observability/dashboard_metrics.py
"""
import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "traces.jsonl"
RAGAS_REPORTS_DIR = PROJECT_ROOT / "utils" / "evaluation" / "reports"


@st.cache_data(ttl=30)
def load_traces() -> pd.DataFrame:
    if not TRACE_LOG_PATH.exists():
        return pd.DataFrame()
    records = []
    with TRACE_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


@st.cache_data(ttl=30)
def load_ragas_history() -> pd.DataFrame:
    pattern = str(RAGAS_REPORTS_DIR / "ragas_*.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        filename = os.path.basename(path)
        # ragas_20260728T120000Z.json -> 20260728T120000Z
        ts_str = filename.replace("ragas_", "").replace(".json", "")
        try:
            ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        with open(path, "r") as f:
            scores = json.load(f)
        row = {"timestamp": ts, "run": filename}
        row.update(scores)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("timestamp")


def main():
    st.set_page_config(page_title="Agent Monitoring Dashboard", layout="wide")
    st.title("Agent Monitoring Dashboard")

    traces = load_traces()
    ragas_history = load_ragas_history()

    st.header("Quality trend (RAGAS)")
    if ragas_history.empty:
        st.info("No RAGAS reports found yet. Run `python -m utils.evaluation.run_ragas` at least twice to see a trend.")
    else:
        metric_cols = [c for c in ragas_history.columns if c not in ("timestamp", "run")]
        st.line_chart(ragas_history.set_index("timestamp")[metric_cols])
        st.dataframe(ragas_history, use_container_width=True)

    st.header("Latency")
    if traces.empty:
        st.info("No traces found yet. Run a few questions through utils.agent4.graph_definition.ask() "
                "with tracing.py wired in, then refresh.")
    else:
        request_rows = traces[traces["type"] == "request_complete"].copy()
        if not request_rows.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Requests traced", len(request_rows))
            col2.metric("Avg latency (ms)", round(request_rows["elapsed_ms"].mean(), 1))
            col3.metric("p95 latency (ms)", round(request_rows["elapsed_ms"].quantile(0.95), 1))
            st.line_chart(request_rows.set_index("datetime")["elapsed_ms"])

        span_rows = traces[traces["type"] == "span"].copy()
        if not span_rows.empty:
            st.subheader("Latency by node (avg ms)")
            st.bar_chart(span_rows.groupby("name")["elapsed_ms"].mean())

    st.header("Token cost")
    if not traces.empty:
        llm_rows = traces[traces["type"] == "llm_call"].copy()
        if not llm_rows.empty:
            col1, col2 = st.columns(2)
            col1.metric("Total tokens (all traced calls)", int(llm_rows["total_tokens"].sum()))
            col2.metric("Avg tokens / call", round(llm_rows["total_tokens"].mean(), 1))
            st.line_chart(llm_rows.set_index("datetime")["total_tokens"])
        else:
            st.info("No LLM token usage logged yet — wire tracing.RequestTrace.log_llm_call() "
                    "into reasoner.py/planner.py's Groq calls to populate this.")


if __name__ == "__main__":
    main()