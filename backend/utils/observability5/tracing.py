"""
Distributed tracing for the agent pipeline, via Langfuse (self-hosted or Cloud,
open-source under MIT).

NOTE: this targets Langfuse Python SDK v4+ (OpenTelemetry-native). The client no
longer has .trace()/.span() methods — those were the v2 API. v4 uses
get_client() + start_as_current_observation() context managers instead, with
automatic parent/child nesting via OTEL context propagation. If you installed
`pip install langfuse` today, you got v4 — this file matches that.

Fails OPEN like cache.py: if Langfuse isn't configured (no keys in .env), the
package isn't installed, or it's unreachable, tracing is skipped and the agent
runs exactly as before — observability is a nice-to-have, never a hard
dependency the pipeline needs to function.

Requires in your .env:
    LANGFUSE_PUBLIC_KEY=...
    LANGFUSE_SECRET_KEY=...
    LANGFUSE_HOST=http://localhost:3000   # or https://cloud.langfuse.com
"""
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("tracing")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

# Local fallback log — every span is ALSO appended here as JSONL regardless of
# whether Langfuse is configured, so dashboard_metrics.py has something to read
# even before you've set up Langfuse. get_client() reads LANGFUSE_PUBLIC_KEY /
# LANGFUSE_SECRET_KEY / LANGFUSE_HOST from the environment itself, which
# load_dotenv() above has already populated.
LOCAL_TRACE_LOG = Path(__file__).resolve().parents[2] / "data" / "processed" / "traces.jsonl"

_langfuse_client = None
_langfuse_enabled = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

if _langfuse_enabled:
    try:
        from langfuse import get_client
        _langfuse_client = get_client()
        if not _langfuse_client.auth_check():
            logger.warning("Langfuse auth_check() failed — check your keys/host. Degrading to local-log-only.")
            _langfuse_enabled = False
    except Exception as exc:
        logger.warning(f"Langfuse client init failed, tracing degrades to local-log-only: {exc}")
        _langfuse_enabled = False


def _append_local_log(record: Dict[str, Any]) -> None:
    try:
        LOCAL_TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOCAL_TRACE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to write local trace log: {exc}")


class RequestTrace:
    """
    One of these per call to ask(). Opens a root OTEL span for the whole
    request; every .span()/.log_llm_call() call inside it becomes a nested
    child automatically via OTEL context propagation (no need to pass a
    parent reference around manually).
    """

    def __init__(self, question: str):
        self.question = question
        self.start_time = time.perf_counter()
        self._root_cm = None   # the context manager object
        self._root_span = None  # the entered span, once __enter__() has run

        if _langfuse_enabled:
            try:
                self._root_cm = _langfuse_client.start_as_current_observation(
                    as_type="span", name="agent_ask", input={"question": question},
                )
                self._root_span = self._root_cm.__enter__()
            except Exception as exc:
                logger.warning(f"Langfuse root span failed to start: {exc}")
                self._root_cm = None

    @contextmanager
    def span(self, name: str, **metadata):
        """Use as: with request_trace.span('retrieve', strategy=strategy): ..."""
        start = time.perf_counter()
        lf_span = None
        lf_cm = None
        if _langfuse_enabled:
            try:
                lf_cm = _langfuse_client.start_as_current_observation(
                    as_type="span", name=name, input=metadata or None,
                )
                lf_span = lf_cm.__enter__()
            except Exception as exc:
                logger.warning(f"Langfuse span() call failed for '{name}': {exc}")

        error: Optional[str] = None
        try:
            yield
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            if lf_span is not None:
                try:
                    lf_span.update(output={"elapsed_ms": elapsed_ms, "error": error})
                    lf_cm.__exit__(None, None, None)
                except Exception:
                    pass
            _append_local_log({
                "type": "span", "name": name, "question": self.question,
                "elapsed_ms": elapsed_ms, "error": error, "metadata": metadata,
                "timestamp": time.time(),
            })

    def log_llm_call(self, node: str, model: str, prompt_tokens: int, completion_tokens: int, latency_ms: float):
        """Call this after any Groq call to record token usage/cost-relevant data.
        Groq's response JSON includes a 'usage' block with these fields — see
        reasoner.py/planner.py's response.json()['usage'] if you want to wire this in."""
        record = {
            "type": "llm_call", "node": node, "model": model, "question": self.question,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens, "latency_ms": latency_ms,
            "timestamp": time.time(),
        }
        if _langfuse_enabled:
            try:
                with _langfuse_client.start_as_current_observation(
                    as_type="generation", name=node, model=model,
                ) as gen:
                    gen.update(usage_details={"input": prompt_tokens, "output": completion_tokens})
            except Exception as exc:
                logger.warning(f"Langfuse generation log failed for '{node}': {exc}")
        _append_local_log(record)

    def finish(self, final_answer_dict: Dict[str, Any]):
        elapsed_ms = round((time.perf_counter() - self.start_time) * 1000, 1)
        if self._root_span is not None:
            try:
                self._root_span.update(output=final_answer_dict)
                self._root_cm.__exit__(None, None, None)
                # Short-lived scripts (not a long-running server) need an explicit
                # flush or the last batch of spans may never actually get sent.
                _langfuse_client.flush()
            except Exception as exc:
                logger.warning(f"Langfuse root span finish/flush failed: {exc}")
        _append_local_log({
            "type": "request_complete", "question": self.question,
            "elapsed_ms": elapsed_ms, "final_answer": final_answer_dict,
            "timestamp": time.time(),
        })


if __name__ == "__main__":
    print(f"Langfuse enabled: {_langfuse_enabled}")
    print(f"Local trace log : {LOCAL_TRACE_LOG}")

    trace = RequestTrace("What is SkillOpt?")
    with trace.span("plan", strategy="vector_only"):
        time.sleep(0.05)
    with trace.span("retrieve", strategy="vector_only"):
        time.sleep(0.1)
    trace.log_llm_call("reason", "openai/gpt-oss-20b", prompt_tokens=450, completion_tokens=120, latency_ms=800)
    trace.finish({"answer": "...", "confidence": 0.85, "strategy_used": "vector_only", "retries": 0})
    print(f"Wrote demo spans to {LOCAL_TRACE_LOG} — check dashboard_metrics.py can read them.")
    if _langfuse_enabled:
        print("Also check your Langfuse project's Traces view — should show 'agent_ask' with 2 nested spans + 1 generation.")