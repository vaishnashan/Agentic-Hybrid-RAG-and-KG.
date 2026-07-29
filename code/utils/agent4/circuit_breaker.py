"""
Circuit breakers around each external call (LLM, Neo4j, vector DB) so a SUSTAINED
outage fails fast instead of every request paying the full retry-and-timeout cost
against something that's actually down.

Distinct from retry_policy.py: retries handle a single transient blip; a circuit
breaker handles "this dependency has failed repeatedly, stop hammering it" — after
fail_max consecutive failures, the breaker "opens" and short-circuits immediately
for reset_timeout seconds before allowing a test request through again.
"""
import logging

import pybreaker

logger = logging.getLogger("circuit_breaker")


class LoggingListener(pybreaker.CircuitBreakerListener):
    def state_change(self, cb, old_state, new_state):
        logger.warning(f"Circuit breaker '{cb.name}' state change: {old_state} -> {new_state}")


def make_breaker(name: str, fail_max: int = 3, reset_timeout: int = 30) -> pybreaker.CircuitBreaker:
    return pybreaker.CircuitBreaker(
        fail_max=fail_max,
        reset_timeout=reset_timeout,
        name=name,
        listeners=[LoggingListener()],
    )


# One breaker per external dependency. Tune fail_max/reset_timeout per component —
# the LLM breaker resets faster since Groq rate limits are usually short-lived;
# Neo4j gets a longer reset since a real outage there tends to last longer.
llm_breaker = make_breaker("groq_llm", fail_max=3, reset_timeout=20)
neo4j_breaker = make_breaker("neo4j", fail_max=3, reset_timeout=30)
vector_db_breaker = make_breaker("chroma", fail_max=3, reset_timeout=30)


if __name__ == "__main__":
    # Demonstrate the breaker actually opening after repeated failures
    test_breaker = make_breaker("demo", fail_max=2, reset_timeout=5)

    def always_fails():
        raise ConnectionError("simulated dependency down")

    for i in range(4):
        try:
            test_breaker.call(always_fails)
        except pybreaker.CircuitBreakerError as exc:
            print(f"Attempt {i+1}: breaker OPEN, failed fast without calling dependency: {exc}")
        except ConnectionError as exc:
            print(f"Attempt {i+1}: breaker CLOSED, called dependency, it failed: {exc}")
