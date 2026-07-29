"""
Structured retry-with-backoff for TRANSIENT failures (a single dropped connection,
a momentary timeout) — distinct from circuit_breaker.py, which handles SUSTAINED
outages. Use retries for "probably just a blip"; let the circuit breaker handle
"this dependency is actually down".
"""
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Built-in socket-level errors AND requests' own exception types — these do NOT share
# a common base with Python's built-in ConnectionError/TimeoutError, so both must be
# listed explicitly or requests-based calls (Groq, most REST APIs) would silently
# never retry.
TRANSIENT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def with_retries(max_attempts: int = 3):
    """Decorator: retries up to max_attempts times with exponential backoff (0.5s, 1s, 2s...)."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
        reraise=True,
    )


if __name__ == "__main__":
    attempt_count = {"n": 0}

    @with_retries(max_attempts=3)
    def flaky_call():
        attempt_count["n"] += 1
        if attempt_count["n"] < 3:
            raise ConnectionError(f"simulated failure on attempt {attempt_count['n']}")
        return "success"

    result = flaky_call()
    print(f"Result: {result} (took {attempt_count['n']} attempts)")
    assert attempt_count["n"] == 3
    print("PASS: retried transient failures and succeeded on 3rd attempt")
