"""
Semantic cache using Upstash Redis (REST API — works over plain HTTPS, no persistent
TCP connection needed, which is why it works identically whether you're running
locally or from a deployed container).

Starts as EXACT-match caching (normalize + hash the question string) — simple,
correct, and covers the common case of someone re-asking the same question, or your
own repeated testing. TODO: upgrade to embedding-similarity caching later (hash the
query embedding's nearest existing cache key within a similarity threshold, e.g.
cosine > 0.97) once you want to catch near-duplicate phrasings too.

Cache reads/writes fail OPEN — if Upstash is unreachable, treat it as a cache miss
and let the request proceed normally, rather than failing the whole request over a
cache being down (caching is an optimization, not a dependency the system needs to
function).
"""
import hashlib
import json
import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("cache")

UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

TTL_SECONDS = 60 * 60 * 24  # 1 day


def _key_for(question: str) -> str:
    normalized = question.strip().lower()
    return "qa_cache:" + hashlib.sha256(normalized.encode()).hexdigest()


def _redis_command(command: list, timeout: int = 5) -> Optional[dict]:
    """
    Sends a single Redis command to Upstash's REST API.
    e.g. _redis_command(["GET", "some_key"]) or _redis_command(["SET", "key", "value", "EX", "3600"])
    """
    if not UPSTASH_URL or not UPSTASH_TOKEN:
        logger.warning("UPSTASH_REDIS_REST_URL / TOKEN not set — cache disabled, treating as miss.")
        return None

    try:
        response = requests.post(
            UPSTASH_URL,
            headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
            json=command,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.warning(f"Upstash call failed, treating as cache miss/failed write: {exc}")
        return None


def get_cached_answer(question: str) -> Optional[dict]:
    """Returns the cached answer dict, or None on a miss OR if Upstash is unreachable."""
    result = _redis_command(["GET", _key_for(question)])
    if result is None or result.get("result") is None:
        return None
    try:
        return json.loads(result["result"])
    except (json.JSONDecodeError, TypeError):
        return None


def set_cached_answer(question: str, answer: dict, ttl: int = TTL_SECONDS) -> bool:
    """Writes the answer to cache. Returns False (not raises) if the write fails."""
    result = _redis_command(["SET", _key_for(question), json.dumps(answer), "EX", str(ttl)])
    return result is not None and result.get("result") == "OK"


if __name__ == "__main__":
    test_question = "What is SkillOpt?"
    test_answer = {"answer": "SkillOpt is a text-space optimizer for agent skills.", "confidence": 0.85}

    print(f"Cache lookup for '{test_question}' (before write): {get_cached_answer(test_question)}")

    success = set_cached_answer(test_question, test_answer, ttl=60)
    print(f"Write succeeded: {success}")

    cached = get_cached_answer(test_question)
    print(f"Cache lookup (after write): {cached}")

    if cached:
        assert cached["answer"] == test_answer["answer"]
        print("PASS: cache round-trip works correctly")
    else:
        print("Cache miss/unreachable — check UPSTASH_REDIS_REST_URL / TOKEN in .env")
