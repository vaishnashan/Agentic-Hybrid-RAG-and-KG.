"""
Rate limiting via slowapi — Day 22-23. Reuses RATE_LIMIT from .env (already
present as "30/minute" per your existing _env file).

Limits by client IP by default (get_remote_address) — fine for a public demo;
if you later add per-API-key limits, swap key_func to read X-API-Key instead.
"""
import os

from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address

load_dotenv()

RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])