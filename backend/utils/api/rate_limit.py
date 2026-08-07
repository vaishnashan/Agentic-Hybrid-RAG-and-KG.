"""
Rate limiting via slowapi. Reads RATE_LIMIT from .env (defaults to "30/minute"
if unset).

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