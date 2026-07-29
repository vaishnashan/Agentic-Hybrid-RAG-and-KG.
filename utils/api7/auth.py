"""
Simple API-key auth for the public demo.

Deliberately simple (one shared key via header, not per-user keys/JWT/OAuth) —
this is a portfolio demo API, not a multi-tenant product. Reads API_KEY from .env —
make sure you rotate it off any placeholder value before deploying publicly.
"""
import os

from dotenv import load_dotenv
from fastapi import Header, HTTPException, status

load_dotenv()

API_KEY = os.getenv("API_KEY", "")


async def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """FastAPI dependency — add `Depends(require_api_key)` to any route that needs auth."""
    if not API_KEY or API_KEY == "change_me_dev_key":
        # Fails loudly rather than silently allowing unauthenticated access if
        # someone deploys without ever setting a real API_KEY.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfigured: API_KEY not set to a real value.",
        )
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
    return True