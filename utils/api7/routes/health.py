"""
Health check endpoint — no auth required, since load balancers / uptime monitors
/ Hugging Face Spaces' own health probe need to hit this without a key.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}