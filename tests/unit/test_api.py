"""
API-level tests using FastAPI's TestClient. Auth and validation are tested without
hitting the real agent/LLM — TODO: mock src.agent.graph_definition.ask once the real
LLM call is wired up, so these stay fast and don't need live infra.
"""
from fastapi.testclient import TestClient

from utils.api7.main import app
from utils.config import settings

client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_rejects_missing_api_key():
    resp = client.post("/ask", json={"question": "What is a transformer?"})
    assert resp.status_code in (401, 422)


def test_ask_rejects_bad_api_key():
    resp = client.post(
        "/ask", json={"question": "What is a transformer?"}, headers={"x-api-key": "wrong"}
    )
    assert resp.status_code == 401


def test_ask_rejects_too_short_question():
    resp = client.post(
        "/ask", json={"question": "hi"}, headers={"x-api-key": settings.API_KEY}
    )
    assert resp.status_code == 400
