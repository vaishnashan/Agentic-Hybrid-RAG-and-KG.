"""
Load test for the deployed API — Day 19-21: "find your realistic concurrent-user
ceiling; record p50/p95 latency under load."

Run against local API:
    locust -f locustfile.py --host=http://localhost:8000

Run against deployed API:
    locust -f locustfile.py --host=https://your-space.hf.space

Then open http://localhost:8089, set concurrent users + spawn rate, and watch
p50/p95 in Locust's web UI (or run headless with --headless -u 20 -r 2 -t 3m
for a scripted run you can paste numbers from into your evaluation report).

API_KEY must match whatever your API expects (see utils/api/auth.py) — set it
as an env var before running, don't hardcode it here.
"""
import os
import random

from locust import HttpUser, task, between

API_KEY = os.getenv("LOCUST_API_KEY", "change_me_dev_key")

# A representative mix of simple lookup + multi-hop questions, so the load test
# exercises both retrieval strategies (vector_only vs hybrid_both), not just the
# cheaper single-hop path.
SAMPLE_QUESTIONS = [
    "What is MRKL Systems and what problem does it address?",
    "What is Gorilla and what limitation of LLMs does it target?",
    "What is SkillsBench and how many tasks does it contain?",
    "Compare how Gorilla and MRKL Systems each try to connect large language models to external resources.",
    "Which papers propose mechanisms for agent skills to self-evolve or improve over time?",
    "What does SkillOpt frame as the skill in a self-evolving agent?",
]


class AgentUser(HttpUser):
    # Random think-time between requests, simulating a real user reading the
    # answer before asking the next question — not a tight request loop.
    wait_time = between(2, 6)

    def on_start(self):
        self.headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    @task(3)
    def ask_question(self):
        question = random.choice(SAMPLE_QUESTIONS)
        with self.client.post(
            "/ask",
            json={"question": question},
            headers=self.headers,
            catch_response=True,
            name="/ask",
        ) as response:
            if response.status_code == 429:
                response.success()  # rate-limit responses are expected under load, not a failure
            elif response.status_code != 200:
                response.failure(f"Unexpected status {response.status_code}: {response.text[:200]}")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")