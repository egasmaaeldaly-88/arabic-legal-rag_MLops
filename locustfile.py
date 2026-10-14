"""Locust load test for the Arabic Legal RAG BentoML service.

Run the service first (extractive backend: no API key, no rate limits):

    $env:GENERATOR_BACKEND = "extractive"      # PowerShell
    bentoml serve arabic_legal_rag.bentoml_service:ArabicLegalRAG --port 8000

Then, from the repo root (50 concurrent users, ramp 5/s, 2 minutes):

    locust -f locustfile.py --headless -u 50 -r 5 -t 2m \
        --host http://localhost:8000 \
        --html reports/locust_report.html --csv reports/locust
"""
import json
import random
from pathlib import Path

from locust import HttpUser, between, task

DATASET = Path(__file__).parent / "data" / "eval_dataset.json"

FALLBACK = [
    "ما هي المسؤولية عن العمل الشخصي؟",
    "ما هو التزام البائع بالضمان؟",
    "ما هو تعريف عقد الإيجار والتزامات المؤجر والمستأجر؟",
]


def load_questions() -> list[str]:
    try:
        with open(DATASET, encoding="utf-8") as f:
            return [row["query"] for row in json.load(f)]
    except Exception:
        return FALLBACK


QUESTIONS = load_questions()


class RagUser(HttpUser):
    wait_time = between(1, 3)

    @task(10)
    def ask(self):
        payload = {"question": random.choice(QUESTIONS), "top_k": 3}
        with self.client.post("/ask", json=payload, name="POST /ask", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            body = r.json()
            if not body.get("answer", "").strip() or not body.get("sources"):
                r.failure("empty answer or no sources")

    @task(1)
    def health(self):
        self.client.post("/health", json={}, name="POST /health")