from fastapi.testclient import TestClient
from arabic_legal_rag.api import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health check returns 200 and document count."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["documents_indexed"] > 0


def test_ask_valid_question():
    """Verify Q&A endpoint returns an answer and clean source citations."""
    payload = {"question": "ما هي أحكام بطلان العقد؟"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_ask_empty_question_returns_422():
    """Verify empty and whitespace-only queries return HTTP 422."""
    response_empty = client.post("/ask", json={"question": ""})
    assert response_empty.status_code == 422

    response_whitespace = client.post("/ask", json={"question": "   "})
    assert response_whitespace.status_code == 422