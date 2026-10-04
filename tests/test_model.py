import os
import pytest
from fastapi.testclient import TestClient
from arabic_legal_rag.config import load_config
from arabic_legal_rag.api import app
from arabic_legal_rag.pipeline import run_retrieval

client = TestClient(app)

def test_config_loading():
    """Verify that configuration loads correctly."""
    config = load_config()
    assert isinstance(config, dict)
    assert "vector_db" in config

def test_legal_retrieval():
    """Verify legal retrieval pipeline if local vector store index exists."""
    config = load_config()
    index_dir = config["vector_db"]["index_dir"]
    index_file = os.path.join(index_dir, "index.faiss")
    
    # Skip gracefully if vector store index is missing (e.g. in CI environments)
    if not os.path.exists(index_file):
        pytest.skip(f"Vector store index not found locally at {index_file}. Skipping retrieval integration test.")

    query = "ما هي احكام بطلان العقد وإعادة المتعاقدين إلى الحالة التي كانا عليها؟"
    results = run_retrieval(query)
    assert isinstance(results, list)

def test_health_check_endpoint():
    """Verify the health check endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json().get("status") == "healthy"

def test_ask_endpoint_validation():
    """Verify validation of the /ask endpoint with invalid payload."""
    response = client.post("/ask", json={})
    # Should fail validation because 'query' field is required
    assert response.status_code == 422