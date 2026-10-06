import pytest
from fastapi.testclient import TestClient
from arabic_legal_rag.api import app
from arabic_legal_rag.config import load_config

@pytest.fixture(scope="session")
def trained_model():
    config = load_config()
    return config

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_features():
    return {
        "question": "ما هي أحكام فسخ العقد في القانون المدني؟"
    }

@pytest.fixture
def sample_features_en():
    return {
        "question": "What are the provisions for contract termination under the Civil Code?"
    }