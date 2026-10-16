import os

# Tests must never call an LLM or an external API, whatever the shell has set.
for _var in (
    "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MAX_TOKENS", "LLM_EXTRA_BODY",
    "JUDGE_MODEL", "JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_EXTRA_BODY", "GROQ_API_KEY",
):
    os.environ.pop(_var, None)
os.environ["GENERATOR_BACKEND"] = "extractive"
os.environ["LANGFUSE_ENABLED"] = "0"  # tests never send traces to Langfuse

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
    return {"question": "ما هي أحكام فسخ العقد في القانون المدني؟"}


@pytest.fixture
def sample_features_en():
    return {
        "question": "What are the provisions for contract termination under the Civil Code?"
    }