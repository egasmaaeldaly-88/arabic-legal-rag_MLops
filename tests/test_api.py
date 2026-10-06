from unittest.mock import patch
from fastapi.testclient import TestClient
from arabic_legal_rag.api import app

client = TestClient(app)


@patch("arabic_legal_rag.api.retriever")  # بنعمل Mock للـ retriever المعرف عالمياً في الـ api.py
def test_predict_endpoint_with_mock(mock_retriever):
    # إعداد الـ Mock لكي يرجع كائن وهمي يحتوي على صفحة محتوى ومتاخد من langchain Document
    class MockDoc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    mock_retriever.invoke.return_value = [
        MockDoc(page_content="محتوى المادة القانونية الوهمية للاختبار", metadata={"article_number": "1"})
    ]

    response = client.post("/predict", json={"question": "ما هي شروط العقد؟"})
    
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "sources" in response.json()

def test_health_and_metadata(client):
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json().get("status") == "healthy"

    meta_res = client.get("/metadata")
    assert meta_res.status_code == 200
    assert meta_res.json().get("model_name") == "BAAI/bge-m3"

def test_api_predict_and_ask_happy_path(client, sample_features):
    res_ask = client.post("/ask", json=sample_features)
    assert res_ask.status_code == 200
    data = res_ask.json()
    assert "answer" in data
    assert "sources" in data

    res_predict = client.post("/predict", json=sample_features)
    assert res_predict.status_code == 200
    assert res_predict.json() == data

def test_api_invalid_payload_validation(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422

