from unittest.mock import patch
from fastapi.testclient import TestClient
from arabic_legal_rag.api import app

client = TestClient(app)


@patch("arabic_legal_rag.api.retriever")  # Mock للـ retriever
def test_predict_endpoint_with_mock_arabic(mock_retriever):
    class MockDoc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    # محاكاة مستند يحتوي على Metadata بالعربي والإنجليزي بعد التحديثات الأخيره
    mock_retriever.invoke.return_value = [
        MockDoc(
            page_content="[Article 1]\n[AR]: نص المادة بالعربي\n[EN]: Article text in English", 
            metadata={
                "article_number": "1",
                "text_ar": "نص المادة بالعربي",
                "text_en": "Article text in English"
            }
        )
    ]

    # اختبار السؤال باللغة العربية
    response = client.post("/predict", json={"question": "ما هي شروط العقد؟"})
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    # التأكد من أن الإجابة تحتوي على النص العربي ولم تأخذ النص الإنجليزي
    assert "نص المادة بالعربي" in data["answer"]
    assert "Article text in English" not in data["answer"]


@patch("arabic_legal_rag.api.retriever")
def test_predict_endpoint_with_mock_english(mock_retriever):
    class MockDoc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    mock_retriever.invoke.return_value = [
        MockDoc(
            page_content="[Article 1]\n[AR]: نص المادة بالعربي\n[EN]: Article text in English", 
            metadata={
                "article_number": "1",
                "text_ar": "نص المادة بالعربي",
                "text_en": "Article text in English"
            }
        )
    ]

    # اختبار السؤال باللغة الإنجليزية
    response = client.post("/predict", json={"question": "What are the conditions of the contract?"})
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    # التأكد من أن الإجابة أخذت النص الإنجليزي بناءً على لغة السؤال
    assert "Article text in English" in data["answer"]
    assert "نص المادة بالعربي" not in data["answer"]


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