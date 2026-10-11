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


class _Doc:
    def __init__(self, article="558", text_ar="نص المادة", text_en="Article text"):
        self.page_content = text_ar
        self.metadata = {"article_number": article, "text_ar": text_ar, "text_en": text_en}


class _FakeGenerator:
    name = "fake"
    model_name = "fake-llm"

    def generate(self, question, items, arabic):
        return "إجابة مولدة"

    def stream(self, question, items, arabic):
        yield from ["إجابة ", "مولدة"]


@patch("arabic_legal_rag.api.generator", _FakeGenerator())
@patch("arabic_legal_rag.api.retriever")
def test_ask_uses_generator_and_returns_article_citations(mock_retriever):
    mock_retriever.invoke.return_value = [_Doc("558"), _Doc("مادة 559")]
    res = client.post("/ask", json={"question": "ما هو عقد الإيجار؟"})
    assert res.status_code == 200
    data = res.json()
    assert data["answer"] == "إجابة مولدة"
    assert data["sources"] == ["مادة 558", "مادة 559"]


@patch("arabic_legal_rag.api.generator", _FakeGenerator())
@patch("arabic_legal_rag.api.retriever")
def test_ask_no_documents_returns_not_found(mock_retriever):
    mock_retriever.invoke.return_value = []
    res = client.post("/ask", json={"question": "سؤال بلا نتائج"})
    assert res.status_code == 200
    assert res.json()["sources"] == []


@patch("arabic_legal_rag.api.generator", _FakeGenerator())
@patch("arabic_legal_rag.api.retriever")
def test_ask_stream_emits_sources_tokens_and_done(mock_retriever):
    import json

    mock_retriever.invoke.return_value = [_Doc("558")]
    res = client.post("/ask/stream", json={"question": "ما هو عقد الإيجار؟"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: ")
    ]
    assert events[0] == {"type": "sources", "sources": ["مادة 558"]}
    assert "".join(e["text"] for e in events if e["type"] == "token") == "إجابة مولدة"
    assert events[-1] == {"type": "done"}


def test_whitespace_only_question_rejected():
    res = client.post("/ask", json={"question": "   "})
    assert res.status_code == 422

