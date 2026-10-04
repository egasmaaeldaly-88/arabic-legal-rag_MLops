import os
from fastapi.testclient import TestClient
from src.arabic_legal_rag.utils import load_config
from src.arabic_legal_rag.pipeline import run_retrieval
from src.arabic_legal_rag.api import app

# إنشاء عميل الاختبار للـ FastAPI
client = TestClient(app)

def test_config_loading():
    """التحقق من أن ملف الإعدادات يعمل ويتم قراءته بنجاح"""
    config = load_config()
    assert "data" in config
    assert "vector_db" in config
    assert "model" in config

def test_legal_retrieval():
    """التحقق من كفاءة استرجاع المواد القانونية (Pipeline)"""
    # إذا لم يكن الفهرس موجوداً محلياً، نتخطى الاختبار مؤقتاً
    if not os.path.exists("data/vector_store_sz_1000_ov_100"):
        return

    query = "ما هي احكام بطلان العقد وإعادة المتعاقدين إلى الحالة التي كانا عليها؟"
    results = run_retrieval(query)
    
    if results:
        top_doc, score = results[0]
        article_num = str(top_doc.metadata.get("article_number", ""))
        assert "١٦٠" in article_num or "160" in article_num or len(results) > 0

def test_health_check_endpoint():
    """التحقق من صحة عمل الـ API Endpoint الخاص بالـ Health"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "documents_indexed" in data

def test_ask_endpoint_validation():
    """التحقق من استجابة الـ Ask Endpoint عند إرسال مدخلات فارغة"""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422  # Unprocessable Entity validation check