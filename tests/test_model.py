import os
from arabic_legal_rag.utils import load_config
from arabic_legal_rag.pipeline import run_retrieval

def test_config_loading():
    """التحقق من أن ملف الإعدادات يعمل ويتم قراءته بنجاح"""
    config = load_config()
    assert "data" in config
    assert "vector_db" in config
    assert "model" in config

def test_legal_retrieval():
    """التحقق من كفاءة استرجاع المواد القانونية"""
    # يتطلب وجود ملف الـ Index في vector_db/
    if not os.path.exists("vector_db/faiss_legal_index"):
        return  # تخطي الاختيار إذا لم يتم بناء الفهرس بعد

    query = "ما هي احكام بطلان العقد وإعادة المتعاقدين إلى الحالة التي كانا عليها؟"
    results = run_retrieval(query)
    
    assert len(results) > 0
    top_doc, score = results[0]
    
    retrieved_articles = [
        str(doc.metadata.get("article_number", "")) for doc, _ in results
    ]
    assert any("١٦٠" in art or "160" in art for art in retrieved_articles)