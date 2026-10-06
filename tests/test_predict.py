import pytest
from unittest.mock import patch
from arabic_legal_rag.pipeline import run_retrieval

def test_prediction_output_and_determinism():
    query = "ما هو الإلزام العقدي؟"
    results_first = run_retrieval(query)
    results_second = run_retrieval(query)
    assert isinstance(results_first, list)
    assert isinstance(results_second, list)
    assert len(results_first) == len(results_second)

@patch("arabic_legal_rag.pipeline.run_retrieval")
def test_prediction_with_mock(mock_retrieval):
    # محاكاة بنية النتائج الجديدة التي تتضمن الـ metadata (عربي وإنجليزي)
    mock_retrieval.return_value = [{
        "source": "مادة 1", 
        "page_content": "[Article 1]\n[AR]: نص عربي\n[EN]: English text",
        "metadata": {
            "article_number": "1",
            "text_ar": "نص عربي",
            "text_en": "English text"
        }
    }]
    
    response = mock_retrieval("اختبار وهمي")
    assert response[0]["source"] == "مادة 1"
    assert "metadata" in response[0]
    assert response[0]["metadata"]["text_ar"] == "نص عربي"
    mock_retrieval.assert_called_once_with("اختبار وهمي")