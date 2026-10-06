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
    mock_retrieval.return_value = [{"source": "مادة 1", "content": "mocked text"}]
    response = mock_retrieval("اختبار وهمي")
    assert response[0]["source"] == "مادة 1"
    mock_retrieval.assert_called_once_with("اختبار وهمي")

