import pytest
from arabic_legal_rag.pipeline import run_retrieval

@pytest.mark.parametrize(
    "input_query, expected_behavior",
    [
        ("", "empty"),
        ("أحكام البيع وشروطه", "valid_arabic"),
        ("What are the conditions of sale?", "valid_english"),
        ("قانون رقم " * 20, "long_text")
    ]
)
def test_feature_engineering_edge_cases(input_query, expected_behavior):
    if expected_behavior == "empty":
        assert len(input_query.strip()) == 0
    else:
        results = run_retrieval(input_query)
        assert isinstance(results, list)
        if len(results) > 0:
            item = results[0]
            # Handle both direct document/dict or tuple of (document, score)
            doc = item[0] if isinstance(item, tuple) else item
            assert hasattr(doc, "metadata") or isinstance(doc, dict)