import pytest
from arabic_legal_rag.pipeline import run_retrieval

@pytest.mark.parametrize(
    "input_query, expected_behavior",
    [
        ("", "empty"),
        ("أحكام البيع وشروطه", "valid"),
        ("قانون رقم " * 50, "long_text")
    ]
)
def test_feature_engineering_edge_cases(input_query, expected_behavior):
    if expected_behavior == "empty":
        assert len(input_query.strip()) == 0
    else:
        results = run_retrieval(input_query)
        assert isinstance(results, list)

