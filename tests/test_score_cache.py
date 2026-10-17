"""score_in_batches: questions with missing judgments are retried, never cached as final."""
import types

from arabic_legal_rag import evaluate_ragas as er

NAMES = er.METRIC_NAMES
FULL = {n: 0.9 for n in NAMES}


class FakeFrame:
    """Just enough of a DataFrame for score_in_batches (no pandas needed)."""

    def __init__(self, data):
        self.data = data
        self.columns = list(data)

    def __getitem__(self, name):
        return types.SimpleNamespace(iloc=self.data[name])


def _rows(n):
    return {
        "user_input": [f"q{i}" for i in range(n)],
        "response": [f"a{i}" for i in range(n)],
        "retrieved_contexts": [[f"c{i}"] for i in range(n)],
        "reference": [f"r{i}" for i in range(n)],
    }


def _scorer(calls, overrides=None):
    """Fake scorer: every metric 0.9, except per-question overrides {user_input: {metric: None}}."""
    overrides = overrides or {}

    def scorer(sub, metric_choice, embeddings):
        calls.append(list(sub["user_input"]))
        data = {m: [] for m in NAMES}
        for q in sub["user_input"]:
            for m in NAMES:
                data[m].append(overrides.get(q, {}).get(m, 0.9))
        return FakeFrame(data)

    return scorer


def _run(rows, path, scorer, batch_size=10):
    return er.score_in_batches(rows, "all", "judge", None, path, batch_size=batch_size, scorer=scorer)


def test_complete_scores_are_cached_and_not_rescored(tmp_path):
    path, calls = tmp_path / "s.json", []
    first = _run(_rows(3), path, _scorer(calls))
    again = _run(_rows(3), path, _scorer(calls))
    assert first == again == [FULL] * 3
    assert len(calls) == 1  # the second run found everything in the cache


def test_missing_metric_is_retried_next_run(tmp_path):
    path, calls = tmp_path / "s.json", []
    rows = _rows(3)
    broken = {"q1": {"faithfulness": None}}
    first = _run(rows, path, _scorer(calls, broken))
    assert first[1]["faithfulness"] is None and first[1]["answer_relevancy"] == 0.9
    second = _run(rows, path, _scorer(calls))  # quota is back: no more failures
    assert calls[1] == ["q1"]  # only the incomplete question is scored again
    assert second == [FULL] * 3


def test_gaps_are_accepted_after_repeated_attempts(tmp_path):
    path, calls = tmp_path / "s.json", []
    rows = _rows(3)
    always = {"q1": {"faithfulness": None}}
    for _ in range(er.MAX_SCORE_ATTEMPTS):
        result = _run(rows, path, _scorer(calls, always))
    assert result[1]["faithfulness"] is None
    n = len(calls)
    _run(rows, path, _scorer(calls, always))
    assert len(calls) == n  # accepted as final: no more tokens spent on it


def test_stops_when_a_whole_batch_is_incomplete(tmp_path):
    path, calls = tmp_path / "s.json", []
    rows = _rows(6)
    quota = {f"q{i}": {"faithfulness": None} for i in range(6)}
    _run(rows, path, _scorer(calls, quota), batch_size=3)
    assert len(calls) == 1  # stopped after the first batch instead of burning through the rest
