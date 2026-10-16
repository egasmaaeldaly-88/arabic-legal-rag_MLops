"""Tracing helpers: no-op without Langfuse, correct span layout with a fake client."""
import contextlib
import sys
import types

import pytest

from arabic_legal_rag import tracing


@pytest.fixture(autouse=True)
def _fresh_client():
    tracing.reset()
    yield
    tracing.reset()


class FakeSpan:
    def __init__(self, lf, name, as_type, kwargs, parent, trace_id):
        self.lf, self.name, self.as_type, self.kwargs = lf, name, as_type, kwargs
        self.parent, self.trace_id = parent, trace_id
        self.updates = []

    def update(self, **kw):
        self.updates.append(kw)

    def __enter__(self):
        self.lf.stack.append(self)
        return self

    def __exit__(self, *exc):
        self.lf.stack.pop()
        return False


class FakeLangfuse:
    def __init__(self):
        self.spans, self.stack, self.scores, self.flushed = [], [], [], 0
        self.fail_scores = False

    def start_as_current_observation(self, as_type, name, **kwargs):
        parent = self.stack[-1] if self.stack else None
        trace_id = parent.trace_id if parent else f"trace{len(self.spans)}"
        span = FakeSpan(self, name, as_type, kwargs, parent, trace_id)
        self.spans.append(span)
        return span

    def create_score(self, **kw):
        if self.fail_scores:
            raise RuntimeError("langfuse down")
        self.scores.append(kw)

    def flush(self):
        self.flushed += 1


@pytest.fixture
def fake(monkeypatch):
    lf = FakeLangfuse()
    lf.propagated = []
    module = types.ModuleType("langfuse")
    module.get_client = lambda: lf

    @contextlib.contextmanager
    def propagate_attributes(**kw):
        lf.propagated.append(kw)
        yield

    module.propagate_attributes = propagate_attributes
    monkeypatch.setitem(sys.modules, "langfuse", module)
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    return lf


def test_everything_is_a_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert not tracing.enabled() and tracing.get_langfuse() is None
    with tracing.span("ask", input="q") as s:
        tracing.safe_update(s, output="a")
        assert tracing.trace_id_of(s) is None
    assert tracing.record_ask_trace("q", [], "a", model="m", endpoint="/ask") is None
    tracing.score_trace("t", "faithfulness", 0.9)
    tracing.flush()


def test_can_be_forced_off(fake, monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "0")
    assert tracing.get_langfuse() is None


def test_missing_package_disables_tracing(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setitem(sys.modules, "langfuse", None)  # import -> ImportError
    assert tracing.get_langfuse() is None
    with tracing.span("ask") as s:
        assert tracing.trace_id_of(s) is None


def test_spans_nest_under_the_root(fake):
    with tracing.span("ask", tags=["ask"], input="q"):
        with tracing.span("retrieve") as r:
            tracing.safe_update(r, output="docs")
        with tracing.span("generate", as_type="generation", model="m"):
            pass
    names = [(s.name, s.as_type, s.parent.name if s.parent else None) for s in fake.spans]
    assert names == [
        ("ask", "span", None),
        ("retrieve", "span", "ask"),
        ("generate", "generation", "ask"),
    ]
    assert fake.spans[1].updates == [{"output": "docs"}]
    assert fake.propagated == [{"tags": ["ask"]}]  # tags go through propagate_attributes
    assert fake.spans[2].kwargs["model"] == "m"


def test_record_ask_trace_builds_full_trace(fake):
    trace_id = tracing.record_ask_trace(
        "سؤال", ["مادة 1"], "إجابة", model="m", endpoint="/ask/stream",
        timings_ms={"retrieve": 5},
    )
    assert trace_id == fake.spans[0].trace_id
    assert [s.name for s in fake.spans] == ["ask", "retrieve", "generate"]
    assert fake.spans[0].updates[-1] == {"output": {"answer": "إجابة", "sources": ["مادة 1"]}}
    assert fake.spans[0].kwargs["metadata"]["timings_ms"] == {"retrieve": 5}


def test_score_trace_is_idempotent_and_ignores_missing_values(fake):
    tracing.score_trace("t1", "faithfulness", 0.83, comment="judge")
    tracing.score_trace(None, "faithfulness", 0.5)
    tracing.score_trace("t2", "faithfulness", None)
    assert len(fake.scores) == 1
    sent = fake.scores[0]
    assert sent["trace_id"] == "t1" and sent["value"] == 0.83
    assert sent["score_id"] == "t1-faithfulness" and sent["data_type"] == "NUMERIC"


def test_langfuse_errors_never_break_the_caller(fake):
    fake.fail_scores = True
    tracing.score_trace("t", "faithfulness", 0.9)  # must not raise

    class Broken(FakeLangfuse):
        def start_as_current_observation(self, *a, **k):
            raise RuntimeError("boom")

    tracing._client = Broken()
    with tracing.span("ask") as s:  # falls back to a no-op span
        assert tracing.trace_id_of(s) is None


def test_flush_calls_client(fake):
    tracing.flush()
    assert fake.flushed == 1


def test_tags_are_skipped_when_sdk_lacks_propagate_attributes(fake, monkeypatch):
    monkeypatch.delattr(sys.modules["langfuse"], "propagate_attributes")
    with tracing.span("ask", tags=["ask"], input="q") as s:
        assert tracing.trace_id_of(s) is not None  # span still created, tags just skipped
    assert [x.name for x in fake.spans] == ["ask"]
