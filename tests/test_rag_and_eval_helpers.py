import json

import httpx
import pytest

from arabic_legal_rag.evaluate_ragas import (
    build_reference,
    collect_rows,
    generate_with_retry,
    load_questions,
)
from arabic_legal_rag.rag import NOT_FOUND_AR, is_arabic, prepare_context


class _Doc:
    def __init__(self, article, text_ar="نص", text_en="text"):
        self.page_content = text_ar
        self.metadata = {"article_number": article, "text_ar": text_ar, "text_en": text_en}


class _Retriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, q):
        return self.docs


class _Gen:
    name, model_name = "fake", "fake-llm"

    def __init__(self):
        self.calls = 0

    def generate(self, question, items, arabic):
        self.calls += 1
        return "إجابة"


def test_is_arabic():
    assert is_arabic("ما هو العقد؟") and not is_arabic("What is a contract?")


def test_prepare_context_arabic_dedupes_sources_and_picks_language():
    docs = [_Doc("163", "عربي", "english"), _Doc("مادة 163", "ثان", "second"), _Doc("", "بلا رقم")]
    items, sources, arabic = prepare_context(_Retriever(docs), "سؤال")
    assert arabic and sources == ["مادة 163"]
    assert [i["text"] for i in items] == ["عربي", "ثان", "بلا رقم"]
    items_en, _, arabic_en = prepare_context(_Retriever(docs), "question")
    assert not arabic_en and items_en[0]["text"] == "english"


def test_load_questions_requires_enough(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps([{"query": "a"}, {"query": "b"}]), encoding="utf-8")
    assert len(load_questions(str(p), 1)) == 1
    with pytest.raises(ValueError):
        load_questions(str(p), 5)


def test_build_reference_uses_expected_articles_in_question_language():
    texts = {"558": {"text_ar": "نص ٥٥٨", "text_en": "text 558"}}
    q = {"query": "ما هو الإيجار؟", "expected_articles": ["558", "999"]}
    assert build_reference(q, texts) == "نص ٥٥٨"
    assert build_reference({"query": "lease?", "expected_articles": ["558"]}, texts) == "text 558"


def test_collect_rows_caches_answers(tmp_path):
    gen, cache = _Gen(), tmp_path / "cache.json"
    retriever = _Retriever([_Doc("1")])
    qs = [{"query": "سؤال", "expected_articles": []}]
    rows = collect_rows(qs, retriever, gen, {}, cache)
    rows2 = collect_rows(qs, retriever, gen, {}, cache)
    assert gen.calls == 1 and rows["response"] == rows2["response"] == ["إجابة"]
    assert rows["retrieved_contexts"] == [["نص"]]


def test_collect_rows_no_documents_skips_generation(tmp_path):
    gen = _Gen()
    rows = collect_rows([{"query": "سؤال"}], _Retriever([]), gen, {}, tmp_path / "c.json")
    assert gen.calls == 0 and rows["response"] == [NOT_FOUND_AR]


def test_generate_with_retry_recovers_from_429(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    class Flaky(_Gen):
        def generate(self, *a):
            self.calls += 1
            if self.calls == 1:
                req = httpx.Request("POST", "http://x")
                raise httpx.HTTPStatusError("429", request=req, response=httpx.Response(429, request=req))
            return "تمام"

    assert generate_with_retry(Flaky(), "q", [], True) == "تمام"
