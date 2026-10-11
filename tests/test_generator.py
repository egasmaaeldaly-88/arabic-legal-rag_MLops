import json

import httpx
import pytest

from arabic_legal_rag.generator import (
    ExtractiveGenerator,
    OpenAICompatGenerator,
    build_generator,
    build_messages,
)

ITEMS = [{"article": "558", "text": "نص المادة"}]


def test_build_messages_arabic_contains_context_and_question():
    msgs = build_messages("ما هو عقد الإيجار؟", ITEMS, arabic=True)
    assert msgs[0]["role"] == "system" and "لا أعرف" in msgs[0]["content"]
    assert "[558]: نص المادة" in msgs[1]["content"]
    assert "ما هو عقد الإيجار؟" in msgs[1]["content"]


def test_build_messages_english_uses_english_prompt():
    msgs = build_messages("What is a lease?", ITEMS, arabic=False)
    assert "Egyptian Civil Code" in msgs[0]["content"]
    assert "Question: What is a lease?" in msgs[1]["content"]


def test_extractive_generate_and_stream():
    gen = ExtractiveGenerator()
    assert gen.generate("q", ITEMS, True) == "[558]: نص المادة"
    assert gen.generate("q", ITEMS, False) == "[Article 558]: نص المادة"
    assert "".join(gen.stream("q", ITEMS, True)).strip() == "[558]: نص المادة"


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_generate_parses_response():
    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is False and body["messages"][0]["role"] == "system"
        return httpx.Response(200, json={"choices": [{"message": {"content": " إجابة "}}]})

    gen = OpenAICompatGenerator("http://x/v1", "m", client=_client(handler))
    assert gen.generate("q", ITEMS, True) == "إجابة"


def test_openai_stream_parses_sse_and_sends_auth_header():
    sse = (
        'data: {"choices":[{"delta":{"content":"إجابة "}}]}\n\n'
        'data: {"choices":[{"delta":{}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"مولدة"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request):
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, content=sse.encode("utf-8"))

    gen = OpenAICompatGenerator("http://x/v1", "m", api_key="secret", client=_client(handler))
    assert "".join(gen.stream("q", ITEMS, True)) == "إجابة مولدة"


def test_build_generator_env_override_and_unknown_backend(monkeypatch):
    monkeypatch.setenv("GENERATOR_BACKEND", "extractive")
    assert isinstance(build_generator({"backend": "transformers"}), ExtractiveGenerator)

    monkeypatch.setenv("GENERATOR_BACKEND", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9000/v1/")
    gen = build_generator({"openai": {"model_name": "cfg-model"}})
    assert gen.base_url == "http://localhost:9000/v1" and gen.model_name == "cfg-model"

    monkeypatch.setenv("GENERATOR_BACKEND", "nope")
    with pytest.raises(ValueError):
        build_generator({})
