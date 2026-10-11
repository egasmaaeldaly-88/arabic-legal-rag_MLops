"""Answer generation for the Arabic Legal RAG system.

Three interchangeable backends share one interface:

* ``extractive``  - no LLM; returns the retrieved articles (tests, CI, fallback).
* ``transformers`` - local Hugging Face model (default: Qwen2.5-1.5B-Instruct).
* ``openai``      - any OpenAI-compatible server (vLLM, Groq, OpenRouter, ...).

Each backend exposes ``generate(question, items, arabic) -> str`` and
``stream(question, items, arabic) -> Iterator[str]``. ``items`` is a list of
``{"article": str, "text": str}`` dicts built by the API from retrieved docs.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Iterator, Optional

import httpx
from loguru import logger

SYSTEM_AR = (
    "أنت مساعد قانوني متخصص في القانون المدني المصري. أجب باللغة العربية "
    "في خمس جمل على الأكثر، وبالاعتماد فقط على المواد الواردة في السياق دون "
    "إضافة أي معلومة من خارجها. اذكر رقم المادة التي استندت إليها. "
    "إذا لم تتناول المواد الواردة السؤال فقل: لا أعرف."
)
SYSTEM_EN = (
    "You are a legal assistant for the Egyptian Civil Code. Answer in English "
    "using only the legal articles given in the context. Cite the article "
    "numbers you rely on. If the answer is not in the context, say: I don't know."
)


def build_messages(question: str, items: list[dict], arabic: bool) -> list[dict]:
    """Build the chat messages (system + user) for a question and its context."""
    context = "\n\n".join(f"[{it['article']}]: {it['text']}" for it in items)
    if arabic:
        user = f"السياق:\n{context}\n\nالسؤال: {question}"
    else:
        user = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_AR if arabic else SYSTEM_EN},
        {"role": "user", "content": user},
    ]


def _format_item(item: dict, arabic: bool) -> str:
    article = item["article"]
    if arabic:
        return f"[{article}]: {item['text']}"
    return f"[Article {article}]: {item['text']}"


class ExtractiveGenerator:
    """Returns the retrieved article texts without calling any model."""

    name = "extractive"
    model_name = "none"

    def generate(self, question: str, items: list[dict], arabic: bool) -> str:
        return "\n\n".join(_format_item(it, arabic) for it in items)

    def stream(self, question: str, items: list[dict], arabic: bool) -> Iterator[str]:
        for it in items:
            yield _format_item(it, arabic) + "\n\n"


class OpenAICompatGenerator:
    """Client for OpenAI-compatible chat servers (vLLM, Groq, OpenRouter, ...)."""

    name = "openai"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        max_new_tokens: int = 256,
        timeout: float = 120.0,
        client: Optional[httpx.Client] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.client = client or httpx.Client(timeout=timeout)

    def _payload(self, question, items, arabic, stream: bool) -> dict:
        return {
            "model": self.model_name,
            "messages": build_messages(question, items, arabic),
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
            "stream": stream,
        }

    def generate(self, question: str, items: list[dict], arabic: bool) -> str:
        resp = self.client.post(
            f"{self.base_url}/chat/completions",
            json=self._payload(question, items, arabic, stream=False),
            headers=self.headers,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def stream(self, question: str, items: list[dict], arabic: bool) -> Iterator[str]:
        with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=self._payload(question, items, arabic, stream=True),
            headers=self.headers,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                token = delta.get("content")
                if token:
                    yield token


class TransformersGenerator:  # pragma: no cover - needs a model download
    """Local Hugging Face causal LM (default Qwen2.5-1.5B-Instruct)."""

    name = "transformers"

    def __init__(self, model_name: str, max_new_tokens: int = 256, dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            dtype = "float16" if self.device == "cuda" else "bfloat16"
        torch_dtype = getattr(torch, dtype)

        logger.info(f"Loading generator {model_name} on {self.device} ({dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
            .to(self.device)
            .eval()
        )
        self._lock = threading.Lock()  # one generation at a time on a single model

    def _inputs(self, question, items, arabic):
        prompt = self.tokenizer.apply_chat_template(
            build_messages(question, items, arabic),
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def generate(self, question: str, items: list[dict], arabic: bool) -> str:
        import torch

        inputs = self._inputs(question, items, arabic)
        with self._lock, torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(self, question: str, items: list[dict], arabic: bool) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        inputs = self._inputs(question, items, arabic)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            **inputs, streamer=streamer,
            max_new_tokens=self.max_new_tokens, do_sample=False,
        )
        with self._lock:
            thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
            thread.start()
            try:
                for token in streamer:
                    yield token
            finally:
                thread.join()


def build_generator(cfg: Optional[dict] = None):
    """Create the generator from config; env vars override the config file.

    GENERATOR_BACKEND, LLM_MODEL, LLM_BASE_URL and LLM_API_KEY (name set by
    ``openai.api_key_env``) take precedence over ``configs/config.yaml``.
    """
    cfg = cfg or {}
    backend = os.getenv("GENERATOR_BACKEND") or cfg.get("backend", "transformers")
    max_new_tokens = int(cfg.get("max_new_tokens", 256))

    if backend == "extractive":
        return ExtractiveGenerator()

    if backend == "openai":
        oc = cfg.get("openai", {})
        key_env = oc.get("api_key_env", "LLM_API_KEY")
        return OpenAICompatGenerator(
            base_url=os.getenv("LLM_BASE_URL") or oc.get("base_url", "http://localhost:8001/v1"),
            model_name=os.getenv("LLM_MODEL") or oc.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
            api_key=os.getenv(key_env),
            max_new_tokens=max_new_tokens,
        )

    if backend == "transformers":  # pragma: no cover
        return TransformersGenerator(
            model_name=os.getenv("LLM_MODEL") or cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
            max_new_tokens=max_new_tokens,
            dtype=cfg.get("dtype", "auto"),
        )

    raise ValueError(f"Unknown generator backend: {backend!r}")
