"""BentoML service for the Arabic Legal RAG system.

Wraps the same pipeline the FastAPI app uses (retriever -> prepare_context -> generator),
so answers, citations and the LLM backend (vLLM / Groq / transformers / extractive) are
identical. Heavy, blocking work runs in worker threads so the event loop stays free for
concurrent requests (e.g. a 50-user Locust test).

Run (from the repo root, after `pip install -e . bentoml`):

    bentoml serve arabic_legal_rag.bentoml_service:ArabicLegalRAG --port 8000

Endpoints (BentoML APIs are POST; the JSON body is keyed by argument name):

    POST /ask         {"question": "...", "top_k": 3}  -> {"answer": "...", "sources": [...]}
    POST /ask_stream  {"question": "...", "top_k": 3}  -> tokens as they are generated,
                                                          then a final "Sources: ..." line
    POST /health      {}                                -> {"status": "healthy", "documents_indexed": N}

Generator selection uses the same environment variables as the API
(GENERATOR_BACKEND, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, ...).
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import bentoml
from bentoml.exceptions import InvalidArgument
from pydantic import BaseModel

MAX_TOP_K = 10
# Repo root (src/arabic_legal_rag/bentoml_service.py -> parents[2]), so the service works
# from any working directory. Override with RAG_ROOT (e.g. inside a container).
ROOT = Path(os.getenv("RAG_ROOT") or Path(__file__).resolve().parents[2])


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


async def _aiter(sync_iter: Iterator[str]) -> AsyncGenerator[str, None]:
    """Consume a blocking iterator without blocking the event loop."""
    iterator = iter(sync_iter)
    done = object()
    while True:
        item = await asyncio.to_thread(next, iterator, done)
        if item is done:
            return
        yield item


def _clean_question(question: str) -> str:
    question = (question or "").strip()
    if not question:
        raise InvalidArgument("Question cannot be empty or consist only of whitespace.")
    return question


def _clamp_top_k(top_k: int) -> int:
    return max(1, min(int(top_k), MAX_TOP_K))


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 180},
)
class ArabicLegalRAG:
    def __init__(self) -> None:
        from arabic_legal_rag.generator import build_generator
        from arabic_legal_rag.retriever import get_legal_retriever
        from arabic_legal_rag.utils import load_config

        config = load_config(str(ROOT / "configs" / "config.yaml"))
        index_dir = Path(config["vector_db"]["index_dir"])
        if not index_dir.is_absolute():
            index_dir = ROOT / index_dir

        # Loads bge-m3 and the FAISS index once per worker.
        base = get_legal_retriever(index_path=str(index_dir), k=3)
        self.vector_store = base.vectorstore
        self.generator = build_generator(config.get("generation", {}))

    # ----------------------------------------------------------------- helpers
    def _prepare(self, question: str, top_k: int):
        """Blocking: retrieve and build (items, sources, arabic). Run in a thread."""
        from arabic_legal_rag.rag import prepare_context

        retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )
        return prepare_context(retriever, question)

    # --------------------------------------------------------------------- API
    @bentoml.api(route="/health")
    def health(self) -> dict:
        return {
            "status": "healthy",
            "documents_indexed": int(self.vector_store.index.ntotal),
        }

    @bentoml.api
    async def ask(self, question: str, top_k: int = 3) -> AskResponse:
        """Full answer with article citations."""
        from arabic_legal_rag.rag import NOT_FOUND_AR

        question = _clean_question(question)
        items, sources, arabic = await asyncio.to_thread(
            self._prepare, question, _clamp_top_k(top_k)
        )
        if not items:
            return AskResponse(answer=NOT_FOUND_AR, sources=[])

        answer = await asyncio.to_thread(self.generator.generate, question, items, arabic)
        return AskResponse(answer=answer, sources=sources)

    @bentoml.api
    async def ask_stream(self, question: str, top_k: int = 3) -> AsyncGenerator[str, None]:
        """Stream the answer token by token, then the cited articles."""
        from arabic_legal_rag.rag import NOT_FOUND_AR

        question = _clean_question(question)
        items, sources, arabic = await asyncio.to_thread(
            self._prepare, question, _clamp_top_k(top_k)
        )
        if not items:
            yield NOT_FOUND_AR
            return

        async for token in _aiter(self.generator.stream(question, items, arabic)):
            yield token
        yield "\n\nSources: " + ", ".join(sources)