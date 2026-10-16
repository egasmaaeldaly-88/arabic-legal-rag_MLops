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
import time
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import bentoml
from bentoml.exceptions import InvalidArgument
from pydantic import BaseModel

from arabic_legal_rag import tracing

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
        k = _clamp_top_k(top_k)
        # One Langfuse trace per request (no-op if Langfuse is off). asyncio.to_thread copies
        # the context, so the worker-thread work still nests under these spans.
        with tracing.span(
            "ask",
            tags=["bentoml", "ask"],
            input={"question": question, "top_k": k},
            metadata={"endpoint": "bentoml:/ask"},
        ) as root:
            with tracing.span("retrieve", input={"question": question, "top_k": k}) as r:
                items, sources, arabic = await asyncio.to_thread(self._prepare, question, k)
                tracing.safe_update(r, output={"sources": sources, "documents": len(items)})

            if not items:
                tracing.safe_update(root, output={"answer": NOT_FOUND_AR, "sources": []})
                return AskResponse(answer=NOT_FOUND_AR, sources=[])

            with tracing.span(
                "generate",
                as_type="generation",
                model=self.generator.model_name,
                input={"question": question, "contexts": items},
                metadata={"backend": self.generator.name},
            ) as g:
                answer = await asyncio.to_thread(self.generator.generate, question, items, arabic)
                tracing.safe_update(g, output=answer)

            tracing.safe_update(root, output={"answer": answer, "sources": sources})
            return AskResponse(answer=answer, sources=sources)

    @bentoml.api
    async def ask_stream(self, question: str, top_k: int = 3) -> AsyncGenerator[str, None]:
        """Stream the answer token by token, then the cited articles."""
        from arabic_legal_rag.rag import NOT_FOUND_AR

        question = _clean_question(question)
        t0 = time.perf_counter()
        items, sources, arabic = await asyncio.to_thread(
            self._prepare, question, _clamp_top_k(top_k)
        )
        retrieve_ms = round((time.perf_counter() - t0) * 1000)
        if not items:
            tracing.record_ask_trace(
                question, [], NOT_FOUND_AR, model=self.generator.model_name,
                endpoint="bentoml:/ask_stream", timings_ms={"retrieve": retrieve_ms},
                tags=["bentoml", "ask_stream"],
            )
            yield NOT_FOUND_AR
            return

        parts: list[str] = []
        first_token_ms = None
        t_gen = time.perf_counter()
        async for token in _aiter(self.generator.stream(question, items, arabic)):
            if first_token_ms is None:
                first_token_ms = round((time.perf_counter() - t_gen) * 1000)
            parts.append(token)
            yield token
        # A span can't stay open across yields, so the trace is recorded once streaming ends.
        tracing.record_ask_trace(
            question, sources, "".join(parts), model=self.generator.model_name,
            endpoint="bentoml:/ask_stream",
            timings_ms={
                "retrieve": retrieve_ms,
                "time_to_first_token": first_token_ms,
                "generate_total": round((time.perf_counter() - t_gen) * 1000),
            },
            tags=["bentoml", "ask_stream"],
        )
        yield "\n\nSources: " + ", ".join(sources)
