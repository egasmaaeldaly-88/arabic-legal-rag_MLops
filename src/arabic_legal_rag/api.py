import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from arabic_legal_rag.generator import build_generator
from arabic_legal_rag.rag import NOT_FOUND_AR, is_arabic, prepare_context  # noqa: F401
from arabic_legal_rag.retriever import get_legal_retriever
from arabic_legal_rag.utils import load_clean_corpus, load_config

# Globals populated at startup (generator is also built lazily on first use)
retriever = None
generator = None
PRODUCTION_INDEX_PATH = "data/vector_store_bgem3"
_CONFIG = load_config()
_CORPUS_SIZE = sum(
    1 for r in load_clean_corpus(_CONFIG["data"]["json_path"]) if not r.get("is_stub")
)  # repealed placeholders are not indexed


def get_generator():
    """Return the active generator, building it from config on first use."""
    global generator
    if generator is None:
        generator = build_generator(_CONFIG.get("generation", {}))
    return generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, generator
    logger.info("Initializing production legal retriever...")
    if not os.path.exists(PRODUCTION_INDEX_PATH):
        raise RuntimeError(f"Vector store not found at {PRODUCTION_INDEX_PATH}")

    retriever = get_legal_retriever(index_path=PRODUCTION_INDEX_PATH, k=3)
    logger.info("Retriever successfully loaded!")
    get_generator()  # load the LLM once at startup, not on the first request
    logger.info("Generator ready.")
    yield
    logger.info("Cleaning up resources...")
    retriever = None
    generator = None


app = FastAPI(
    title="Arabic Legal RAG API",
    description="API for Egyptian Civil Code Q&A and Legal Article Retrieval",
    version="1.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Traceback logged securely."},
    )


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question string cannot be empty")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int


class MetadataResponse(BaseModel):
    model_name: str
    embedding_framework: str
    corpus_size: int
    vector_store_path: str
    generation_backend: str = "unknown"
    generation_model: str = "unknown"


@app.get("/health", response_model=HealthResponse)
def health_check():
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retriever object not loaded in memory",
        )
    return HealthResponse(status="healthy", documents_indexed=_CORPUS_SIZE)


@app.get("/metadata", response_model=MetadataResponse)
def get_metadata():
    gen = get_generator()
    return MetadataResponse(
        model_name="BAAI/bge-m3",
        embedding_framework="LangChain + FAISS",
        corpus_size=_CORPUS_SIZE,
        vector_store_path=PRODUCTION_INDEX_PATH,
        generation_backend=gen.name,
        generation_model=gen.model_name,
    )


def _prepare(question: str):
    """Validate, retrieve, and build (items, sources, arabic) for generation."""
    clean_question = question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or consist only of whitespace.",
        )
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever is not initialized.")

    items, sources, arabic = prepare_context(retriever, clean_question)
    return clean_question, items, sources, arabic


@app.post("/ask", response_model=QueryResponse)
def ask_legal_question(request: QueryRequest):
    question, items, sources, arabic = _prepare(request.question)
    if not items:
        return QueryResponse(answer=NOT_FOUND_AR, sources=[])
    try:
        answer = get_generator().generate(question, items, arabic)
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail="Answer generation failed.")
    return QueryResponse(answer=answer, sources=sources)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
def ask_legal_question_stream(request: QueryRequest):
    """Server-Sent Events: a `sources` event, then `token` events, then `done`."""
    question, items, sources, arabic = _prepare(request.question)

    def event_stream():
        yield _sse({"type": "sources", "sources": sources})
        if not items:
            yield _sse({"type": "token", "text": NOT_FOUND_AR})
            yield _sse({"type": "done"})
            return
        try:
            for token in get_generator().stream(question, items, arabic):
                yield _sse({"type": "token", "text": token})
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}")
            yield _sse({"type": "error", "detail": "Answer generation failed."})
        yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/predict", response_model=QueryResponse)
def predict_legal_question(request: QueryRequest):
    return ask_legal_question(request)
