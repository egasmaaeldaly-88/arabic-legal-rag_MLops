from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from loguru import logger
import os

# Import our production-ready retriever and config utils
from src.arabic_legal_rag.retriever import get_legal_retriever
from src.arabic_legal_rag.utils import load_config, load_clean_corpus

# Global retriever variable and paths
retriever = None
PRODUCTION_INDEX_PATH = "data/vector_store_sz_1000_ov_100"
_CONFIG = load_config()
_CORPUS_SIZE = len(load_clean_corpus(_CONFIG["data"]["json_path"]))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    logger.info("Initializing production legal retriever...")
    if not os.path.exists(PRODUCTION_INDEX_PATH):
        raise RuntimeError(f"Vector store not found at {PRODUCTION_INDEX_PATH}")
    # Initialize retriever fetching top 3 articles
    retriever = get_legal_retriever(index_path=PRODUCTION_INDEX_PATH, k=3)
    logger.info("Retriever successfully loaded!")
    yield

app = FastAPI(
    title="Arabic Legal RAG API",
    description="API for Egyptian Civil Code Q&A and Legal Article Retrieval",
    version="1.0.0",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question string cannot be empty")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        documents_indexed=_CORPUS_SIZE
    )


@app.post("/ask", response_model=QueryResponse)
def ask_legal_question(request: QueryRequest):
    global retriever
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Question cannot be empty or consist only of whitespace."
        )

    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever is not initialized.")

    try:
        docs = retriever.invoke(clean_question)
        if not docs:
            return QueryResponse(answer="لم يتم العثور على مواد قانونية متعلقة.", sources=[])

        sources = []
        contents = []

        for doc in docs:
            art_num = str(doc.metadata.get("article_number", "")).strip()
            if art_num:
                citation = art_num if art_num.startswith("مادة") else f"مادة {art_num}"
                if citation not in sources:
                    sources.append(citation)
            contents.append(f"[{art_num}]: {doc.page_content}")

        return QueryResponse(
            answer="\n\n".join(contents),
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))