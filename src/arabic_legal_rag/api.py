from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
import os
import re

# Import our production-ready retriever and config utils
from arabic_legal_rag.retriever import get_legal_retriever
from arabic_legal_rag.utils import load_config, load_clean_corpus

# Global retriever variable and paths
retriever = None
PRODUCTION_INDEX_PATH = "data/vector_store_bgem3"
_CONFIG = load_config()
_CORPUS_SIZE = len(load_clean_corpus(_CONFIG["data"]["json_path"]))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    logger.info("Initializing production legal retriever...")
    if not os.path.exists(PRODUCTION_INDEX_PATH):
        raise RuntimeError(f"Vector store not found at {PRODUCTION_INDEX_PATH}")
    
    retriever = get_legal_retriever(index_path=PRODUCTION_INDEX_PATH, k=3)
    logger.info("Retriever successfully loaded!")
    yield
    logger.info("Cleaning up retriever resources...")
    retriever = None

app = FastAPI(
    title="Arabic Legal RAG API",
    description="API for Egyptian Civil Code Q&A and Legal Article Retrieval",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": exc.errors()},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Traceback logged securely."},
    )

def is_arabic(text: str) -> bool:
    """التحقق مما إذا كان السؤال يحتوي على حروف عربية"""
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    return bool(arabic_pattern.search(text))

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

@app.get("/health", response_model=HealthResponse)
def health_check():
    if retriever is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Retriever object not loaded in memory")
    return HealthResponse(
        status="healthy",
        documents_indexed=_CORPUS_SIZE
    )

@app.get("/metadata", response_model=MetadataResponse)
def get_metadata():
    return MetadataResponse(
        model_name="BAAI/bge-m3",
        embedding_framework="LangChain + FAISS",
        corpus_size=_CORPUS_SIZE,
        vector_store_path=PRODUCTION_INDEX_PATH
    )

@app.post("/ask", response_model=QueryResponse)
def ask_legal_question(request: QueryRequest):
    global retriever
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty or consist only of whitespace."
        )

    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever is not initialized.")

    try:
        docs = retriever.invoke(clean_question)
        if not docs:
            return QueryResponse(answer="لم يتم العثور على مواد قانونية متعلقة.", sources=[])

        # تحديد لغة المستخدم بناءً على سؤاله
        user_wants_arabic = is_arabic(clean_question)

        sources = []
        contents = []

        for doc in docs:
            art_num = str(doc.metadata.get("article_number", "")).strip()
            if art_num:
                citation = art_num if art_num.startswith("مادة") else f"مادة {art_num}"
                if citation not in sources:
                    sources.append(citation)
            
            # اختيار اللغة المناسبة للرد بناءً على لغة السؤال
            if user_wants_arabic:
                article_text = doc.metadata.get("text_ar", doc.page_content)
                contents.append(f"[{art_num}]: {article_text}")
            else:
                article_text = doc.metadata.get("text_en", doc.page_content)
                contents.append(f"[Article {art_num}]: {article_text}")

        return QueryResponse(
            answer="\n\n".join(contents),
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict", response_model=QueryResponse)
def predict_legal_question(request: QueryRequest):
    return ask_legal_question(request)