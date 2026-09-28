from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from arabic_legal_rag.pipeline import run_retrieval
from arabic_legal_rag.utils import load_config, load_clean_corpus

app = FastAPI(
    title="Arabic Legal RAG API",
    description="API for Egyptian Civil Code Q&A and Legal Article Retrieval",
    version="0.1.0"
)

# Load metadata once at application boot time
_CONFIG = load_config()
_CORPUS_SIZE = len(load_clean_corpus(_CONFIG["data"]["json_path"]))


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
    # Instant response using cached count
    return HealthResponse(
        status="healthy",
        documents_indexed=_CORPUS_SIZE
    )


@app.post("/ask", response_model=QueryResponse)
def ask_legal_question(request: QueryRequest):
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Question cannot be empty or consist only of whitespace."
        )

    results = run_retrieval(clean_question)
    if not results:
        return QueryResponse(answer="لم يتم العثور على مواد قانونية متعلقة.", sources=[])

    sources = []
    contents = []

    for doc, score in results:
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