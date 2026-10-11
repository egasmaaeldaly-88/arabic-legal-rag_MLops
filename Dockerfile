# ==========================================
# Stage 1: Builder (virtualenv, CPU-only torch)
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /build

# A virtualenv lets pip see that torch is already installed, so the later install
# does not pull multi-GB CUDA wheels. No apt-get: every dependency ships wheels.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    faiss-cpu sentence-transformers fastapi uvicorn pydantic pyyaml httpx \
    langchain-community langchain-core langchain-huggingface loguru

# Pre-download the embedding model into /build/hf
RUN HF_HOME=/build/hf \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"


# ==========================================
# Stage 2: Runtime
# ==========================================
FROM python:3.10-slim AS runtime

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface

# Embedding model baked into the image, so the first start does not re-download it
COPY --from=builder --chown=appuser:appuser /build/hf /home/appuser/.cache/huggingface

COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser configs/ configs/
COPY --chown=appuser:appuser src/ src/

# Corpus + vector store baked into the image (CI builds the index before this step)
COPY --chown=appuser:appuser data/legal_corpus_structured.json data/legal_corpus_structured.json
COPY --chown=appuser:appuser data/vector_store_bgem3/ data/vector_store_bgem3/

USER appuser

EXPOSE 8000

# start-period: loading the embedding model and the generator takes a while
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "arabic_legal_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
