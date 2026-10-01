FROM python:3.10-slim

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install CPU PyTorch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install ML and API dependencies

RUN pip install --no-cache-dir faiss-cpu sentence-transformers fastapi uvicorn pydantic pyyaml pytest httpx langchain-community langchain-core loguru langchain-huggingface

# Pre-download embedding model into the image cache (already cached in your local build!)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Copy project files
COPY pyproject.toml .
COPY configs/ configs/
COPY data/ data/
COPY src/ src/

# Set PYTHONPATH to locate arabic_legal_rag package cleanly
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "arabic_legal_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]