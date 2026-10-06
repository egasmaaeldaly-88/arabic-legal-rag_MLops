# ==========================================
# Stage 1: Builder Stage
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /build

# Install essential system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies into a temporary prefix directory (/install)
RUN pip install --no-cache-dir --upgrade pip

# Install CPU PyTorch
RUN pip install --no-cache-dir --prefix=/install torch --index-url https://download.pytorch.org/whl/cpu

# Install ML and API dependencies
RUN pip install --no-cache-dir --prefix=/install faiss-cpu sentence-transformers fastapi uvicorn pydantic pyyaml pytest httpx langchain-community langchain-core loguru langchain-huggingface

# Pre-download the embedding model into the build cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"


# ==========================================
# Stage 2: Runtime Stage (Clean & Secure)
# ==========================================
FROM python:3.10-slim AS runtime

# Create a non-root user (appuser) with UID 1000 for security compliance
RUN useradd --create-home --uid 1000 appuser

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy project files and assign ownership to appuser
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser configs/ configs/
COPY --chown=appuser:appuser src/ src/

# Switch to the non-root user
USER appuser

# Set PYTHONPATH to locate the arabic_legal_rag package cleanly
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Add container health check for automated monitoring
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "arabic_legal_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]