FROM python:3.10-slim

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install CPU PyTorch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install ML and API dependencies (including langchain-core)
RUN pip install --no-cache-dir faiss-cpu sentence-transformers fastapi uvicorn pydantic pyyaml pytest httpx langchain-community langchain-core

# Copy project files
COPY pyproject.toml .
COPY configs/ configs/
COPY data/ data/
COPY src/ src/
COPY vector_db/ vector_db/

# Install application package
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "arabic_legal_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]