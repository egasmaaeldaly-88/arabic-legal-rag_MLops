# arabic-legal-rag_MLops
# ⚖️ Arabic Legal RAG System

A production-grade Retrieval-Augmented Generation (RAG) system engineered for querying and searching the **Egyptian Civil Code** using dense vector embeddings and FAISS similarity search.

---

## 📁 Repository Structure

```text
arabic-legal-rag/
├── configs/
│   └── config.yaml              # Global configuration (paths, models, k-neighbors)
├── data/
│   └── legal_corpus_cleaned.json# Cleaned Egyptian Civil Code dataset
├── src/
│   └── arabic_legal_rag/        # Core package source code
│       ├── __init__.py          # Public API exports
│       ├── model.py             # Embedding and vector store loaders
│       ├── pipeline.py          # Database ingestion and retrieval orchestration
│       └── utils.py             # Config & dataset loading helpers
├── tests/
│   └── test_model.py            # Automated integration and retrieval tests
├── vector_db/
│   └── faiss_legal_index/       # Generated FAISS vector index files
├── pyproject.toml               # Package metadata and dependencies
└── README.md

🛠️ Tech Stack
Python 3.10+

Package Manager: uv

Vector Database: FAISS (faiss-cpu)

Embeddings Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Framework: LangChain

Testing: pytest

🚀 Getting Started
1. Environment Setup
Clone the repository and activate your environment:
PowerShell
# Activate your environment
conda activate arabic_rag
2. Package Installation
Install the package in editable mode with development dependencies using uv:

PowerShell
uv pip install -e ".[dev]"
💻 Usage
Build the FAISS Vector Database
To process the Egyptian Civil Code dataset and construct the local vector index:

PowerShell
python -m arabic_legal_rag.pipeline
Run Vector Retrieval in Python
Python
from arabic_legal_rag import run_retrieval

query = "ما هي احكام بطلان العقد وإعادة المتعاقدين إلى الحالة التي كانا عليها؟"
results = run_retrieval(query)

for doc, score in results:
    print(f"Article: {doc.metadata.get('article_number')}")
    print(f"Content: {doc.page_content}\n")
🧪 Running Tests
Execute the test suite using pytest:

PowerShell
python -m pytest

---
## Quickstart (Run on Any Machine via Docker)

Run the entire Arabic Legal RAG system in 3 commands:

```powershell
# 1. Build the portable production container
docker build -t arabic-legal-rag:v1 .

# 2. Run the application server on port 8000
docker run -d -p 8000:8000 --name legal_rag_app arabic-legal-rag:v1

# 3. Query the Arabic Legal Q&A API
curl.exe -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"ما هي المسؤولية عن العمل الشخصي؟\"}"

