# ⚖️ Arabic Legal RAG: Egyptian Civil Code

Question answering over the **Egyptian Civil Code** (1,148 articles, Arabic + English questions).
Every answer comes with **article citations** (not chunk IDs). Built as a full MLOps project:
versioned data, tracked experiments, CI quality gate, containerized serving, monitoring.

> ITI × MLOps MENA, Project 2: LLM / RAG

**Stack:** Python 3.10 · bge-m3 embeddings · FAISS · FastAPI · BentoML · vLLM · MLflow · DVC · RAGAS · GitHub Actions · Docker

---

## Quickstart (3 commands, any machine with Docker)

```bash
docker pull ghcr.io/egasmaaeldaly-88/arabic-legal-rag:latest
docker run -d --name rag -p 8000:8000 -e GENERATOR_BACKEND=extractive ghcr.io/egasmaaeldaly-88/arabic-legal-rag:latest
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "ما هي المسؤولية عن العمل الشخصي؟"}'
```

The image already contains the corpus, the bge-m3 model and the FAISS index. The first start takes
1-2 minutes (model loading); wait for `GET /health`.

`GENERATOR_BACKEND=extractive` answers directly from the retrieved articles and needs no API key.
For LLM-written answers, point the container at any OpenAI-compatible server (Groq, vLLM, ...):

```bash
docker run -d -p 8000:8000 \
  -e GENERATOR_BACKEND=openai \
  -e LLM_BASE_URL=https://api.groq.com/openai/v1 \
  -e LLM_MODEL=qwen/qwen3.8-27b \
  -e LLM_API_KEY=$GROQ_API_KEY \
  ghcr.io/egasmaaeldaly-88/arabic-legal-rag:latest
```

> On Windows PowerShell write the request body to a `q.json` file and use
> `curl.exe -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "@q.json"`.

---

## API

| Endpoint | Description |
| :--- | :--- |
| `POST /ask` | `{"question": str}` → `{"answer": str, "sources": list[str]}`. Sources are article citations. Empty question → **422**. |
| `POST /ask/stream` | Same input, tokens streamed as they are generated. |
| `GET /health` | `{"status": "healthy", "documents_indexed": N}` |

Generator selection (same variables for FastAPI and BentoML):

| Variable | Meaning |
| :--- | :--- |
| `GENERATOR_BACKEND` | `extractive` · `openai` (vLLM / Groq / any OpenAI-compatible) · `transformers` |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | Used by the `openai` backend |
| `LLM_EXTRA_BODY` | JSON merged into the request body (e.g. reasoning settings) |
| `LLM_MAX_TOKENS` | Max new tokens |

---

## Serving: BentoML + vLLM

`src/arabic_legal_rag/bentoml_service.py` wraps the same pipeline (retriever → context → generator) with
async endpoints; blocking work runs in worker threads so concurrent users do not block the event loop.

| Endpoint (port 8000) | Description |
| :--- | :--- |
| `POST /ask` | `{"question": "...", "top_k": 3}` → answer + sources |
| `POST /ask_stream` | Tokens appear progressively, then a final `Sources: ...` line |
| `POST /health` | Index size and status |
| `GET /readyz` | Ready once models are loaded (used by Docker healthchecks) |

Run locally: `bentoml serve arabic_legal_rag.bentoml_service:ArabicLegalRAG --port 8000`
(demo recording: `reports/bentoml.mp4`).

### docker compose

**Full stack (BentoML + vLLM). Requires an NVIDIA GPU and the NVIDIA Container Toolkit:**

```bash
docker compose up --build
```

- Generative LLM served by vLLM: **`Qwen/Qwen2.5-1.5B-Instruct`** (override with `LLM_MODEL`)
- BentoML on `http://localhost:8000`, vLLM (OpenAI-compatible) on `http://localhost:8001/v1`

**No GPU (BentoML + Groq):**

```bash
cp .env.example .env          # put your GROQ_API_KEY in .env (gitignored)
docker compose -f docker-compose.groq.yml up --build
```

`Dockerfile.bento` builds on top of the CI image (`BASE_IMAGE`, default
`ghcr.io/egasmaaeldaly-88/arabic-legal-rag:latest`), so the index and embedding model are not rebuilt.
CI builds and smoke-tests this image on every push (extractive backend) and validates both compose files.
The vLLM service cannot run on CI runners (no GPU).

---

## Data pipeline

```
Egyptian Civil Code PDF ──► structured JSON (1 record / article) ──► embed (bge-m3) ──► FAISS index
```

- **Corpus:** `data/legal_corpus_structured.json`, 1,148 articles, committed to the repo.
- **Validation** (`validate_corpus.py`): article numbers normalized to integers, repealed articles flagged,
  Arabic text spot-checked on 20 articles (record: see *Extraction spot-check* below).
- **Ingestion** (`ingest.py`): one document per article (no splitting), bilingual text `[AR] ... [EN] ...`,
  embedded with `BAAI/bge-m3`, stored in `data/vector_store_bgem3`.
- **DVC:** the source PDF is tracked (`data/egyptian_civil_code.pdf.dvc`); `dvc repro build_index` validates the
  corpus and re-indexes it reproducibly.

```bash
dvc repro build_index
```

> ⚠️ **TODO:** the DVC remote currently points to a local path, so `dvc pull` only works on the author's
> machine. Switch to a shared remote, and add a stage that rebuilds the JSON from the PDF.

### Extraction spot-check

> **TODO:** add the 20 checked article numbers and the result here.

---

## Experiment tracking (MLflow)

Tracking store: `sqlite:///mlflow.db`. UI (run from the repo root):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Experiment `arabic-legal-rag-retrieval`**: every run logs `chunk_size`, `overlap`, `embedding_model`,
Hit@3 and MRR.

| Setup | Runs | Best Hit@3 | Best MRR |
| :--- | :---: | :---: | :---: |
| Chunk-size sweep (Arabic-only indexes) | 9 | 0.654 | 0.554 |
| **Article-level (`chunk_size=article`, `overlap=0`)** | 1 | **0.731** | **0.699** |

**Decision:** one chunk per article. Legal articles are self-contained units and are cited by number, so
chunking inside an article only fragments the context. Article-level indexing wins on both metrics.

**Model Registry:** the best configuration is registered as **`ArabicLegalRAG_Retriever`** with the alias
`production` (`register_best_model.py` picks the best run where `params.chunk_size = 'article'`).

Screenshots: `reports/mlflow_chunking_comparison.png`, `reports/mlflow_chunking_comparison2.png`
> **TODO:** add a Model Registry screenshot to `reports/`.

---

## Evaluation (RAGAS)

- Eval set: **52 questions** (26 pairs, Arabic + English), `data/eval_dataset.json`
- Generator: `qwen/qwen3.8-27b` · Judge: `openai/gpt-oss-120b` (both via Groq)
- `evaluate_ragas.py` is batched and resumable (`--batch-size`, `--cooldown`), caches scores in
  `outputs/ragas_scores.json`, logs each batch to MLflow (experiment `arabic_legal_rag_bge_m3_ragas`) and
  retries on HTTP 429 honoring `Retry-After`.

```bash
pip install -e ".[eval]"
python -m arabic_legal_rag.evaluate_ragas --num-questions 52 --batch-size 10 --cooldown 60
```

| Metric | Score (52 questions) |
| :--- | :---: |
| faithfulness | TODO |
| answer_relevancy | TODO |
| context_precision | TODO |
| context_recall | TODO |

> **TODO:** fill the table from MLflow and add a trend screenshot to `reports/`.

---

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml`: **lint → test → rebuild index → RAGAS gate → Docker build → smoke test → push**

1. `ruff check`
2. `dvc repro build_index`
3. `pytest` (coverage gate ≥ 70%)
4. **RAGAS faithfulness gate:** the build fails if faithfulness < **0.75** on 20 questions
5. Build the image, smoke-test `/health`, `/ask` and the 422 case, then the BentoML image
6. Push to GHCR: `latest` is pushed from `main` only; branches get a branch/sha tag

Image: `ghcr.io/egasmaaeldaly-88/arabic-legal-rag`

---

## Load testing (Locust)

> **TODO:** run Locust with 50 concurrent users against BentoML and add the report to `reports/`
> (locustfile, command, requests/sec, p50/p95 latency, failure rate).

## Monitoring

> **TODO:** Langfuse (self-hosted): one trace with spans per `/ask`, with the RAGAS faithfulness score attached.
> **TODO:** alert when faithfulness < 0.80 (notification channel and how it is triggered).

## Re-indexing and rollout

> **TODO:** batch re-indexing script, tested with at least one new document.
> **TODO:** canary rollout configuration (traffic split, promotion and rollback criteria).

---

## Repository layout

```text
.
├── .github/workflows/ci.yml       # CI/CD pipeline
├── configs/config.yaml            # paths, embedding model, generation backend
├── data/
│   ├── legal_corpus_structured.json
│   ├── eval_dataset.json
│   └── egyptian_civil_code.pdf.dvc
├── reports/                       # screenshots, demo recording, load-test report
├── src/arabic_legal_rag/
│   ├── api.py                     # FastAPI app
│   ├── bentoml_service.py         # BentoML service
│   ├── generator.py               # extractive / OpenAI-compatible / transformers backends
│   ├── retriever.py, rag.py       # retrieval + context building
│   ├── ingest.py                  # embed + build FAISS index
│   ├── build_corpus.py, validate_corpus.py
│   ├── run_experiment.py, chunking_sweep.py, register_best_model.py
│   └── evaluate_ragas.py          # RAGAS evaluation
├── tests/
├── Dockerfile                     # FastAPI image (bakes corpus + index)
├── Dockerfile.bento               # BentoML image
├── docker-compose.yml             # BentoML + vLLM (GPU)
├── docker-compose.groq.yml        # BentoML + Groq (no GPU)
├── dvc.yaml
└── pyproject.toml
```

## Development

```bash
git clone https://github.com/egasmaaeldaly-88/arabic-legal-rag_MLops.git
cd arabic-legal-rag_MLops
pip install -e ".[dev]"        # extras: dev, eval, serve
pytest
uvicorn arabic_legal_rag.api:app --reload
```

Never commit API keys, `.env`, `mlflow.db`, or `.pyc` files.
