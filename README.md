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

>## Load testing (Locust)

`locustfile.py` sends `POST /ask` (10:1 vs `POST /health`) with random questions from
`data/eval_dataset.json` to the BentoML service. A request fails if the answer is empty or has no sources.

```bash
GENERATOR_BACKEND=local Qwen generator bentoml serve arabic_legal_rag.bentoml_service:ArabicLegalRAG --port 8000
locust -f locustfile.py --headless -u 50 -r 5 -t 2m --host http://localhost:8000 \
  --html reports/locust_report.html --csv reports/locust
```

**Result: 50 concurrent users, 2 minutes, 511 requests, 0 failures.**

| Endpoint | Requests | p50 | p95 | p99 |
| :--- | :---: | :---: | :---: | :---: |
| POST /ask | 459 | 9.4 s | 13 s | 15 s |
| POST /health | 52 | 25 ms | 2.1 s | 2.2 s |

Throughput ≈ 4.3 req/s. Setup: Windows laptop (Ryzen 7 PRO 5850U, no GPU), extractive backend, server and
Locust on the same machine, so latency is bounded by CPU (bge-m3 query embedding). Full report:
`reports/locust_report.html`.

## Monitoring

Faithfulness alert (< 0.80)

After each RAGAS run, evaluate_ragas.py writes reports/ragas_summary.json. The alert check reads it:

bash
python -m arabic_legal_rag.alerting                    # threshold 0.80
python -m arabic_legal_rag.alerting --threshold 0.99   # force an alert to test it

If faithfulness < 0.80 it prints ALERT: ..., appends to reports/alerts.log, and POSTs to a Discord or Slack incoming webhook when ALERT_WEBHOOK_URL is set (keep the URL secret, never commit it). Exit code 2 means an alert fired, so CI can notify or fail on it.

TODO: Langfuse (self-hosted): one trace with spans per /ask, with the RAGAS faithfulness score attached.

## Re-indexing and rollout
Batch re-indexing

reindex.py adds new articles to the existing FAISS index without rebuilding it. Only the new articles are embedded; articles whose number is already indexed are skipped, so it is safe to re-run.

bash
python -m arabic_legal_rag.reindex --input data/new_articles_sample.json
# documents_indexed: 1093 -> 1094 (added 1, skipped 0 already indexed)

Restart the API/BentoML service afterwards so it loads the updated index. Tested in tests/test_reindex.py and with one new document (/health shows documents_indexed going up by 1). Note: dvc repro build_index rebuilds from data/legal_corpus_structured.json, so add permanent articles to that file too.

### Batch re-indexing

`src/arabic_legal_rag/reindex.py` adds **new articles to the existing FAISS index** without rebuilding it:
only the new documents are embedded (bge-m3), so it takes seconds instead of re-embedding the whole corpus.
Articles whose number is already indexed are skipped, so running it twice never creates duplicates.

```bash
python -m arabic_legal_rag.reindex --input data/new_articles_sample.json
```

The input is a JSON list with the same schema as `data/legal_corpus_structured.json`.

**Tested with a new document:** `data/new_articles_sample.json` contains one test article (No. 1150).

```text
run 1: documents_indexed: 1093 -> 1094 (added 1, skipped 0 already indexed)
run 2: documents_indexed: 1094 -> 1094 (added 0, skipped 1 already indexed)
```

Unit tests (`tests/test_reindex.py`, fake embeddings) cover: only new articles are added, the new article is
searchable, re-running is idempotent, batching, and duplicates inside the input file.

Notes:
- Restart the API / BentoML service afterwards so it loads the updated index.
- `dvc repro build_index` rebuilds the index from `legal_corpus_structured.json`, so to keep a new article
  permanently, also add it to that file. The test article is intentionally not part of the corpus.

> **TODO:** canary rollout configuration (traffic split, promotion and rollback criteria).

### Canary rollout

Configuration: deploy/canary/ (nginx gateway with weighted upstreams, 90% stable / 10% canary).

bash
STABLE_IMAGE=arabic-legal-rag-bento:stable CANARY_IMAGE=arabic-legal-rag-bento:canary \
  docker compose -f deploy/canary/docker-compose.canary.yml up -d

Clients call http://localhost:8080; the X-Served-By response header shows which version answered.

Stage	Canary traffic	Move to the next stage when
1	10% (weights 9:1)	at least 200 canary requests and all checks below pass
2	50% (weights 1:1)	same checks hold for another 200 requests
3	100%	canary becomes the new stable

Promotion checks (canary vs stable): error rate no more than 0.5 pp higher; p95 latency within 1.2x; RAGAS faithfulness >= 0.80 on the 20-question set and no more than 0.03 below stable.

Rollback: if any check fails or the faithfulness alert fires, mark the canary server down in nginx.conf and run docker compose -f deploy/canary/docker-compose.canary.yml exec gateway nginx -s reload. Traffic returns to stable immediately; no rebuild needed.

## Observability: Langfuse traces

Every `/ask` creates a [Langfuse](https://langfuse.com) trace with spans:

```
ask                  input = question, output = answer + cited articles
├── retrieve         cited articles, number of documents
└── generate         generation: model name, context passed to the LLM, answer
```

The RAGAS evaluation (`python -m arabic_legal_rag.evaluate_ragas`) creates the same trace for
every evaluation question and then attaches each metric (faithfulness, answer_relevancy,
context_precision, context_recall) to that question's trace as a Langfuse score.

### Run Langfuse (self-hosted)

Langfuse is not part of this repo; run its official compose file next to the project:

```powershell
cd C:\
git clone --depth 1 https://github.com/langfuse/langfuse.git
cd C:\langfuse
docker compose up          # UI at http://localhost:3000
```

Create an account, an organization and a project, then create API keys
(Settings -> API Keys).

### Connect the app

Keys are never stored in files; set them in the shell that starts the app:

```powershell
pip install -e ".[tracing]"
$env:LANGFUSE_HOST = "http://localhost:3000"
$env:LANGFUSE_PUBLIC_KEY = "<public key>"
$env:LANGFUSE_SECRET_KEY = "<secret key>"
uvicorn arabic_legal_rag.api:app --port 8000      # or: bentoml serve ...
```

Without these variables (or without the `langfuse` package) tracing is switched off and the
API behaves exactly as before. `LANGFUSE_ENABLED=0` forces it off.

When the API runs in Docker, point it at the host with
`LANGFUSE_HOST=http://host.docker.internal:3000`.

### Notes

- `/ask/stream` (and BentoML `/ask_stream`) record their trace when the stream finishes; the
  durations (retrieve, time to first token, total) are stored in the trace metadata.
- `reports/ragas_results.csv` has a `langfuse_trace_id` column linking each question to its trace.
- Screenshot of a trace with spans and the faithfulness score: `reports/langfuse_trace.png`.


---

## Known limitations

- **Retrieval depends on wording.** Each article is embedded on its own text, so a question phrased far from the
  article's legal wording can retrieve related articles instead of the right one. Example: *"المسؤولية عن العمل
  الشخصي"* returns articles about liability for others' acts, while Article 163 (*"كل خطأ سبب ضررا للغير يلزم
  من ارتكبه بالتعويض"*) is found when the question uses its wording (*"ما جزاء من يرتكب خطأ يسبب ضررا للغير؟"*).
  The phrase is a section title in the Code, not part of the article text.
- **Planned improvements (not implemented):** add each article's book/chapter/section titles to the text that is
  embedded; rerank the top 15-20 candidates (e.g. `BAAI/bge-reranker-v2-m3`) before keeping the top 3. Both
  should be measured on a set of colloquial questions (Hit@3 before/after) and logged to MLflow.
- **PDF heading cleanup:** chapter and section headings printed between articles used to stay glued to the end
  of the previous article (e.g. Article 162). `build_corpus.py` now removes them
  (`strip_trailing_headings`, covered by `tests/test_strip_headings.py`). Text extracted from the PDF may still
  contain small OCR-style errors, so spot-check before relying on exact wording.

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
│   ├── reindex.py                 # add new articles to the existing index
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