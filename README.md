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
### BentoML Streaming Demonstration

The RAG pipeline is served using **BentoML** to provide efficient, production-ready async inference and token-by-token streaming capabilities via the `/ask_stream` endpoint.

To test real-time token streaming, execute the following commands in PowerShell:

```powershell
python -c "import json; open('q.json','w',encoding='utf-8').write(json.dumps({'question':'ما هو التزام الشخص بالتعويض عن الأعمال غير المشروعة؟'}, ensure_ascii=False))"
curl.exe -N -X POST http://localhost:8000/ask_stream -H "Content-Type: application/json" -d "@q.json"

Streamed Real-Time Output:

يكون الشخص مسؤولاً عن أعماله غير المشروعة متى صدرت منه وهو مميز وفقاً للمادة 164. كما يحق لكل من وقع عليه اعتداء غير مشروع في حق من حقوق شخصيته أن يطلب وقف هذا الاعتداء مع التعويض عن الضرر طبقاً للمادة 50. وفي حال وقوع الضرر من شخص غير مميز دون وجود مسؤول عنه، يجوز للقاضي إلزام من وقع منه الضرر بتعويض عادل مراعيًا مركز الخصوم بموجب المادة 164.

Sources: مادة 164, مادة 50, مادة 172

Note: The -N flag disables output buffering in curl, enabling progressive token display as generation occurs.
Full live traces for each streaming execution are automatically attached to Langfuse (trace=...).

## Data pipeline

```
Egyptian Civil Code PDF ──► structured JSON (1 record / article) ──► embed (bge-m3) ──► FAISS index
```

- **Corpus:** `data/legal_corpus_structured.json`, 1,148 articles, committed to the repo.
- **Validation** (`validate_corpus.py`): article numbers normalized to integers, repealed articles flagged,
  Arabic text spot-checked on 20 articles (record: see *Extraction spot-check* below).
- **Ingestion** (`ingest.py`): one document per article (no splitting), bilingual text `[AR] ... [EN] ...`,
  embedded with `BAAI/bge-m3`, stored in `data/vector_store_bgem3`.
- **DVC:** the source PDF is tracked (`data/egyptian_civil_code.pdf.dvc`) and the DAG starts from it:
  `build_corpus` (PDF → JSON) → `validate_corpus` → `build_index`. `dvc repro build_index` rebuilds the JSON from
  the PDF, validates it and re-indexes the corpus reproducibly.

```bash
dvc pull data/egyptian_civil_code.pdf.dvc   # the PDF comes from the shared remote
dvc repro build_index
```

**Remote:** the `dagshub` remote is an S3-compatible DagsHub storage (`s3://dvc`), so `dvc pull` works on any
machine. Credentials are never committed: store them locally with
`dvc remote modify --local dagshub access_key_id <token>` and
`dvc remote modify --local dagshub secret_access_key <token>` (written to the git-ignored `.dvc/config.local`).
CI reads the same token from the GitHub secret `DAGSHUB_TOKEN`.

### Extraction spot-check

20 articles were sampled with a fixed seed (`python scripts/spotcheck_sample.py`, 2 repealed + 18 others) and
compared with the PDF page they came from. Each article, its PDF page and the result are recorded in
[`reports/extraction_spotcheck.md`](reports/extraction_spotcheck.md).

20/20 sampled articles (2 repealed, 18 active) match the original PDF pages without extraction errors.

---

## Experiment tracking (MLflow)

Tracking server: the DagsHub-hosted MLflow of this project. The scripts read `MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` from the environment (never from files); without
`MLFLOW_TRACKING_URI` they fall back to the local store in `configs/config.yaml`. Local UI:

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
`production` (`register_best_model.py` registers the run with the best Hit@3 of the experiment, which is the
article-level run: Hit@3 0.731, MRR 0.699).

Screenshots: the five-run comparison on DagsHub (`reports/experiments compare*.png`),
`reports/mlflow_chunking_comparison.png`, `reports/mlflow_chunking_comparison2.png`, and the Model Registry
(`reports/retriever.png`, `reports/version4.png`: version 4 carries the `production` alias).

---

## Evaluation (RAGAS)

- Eval set: **52 questions** (26 pairs, Arabic + English), `data/eval_dataset.json`
- Generator: `qwen/qwen3.8-27b` · Judge: `openai/gpt-oss-120b` (both via Groq)
- `evaluate_ragas.py` is batched and resumable (`--batch-size`, `--cooldown`), caches scores in
  `outputs/ragas_scores.json`, logs each batch to MLflow (experiment `arabic_legal_rag_bge_m3_ragas`) and
  retries on HTTP 429 honoring `Retry-After`.

```bash
pip install -e ".[eval]"
python -m arabic_legal_rag.evaluate_ragas --num-questions 52 --metrics all --batch-size 5 --cooldown 30
```

Judge settings used (environment variables, see `ci.yml`): `JUDGE_MODEL=openai/gpt-oss-120b`,
`JUDGE_API_KEY` (a Groq key), `JUDGE_MAX_TOKENS`, and `JUDGE_EXTRA_BODY='{"reasoning_effort":"low"}'`, which keeps
the reasoning model's thinking short so faithfulness answers are not cut off.

**Results (partial: the judge's daily token limit ran out)**

| Metric | Mean score | Questions scored (of 52) |
| :--- | :---: | :---: |
| faithfulness | 0.912 | 22 |
| answer_relevancy | 0.666 | 28 |
| context_precision | 0.943 | 22 |
| context_recall | 0.823 | 24 |

- The means are over the scored questions only. The scored questions come mostly from the start of the
  dataset, so they are not a random sample of the 52.
- The judge is `openai/gpt-oss-120b` on Groq's free tier, which allows 200,000 tokens per day. The run used it
  up, so many judgments failed with HTTP 429. The script reports `INVALID` when fewer than 90% of the judgments
  are valid; here it means "not enough scored questions yet", not a quality failure.
- The table was produced with `--cache-only` (no judge calls; only the saved scores are reported). Re-running the
  command above without `--cache-only` after the quota resets scores the remaining questions: saved scores are
  reused and questions with missing judgments are retried (up to 3 runs).
- CI gate (faithfulness only, 20 questions, same generator): **0.919**, all 20 judgments valid. The run is logged
  in MLflow, experiment `arabic_legal_rag_bge_m3_ragas` (`reports/ragas_mlflow_run.png`).
- Each batch of the 52-question run is logged to the same MLflow experiment (`ragas_*_running` metrics per batch),
  and every question's scores are attached to its Langfuse trace (see *Monitoring*).

---

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml`: **lint → test → rebuild index → RAGAS gate → Docker build → smoke test → push**

1. `ruff check`
2. `dvc repro build_index`
3. `pytest` (coverage gate ≥ 70%)
4. **RAGAS faithfulness gate:** the build fails if faithfulness < **0.75** on 20 questions
5. **Faithfulness alert:** a notification is sent when faithfulness < **0.80** (see *Monitoring*)
6. Build the image, smoke-test `/health`, `/ask` and the 422 case, then the BentoML image
7. Push to GHCR: `latest` is pushed from `main` only; branches get a branch/sha tag

Image: `ghcr.io/egasmaaeldaly-88/arabic-legal-rag`

---

## Load testing (Locust)

`locustfile.py` sends `POST /ask` (10:1 vs `POST /health`) with random questions from
`data/eval_dataset.json` to the BentoML service. A request fails if the answer is empty or has no sources.

```bash
GENERATOR_BACKEND=extractive bentoml serve arabic_legal_rag.bentoml_service:ArabicLegalRAG --port 8000
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

### Langfuse traces (self-hosted)

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
$env:LANGFUSE_BASE_URL = "http://localhost:3000"   # newer SDK versions read this name
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
- Screenshots: a trace with its spans and the four RAGAS scores `reports/langfuse_trace.png`; the Scores table `reports/langfuse_scores.png`.

### Alert: faithfulness < 0.80

`src/arabic_legal_rag/alerting.py` reads `reports/ragas_summary.json` (written by every
`evaluate_ragas` run). If faithfulness is below **0.80** it

1. prints an `ALERT` line,
2. appends it to `reports/alerts.log`,
3. posts the message to a Discord or Slack incoming webhook when `ALERT_WEBHOOK_URL` is set.

```bash
python -m arabic_legal_rag.alerting                    # threshold 0.80
python -m arabic_legal_rag.alerting --threshold 0.99   # force an alert to test it
```

**How it is triggered:** the CI job runs it right after the RAGAS gate on `main` and on manual
runs (`workflow_dispatch`). The webhook URL is the GitHub secret `ALERT_WEBHOOK_URL`; it is never
stored in the repo. The alert step only notifies and never fails the build: the build fails at
the 0.75 gate, so a score between 0.75 and 0.80 passes CI but still sends a notification.
Screenshot of a triggered alert: `reports/alerting.png`.

## Re-indexing and rollout

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

### Canary rollout

`deploy/canary/` runs **two versions of the BentoML service behind an nginx gateway**. Clients call the gateway
(`http://localhost:8080`, same API as the single service); nginx sends ~90% of requests to the stable version and
~10% to the canary (weights 9:1 in `deploy/canary/nginx.conf`). Every response carries an `X-Served-By` header
showing which version answered.

```text
client ──► nginx :8080 ──┬── 90% ──► bento_stable  (current production image)
                         └── 10% ──► bento_canary  (new index / model / code)
```

**1. Start the canary** (tag the current image as `stable` and the new build as `canary`):

```bash
STABLE_IMAGE=arabic-legal-rag-bento:stable CANARY_IMAGE=arabic-legal-rag-bento:canary \
  docker compose -f deploy/canary/compose.canary.yml up -d
```

**2. Check the split** (about 18 vs 2 out of 20):

```bash
for i in $(seq 1 20); do curl -s -o /dev/null -D - -X POST http://localhost:8080/health | grep -i x-served-by; done | sort | uniq -c
```

On PowerShell:
`1..20 | ForEach-Object { "$((Invoke-WebRequest -Method Post -Uri http://localhost:8080/health -UseBasicParsing).Headers['X-Served-By'])" } | Group-Object`.

**3. Promote or roll back.** The canary stays at 10% until it has served at least 100 requests (for example
`locust -f locustfile.py --headless -u 50 -r 5 -t 5m --host http://localhost:8080`), then these checks decide:

| Check | Promote if | Roll back if |
| :--- | :--- | :--- |
| Errors (5xx, empty answers, missing sources) | canary error rate <= stable | any burst of 5xx, or error rate above stable |
| `/ask` latency (p95, Locust report) | <= 1.2x stable | > 1.5x stable |
| Quality (RAGAS faithfulness, same 20-question set as the CI gate) | >= 0.75 and no more than 0.05 below stable | < 0.75 |

Promotion goes in steps: 10% → 50% (weights 1:1) → 100%. After editing the weights, reload nginx without
downtime:

```bash
docker compose -f deploy/canary/compose.canary.yml exec gateway nginx -s reload
```

To finish a promotion, run the stable service on the canary image (`STABLE_IMAGE=arabic-legal-rag-bento:canary`)
and stop the canary service. To **roll back**, mark the canary server `down` in `nginx.conf` (the commented line
is already there), reload nginx, then `docker compose -f deploy/canary/compose.canary.yml stop bento_canary`.
If an edit does not show up in the running gateway (some editors replace the file and break the bind mount), run
`docker compose -f deploy/canary/compose.canary.yml restart gateway`.

---

## Known limitations

- **RAGAS on the full 52-question set is only partly scored.** The judge (`openai/gpt-oss-120b` on Groq's free
  tier) has a daily limit of 200,000 tokens and the evaluation used it up, so only part of the 52 questions has
  scores (the table in *Evaluation* shows how many per metric). Re-running the same command after the quota
  resets scores the rest: cached scores are reused and questions with missing judgments are retried.
- **The vLLM stack was not run end-to-end on the author's laptop** (no NVIDIA GPU). CI validates both compose
  files and builds and smoke-tests the BentoML image; the Groq and extractive backends are the ones that were
  exercised.
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
├── deploy/canary/                 # canary rollout: compose + nginx traffic split
├── data/
│   ├── legal_corpus_structured.json
│   ├── eval_dataset.json
│   └── egyptian_civil_code.pdf.dvc
├── reports/                       # screenshots, demo recording, load-test report
├── scripts/spotcheck_sample.py    # picks the 20 articles for the extraction spot-check
├── src/arabic_legal_rag/
│   ├── api.py                     # FastAPI app
│   ├── bentoml_service.py         # BentoML service
│   ├── generator.py               # extractive / OpenAI-compatible / transformers backends
│   ├── retriever.py, rag.py       # retrieval + context building
│   ├── ingest.py                  # embed + build FAISS index
│   ├── reindex.py                 # add new articles to the existing index
│   ├── tracing.py                 # optional Langfuse traces (no-op when not configured)
│   ├── alerting.py                # faithfulness < 0.80 notification
│   ├── build_corpus.py, validate_corpus.py
│   ├── run_experiment.py, chunking_sweep.py, register_best_model.py
│   └── evaluate_ragas.py          # RAGAS evaluation
├── tests/
├── locustfile.py                  # load test (50 concurrent users)
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
pip install -e ".[dev]"        # extras: dev, eval, serve, tracing
pytest
uvicorn arabic_legal_rag.api:app --reload
```

Never commit API keys, `.env`, `mlflow.db`, or `.pyc` files.