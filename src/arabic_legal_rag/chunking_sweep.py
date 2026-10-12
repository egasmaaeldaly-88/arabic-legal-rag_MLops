"""Evaluate every existing chunk-size/overlap FAISS index and log one MLflow run each.

Looks for data/vector_store_sz_<size>_ov_<overlap>/ folders, scores each one on
data/eval_dataset.json (Hit Rate@K, MRR@K, latency) and logs chunk_size, overlap,
embedding_model, top_k and total_chunks. Runs go into the same experiment as eval.py
("arabic-legal-rag-retrieval") so register_best_model.py can pick the best one.

Query embeddings are computed ONCE and reused for all indexes (light on the laptop).

Usage (from the repo root):
    python src/arabic_legal_rag/chunking_sweep.py
    python src/arabic_legal_rag/chunking_sweep.py --top-k 5
"""

import argparse
import json
import re
import time
from pathlib import Path

import mlflow
import numpy as np
from loguru import logger

from arabic_legal_rag.model import get_embedding_model, load_vector_store
from arabic_legal_rag.utils import load_config

EXPERIMENT = "arabic-legal-rag-retrieval"
INDEX_PATTERN = re.compile(r"vector_store_sz_(\d+)_ov_(\d+)$")
EASTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def article_num(raw) -> str:
    """Normalize an article number (handles Eastern Arabic digits)."""
    if raw is None or raw == "":
        return ""
    text = str(raw).translate(EASTERN)
    digits = re.findall(r"\d+", text)
    return digits[0] if digits else text.strip()


def find_indexes(data_dir: Path) -> list[tuple[int, int, Path]]:
    found = []
    for p in sorted(data_dir.iterdir()):
        m = INDEX_PATTERN.match(p.name)
        if m and p.is_dir():
            found.append((int(m.group(1)), int(m.group(2)), p))
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--dataset", default="data/eval_dataset.json")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    config = load_config("configs/config.yaml")
    top_k = args.top_k or int(config["model"]["top_k"])
    model_name = config["model"]["embedding_model"]

    indexes = find_indexes(Path(args.data_dir))
    if not indexes:
        logger.error("No data/vector_store_sz_*_ov_* folders found.")
        return 1
    logger.info(f"Found {len(indexes)} chunking configs; top_k={top_k}")

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    embeddings = get_embedding_model(model_name)

    # Embed every query once and reuse it for all indexes.
    queries = [item["query"] for item in dataset]
    vectors, embed_times = [], []
    for q in queries:
        t0 = time.time()
        vectors.append(embeddings.embed_query(q))
        embed_times.append(time.time() - t0)
    avg_embed = float(np.mean(embed_times))
    logger.info(f"Embedded {len(queries)} queries (avg {avg_embed:.3f}s each)")

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(EXPERIMENT)

    for chunk_size, overlap, index_dir in indexes:
        store = load_vector_store(str(index_dir), embeddings)
        hits, mrr_sum, search_times = 0, 0.0, []

        for item, vec in zip(dataset, vectors):
            expected = {str(a).strip() for a in item["expected_articles"]}
            t0 = time.time()
            docs = store.similarity_search_by_vector(vec, k=top_k)
            search_times.append(time.time() - t0)

            retrieved = [article_num(d.metadata.get("article_number")) for d in docs]
            rank = next((i for i, a in enumerate(retrieved, 1) if a in expected), 0)
            hits += 1 if rank else 0
            mrr_sum += 1.0 / rank if rank else 0.0

        n = len(dataset)
        metrics = {
            "hit_rate_at_k": hits / n,
            "mrr_at_k": mrr_sum / n,
            "avg_latency_sec": avg_embed + float(np.mean(search_times)),
        }

        with mlflow.start_run(run_name=f"chunk_{chunk_size}_ov_{overlap}"):
            mlflow.log_params(
                {
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "embedding_model": model_name,
                    "top_k": top_k,
                    "eval_samples_count": n,
                    "index_dir": str(index_dir).replace("\\", "/"),
                    "total_chunks": int(store.index.ntotal),
                }
            )
            mlflow.log_metrics(metrics)
        logger.info(
            f"chunk={chunk_size:>4} overlap={overlap:>3} "
            f"chunks={store.index.ntotal:>5} "
            f"hit@{top_k}={metrics['hit_rate_at_k']:.3f} mrr@{top_k}={metrics['mrr_at_k']:.3f}"
        )

    logger.info("Done. Open: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())