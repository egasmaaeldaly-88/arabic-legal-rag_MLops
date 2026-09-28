import json
import time
import mlflow
import numpy as np
from pathlib import Path
from arabic_legal_rag.pipeline import RAGPipeline

def compute_metrics(eval_dataset: list, pipeline: RAGPipeline, top_k: int = 3):
    hits = 0
    mrr_sum = 0.0
    latencies = []

    for item in eval_dataset:
        query = item["query"]
        expected = [str(a) for a in item["expected_articles"]]

        start_time = time.time()
        results = pipeline.search(query, top_k=top_k)
        latencies.append(time.time() - start_time)

        # Extract retrieved article numbers from metadata
        retrieved_articles = [str(res.get("article_id", "")) for res in results]

        # Calculate Hit Rate@K
        hit = any(art in retrieved_articles for art in expected)
        if hit:
            hits += 1

        # Calculate MRR (Mean Reciprocal Rank)
        rank = 0
        for idx, art in enumerate(retrieved_articles, start=1):
            if art in expected:
                rank = idx
                break
        
        mrr_sum += (1.0 / rank) if rank > 0 else 0.0

    num_samples = len(eval_dataset)
    hit_rate = hits / num_samples if num_samples > 0 else 0.0
    mrr = mrr_sum / num_samples if num_samples > 0 else 0.0
    avg_latency = float(np.mean(latencies))

    return {
        "hit_rate_at_k": hit_rate,
        "mrr_at_k": mrr,
        "avg_latency_sec": avg_latency
    }

def run_evaluation():
    eval_path = Path("data/eval_dataset.json")
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {eval_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    # Initialize RAG Pipeline
    pipeline = RAGPipeline()
    top_k = 3
    embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Set MLflow Experiment
    mlflow.set_experiment("arabic-legal-rag-retrieval")

    with mlflow.start_run(run_name="baseline_minilm_top3"):
        # Log Hyperparameters
        mlflow.log_param("embedding_model", embedding_model)
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("eval_samples_count", len(eval_dataset))

        # Compute Metrics
        metrics = compute_metrics(eval_dataset, pipeline, top_k=top_k)

        # Log Metrics
        mlflow.log_metric("hit_rate_at_k", metrics["hit_rate_at_k"])
        mlflow.log_metric("mrr_at_k", metrics["mrr_at_k"])
        mlflow.log_metric("avg_latency_sec", metrics["avg_latency_sec"])

        print("\n--- Evaluation Results ---")
        print(f"Hit Rate@{top_k}: {metrics['hit_rate_at_k']:.4f}")
        print(f"MRR@{top_k}:      {metrics['mrr_at_k']:.4f}")
        print(f"Avg Latency: {metrics['avg_latency_sec']:.4f}s")

if __name__ == "__main__":
    run_evaluation()