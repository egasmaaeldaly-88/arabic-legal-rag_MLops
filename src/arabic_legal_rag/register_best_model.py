import os
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger
from pathlib import Path

def register_best_chunking_config():
    """Identify the best-performing retrieval run in MLflow, tag it as production, and create a flag file for DVC."""
    experiment_name = "arabic-legal-rag-retrieval"
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        logger.error(f"Experiment '{experiment_name}' not found!")
        return

    # Search for the best run based on optimal Hit Rate and MRR, breaking ties with latency
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.hit_rate_at_k DESC", "metrics.mrr_at_k DESC", "metrics.avg_latency_sec ASC"],
        max_results=1
    )
    
    if not runs:
        logger.warning("No runs found in the experiment.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    hit_rate = best_run.data.metrics.get("hit_rate_at_k")
    mrr = best_run.data.metrics.get("mrr_at_k")
    latency = best_run.data.metrics.get("avg_latency_sec")
    embedding_model = best_run.data.params.get("embedding_model", "BAAI/bge-m3")
    top_k = best_run.data.params.get("top_k", 5)
    
    logger.info("🏆 Best Run Identified!")
    logger.info(f"   - Run ID: {best_run_id}")
    logger.info(f"   - Embedding Model: {embedding_model} | Top-K: {top_k}")
    logger.info(f"   - Hit Rate: {hit_rate} | MRR: {mrr} | Latency: {latency}s")

    # Tag this run in MLflow as the Production Candidate
    client.set_tag(best_run_id, "mlflow.note.content", "Production-ready article-based RAG configuration for Arabic Legal RAG.")
    client.set_tag(best_run_id, "stage", "production")
    
    logger.info("Best configuration successfully processed and tagged for production registry!")
    
    # Create the output flag file for DVC pipeline completion
    output_flag = Path("outputs/best_config_registered.flag")
    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("Registered successfully!")
    logger.info("Flag file created for DVC.")

if __name__ == "__main__":
    register_best_chunking_config()