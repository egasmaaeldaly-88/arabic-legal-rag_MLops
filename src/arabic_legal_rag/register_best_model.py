import os
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger

def register_best_chunking_config():
    experiment_name = "arabic-legal-rag-retrieval"
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        logger.error(f"Experiment '{experiment_name}' not found!")
        return

    # Search for the best run based on optimal configuration
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.avg_retrieval_latency_sec ASC"],
        max_results=1
    )
    
    if not runs:
        logger.warning("No runs found in the experiment.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_latency = best_run.data.metrics.get("avg_retrieval_latency_sec")
    chunk_size = best_run.data.params.get("chunk_size")
    chunk_overlap = best_run.data.params.get("chunk_overlap")
    
    logger.info(f"🏆 Best Run Identified!")
    logger.info(f"   - Run ID: {best_run_id}")
    logger.info(f"   - Chunk Size: {chunk_size} | Overlap: {chunk_overlap}")
    logger.info(f"   - Latency: {best_latency}s")

    # Tag this run in MLflow as the Production Candidate
    client.set_tag(best_run_id, "mlflow.note.content", "Production-ready best chunking configuration for Arabic Legal RAG.")
    client.set_tag(best_run_id, "stage", "production")
    
    logger.info("Best configuration successfully processed and tagged for production registry!")
    from pathlib import Path
    output_flag = Path("outputs/best_config_registered.flag")
    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("Registered successfully!")
    logger.info("Flag file created for DVC.")
if __name__ == "__main__":
    register_best_chunking_config()