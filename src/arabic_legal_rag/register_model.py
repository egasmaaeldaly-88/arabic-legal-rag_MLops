import os
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger
from arabic_legal_rag.utils import load_config

def register_best_chunking_config():
    config = load_config()
    experiment_name = "Arabic_Legal_RAG_Chunking_Optimization"
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        logger.error(f"Experiment '{experiment_name}' not found!")
        return

    # Search for the best run based on lowest retrieval latency and optimal chunking
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

    # Log/Tag this run in MLflow as the Production Candidate
    client.set_tag(best_run_id, "mlflow.note.content", "Production-ready best chunking configuration for Arabic Legal RAG.")
    
    # Dynamically register model artifact in MLflow Model Registry using the best run's parameters
    model_uri = f"runs:/{best_run_id}/run_info_sz_{chunk_size}_ov_{chunk_overlap}.txt"
    registered_model_name = "ArabicLegalRAG_BestChunking"
    
    try:
        model_version = mlflow.register_model(model_uri, registered_model_name)
        logger.info(f"Model successfully registered as version {model_version.version} under name '{registered_model_name}'!")
    except Exception as e:
        logger.warning(f"Registration note: {e}")
    
    logger.info("Best configuration successfully processed and registered for production registry!")

if __name__ == "__main__":
    register_best_chunking_config()