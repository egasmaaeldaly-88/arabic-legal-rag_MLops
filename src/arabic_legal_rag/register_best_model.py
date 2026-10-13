import os
from pathlib import Path

import mlflow
from loguru import logger
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from arabic_legal_rag.utils import load_config

EXPERIMENT = "arabic-legal-rag-retrieval"
MODEL_NAME = "ArabicLegalRAG_Retriever"


def register_best_model():
    config = load_config("configs/config.yaml")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]))
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT)
    if not exp:
        raise SystemExit(f"Experiment '{EXPERIMENT}' not found. Run eval.py first.")

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="metrics.hit_rate_at_k >= 0",
        order_by=[
            "metrics.hit_rate_at_k DESC",
            "metrics.mrr_at_k DESC",
            "metrics.avg_latency_sec ASC",
        ],
        max_results=1,
    )
    if not runs:
        raise SystemExit("No runs with retrieval metrics found.")

    best = runs[0]
    run_id = best.info.run_id
    metrics = best.data.metrics
    logger.info(f"Best run {run_id} | params={best.data.params} | metrics={metrics}")

    # Attach the deployable pieces to the best run: config + index.
    client.log_artifact(run_id, "configs/config.yaml", artifact_path="retriever")
    index_dir = best.data.params.get("index_dir") or config["vector_db"]["index_dir"]
    if Path(index_dir).exists():
        client.log_artifacts(run_id, index_dir, artifact_path="retriever/index")

    # Register it: registered model + a real version + the 'production' alias.
    try:
        client.create_registered_model(MODEL_NAME)
    except MlflowException as e:
        if e.error_code != "RESOURCE_ALREADY_EXISTS":
            raise

    source = f"{best.info.artifact_uri}/retriever"
    mv = client.create_model_version(
        name=MODEL_NAME,
        source=source,
        run_id=run_id,
        tags={
            "hit_rate_at_k": f"{metrics.get('hit_rate_at_k', 0):.4f}",
            "mrr_at_k": f"{metrics.get('mrr_at_k', 0):.4f}",
            "embedding_model": best.data.params.get("embedding_model", ""),
            "chunk_size": best.data.params.get("chunk_size", ""),
            "overlap": best.data.params.get("overlap", ""),
        },
    )
    client.set_registered_model_alias(MODEL_NAME, "production", mv.version)
    client.set_tag(run_id, "stage", "production")
    logger.info(f"Registered {MODEL_NAME} v{mv.version} with alias 'production'")

    flag = Path("outputs/best_config_registered.flag")
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(f"{MODEL_NAME} v{mv.version} <- run {run_id}\n")


if __name__ == "__main__":
    register_best_model()