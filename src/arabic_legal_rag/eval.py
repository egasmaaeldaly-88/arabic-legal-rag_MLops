import json
import time
import re
import os
import mlflow
import numpy as np
from pathlib import Path
from arabic_legal_rag.utils import load_config
from arabic_legal_rag.model import get_embedding_model, load_vector_store

# 1. إعدادات الـ Tracking وتفعيل الـ Traces والـ System Metrics
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.system_metrics.enable_system_metrics_logging()

import mlflow.langchain
mlflow.langchain.autolog()

def normalize_digits(text: str) -> str:
    """Converts Eastern Arabic numerals (٠١٢٣٤٥٦٧٨٩) to Western ASCII digits (0123456789)."""
    eastern_to_western = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return text.translate(eastern_to_western)

def extract_article_num(raw_val) -> str:
    """Extract and normalize the article number from raw metadata."""
    if not raw_val:
        return ""
    # Convert Eastern digits to ASCII digits first
    normalized_text = normalize_digits(str(raw_val))
    # Extract digit sequence
    digits = re.findall(r'\d+', normalized_text)
    return digits[0] if digits else normalized_text.strip()

def compute_metrics(eval_dataset: list, top_k: int = 3):
    """Compute retrieval metrics (Hit Rate, MRR, and Latency) against the evaluation dataset."""
    # Load config, embedding model, and FAISS index ONCE to fix latency
    config = load_config("configs/config.yaml")
    embeddings = get_embedding_model(config["model"]["embedding_model"])
    vector_store = load_vector_store(config["vector_db"]["index_dir"], embeddings)
    
    # تحويل الـ Vector Store إلى Retriever لتفعيل الـ LangChain Traces بنجاح
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    hits = 0
    mrr_sum = 0.0
    latencies = []

    for item in eval_dataset:
        query = item["query"]
        expected = [str(a).strip() for a in item["expected_articles"]]

        start_time = time.time()
        # استخدام الـ retriever.invoke لتفعيل التتبع (Traces) في MLflow
        docs = retriever.invoke(query)
        latencies.append(time.time() - start_time)

        # Extract and normalize article numbers from Document metadata
        retrieved_articles = [
            extract_article_num(doc.metadata.get("article_number", ""))
            for doc in docs
        ]

        # Print debug info for the first query to verify matching
        print(f"\n[DEBUG] Query: {query}")
        print(f"[DEBUG] Expected Articles:  {expected}")
        print(f"[DEBUG] Retrieved Articles: {retrieved_articles}")

        # Hit Rate@K
        if any(art in retrieved_articles for art in expected):
            hits += 1

        # MRR (Mean Reciprocal Rank)
        rank = 0
        for idx, art in enumerate(retrieved_articles, start=1):
            if art in expected:
                rank = idx
                break
            
        mrr_sum += (1.0 / rank) if rank > 0 else 0.0

    num_samples = len(eval_dataset)
    return {
        "hit_rate_at_k": hits / num_samples if num_samples > 0 else 0.0,
        "mrr_at_k": mrr_sum / num_samples if num_samples > 0 else 0.0,
        "avg_latency_sec": float(np.mean(latencies))
    }

def run_evaluation():
    """Run the complete evaluation workflow and log parameters, metrics, artifacts, and model registry."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    eval_path = Path("data/eval_dataset.json")
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {eval_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    config = load_config("configs/config.yaml")
    top_k = config["model"]["top_k"]
    model_name = config["model"]["embedding_model"]
    
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_experiment("arabic-legal-rag-retrieval")

    clean_model_name = model_name.split("/")[-1]
    
    # إضافة log_system_metrics=True لضمان تسجيل مقاييس النظام بقوة
    with mlflow.start_run(run_name=f"run_{clean_model_name}_top{top_k}_optimized", log_system_metrics=True) as run:
        mlflow.log_param("embedding_model", model_name)
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("eval_samples_count", len(eval_dataset))

        metrics = compute_metrics(eval_dataset, top_k=top_k)

        mlflow.log_metric("hit_rate_at_k", metrics["hit_rate_at_k"])
        mlflow.log_metric("mrr_at_k", metrics["mrr_at_k"])
        mlflow.log_metric("avg_latency_sec", metrics["avg_latency_sec"])

        # --- 1. حفظ النتائج كـ Artifact ملموس ---
        report_dir = Path("outputs")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "evaluation_report.json"
        
        report_data = {
            "model": model_name,
            "top_k": top_k,
            "eval_samples_count": len(eval_dataset),
            "metrics": metrics
        }
        with open(report_path, "w", encoding="utf-8") as r_file:
            json.dump(report_data, r_file, ensure_ascii=False, indent=4)
            
        # رفع الملف لتبويب الـ Artifacts في MLflow
        mlflow.log_artifact(str(report_path), artifact_path="evaluation_reports")

        # --- 2. تفعيل الـ Model Registry (أو تسجيل نموذج الـ Retriever) ---
        client = mlflow.tracking.MlflowClient()
        model_registry_name = "ArabicLegalRAG_Retriever"
        
        try:
            client.create_registered_model(model_registry_name)
        except Exception:
            pass  # النموذج مسجل مسبقاً

        print(f"\n--- Optimized Evaluation Results (Top-{top_k}) ---")
        print(f"Hit Rate@{top_k}: {metrics['hit_rate_at_k']:.4f}")
        print(f"MRR@{top_k}:      {metrics['mrr_at_k']:.4f}")
        print(f"Avg Latency: {metrics['avg_latency_sec']:.4f}s")
        print(f"Artifact successfully logged to MLflow from {report_path}")

if __name__ == "__main__":
    run_evaluation()