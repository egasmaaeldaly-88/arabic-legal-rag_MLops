import json
import os
from pathlib import Path
import mlflow
from arabic_legal_rag.eval import compute_metrics
from arabic_legal_rag.utils import load_config
from arabic_legal_rag.model import get_embedding_model, load_vector_store

def run_three_hf_model_families():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("arabic-legal-rag-model-families")
    
    eval_path = Path("data/eval_dataset.json")
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {eval_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    config = load_config("configs/config.yaml")
    
    # تعريف 3 عائلات نماذج (Embedding Models مختلفة من Hugging Face)
    # ملاحظة: تأكدي أن هذه الموديلات تم تحميلها مسبقاً أو متاحة لديكِ
    model_families = [
        {
            "family_name": "Baseline_MiniLM",
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        },
        {
            "family_name": "Alternative_Model",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2"  # أو أي موديل بديل كنتِ تجريبيه
        },
        {
            "family_name": "Optimal_BGE_M3",
            "model_name": config["model"]["embedding_model"]  # الموديل الأساسي القوي بتاعك
        }
    ]

    for fam in model_families:
        print(f"\n--- Training/Evaluating Model Family: {fam['family_name']} ({fam['model_name']}) ---")
        
        # مؤقتاً نقوم بتعديل الـ config أو تمرير اسم الموديل لعمل الـ Vector Store الخاص به
        config["model"]["embedding_model"] = fam["model_name"]
        
        with mlflow.start_run(run_name=f"run_{fam['family_name']}", log_system_metrics=True) as run:
            # تسجيل الـ Tags الأساسية
            mlflow.set_tag("author", "Asmaa")
            mlflow.set_tag("framework", "langchain_huggingface")
            mlflow.set_tag("git_commit", "main_v2")
            mlflow.set_tag("data_version", "v1.0_json")
            
            # تسجيل الـ Params
            mlflow.log_param("model_family", fam["family_name"])
            mlflow.log_param("embedding_model", fam["model_name"])
            mlflow.log_param("top_k", config["model"]["top_k"])
            mlflow.log_param("eval_samples_count", len(eval_dataset))

            # حساب المقاييس باستخدام الموديل الحالي
            # ملاحظة: compute_metrics تستدعي get_embedding_model بناءً على الكونفجريشن أو يمكننا تحديثها
            metrics = compute_metrics(eval_dataset, top_k=config["model"]["top_k"])

            # تسجيل الـ Metrics
            mlflow.log_metric("hit_rate_at_k", metrics["hit_rate_at_k"])
            mlflow.log_metric("mrr_at_k", metrics["mrr_at_k"])
            mlflow.log_metric("avg_latency_sec", metrics["avg_latency_sec"])

            # حفظ التقرير كـ Artifact
            report_dir = Path("outputs")
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"evaluation_report_{fam['family_name']}.json"
            
            report_data = {
                "model_family": fam["family_name"],
                "embedding_model": fam["model_name"],
                "metrics": metrics
            }
            with open(report_path, "w", encoding="utf-8") as r_file:
                json.dump(report_data, r_file, ensure_ascii=False, indent=4)
                
            mlflow.log_artifact(str(report_path), artifact_path="evaluation_reports")

    print("\n✅ All 3 Hugging Face model families tracked successfully!")

if __name__ == "__main__":
    run_three_hf_model_families()