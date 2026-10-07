import json
import os
import time
from pathlib import Path
import mlflow
import numpy as np
import optuna
from arabic_legal_rag.eval import compute_metrics, normalize_digits, extract_article_num
from arabic_legal_rag.utils import load_config


optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    top_k = trial.suggest_int("top_k", 2, 8)
    
    with mlflow.start_run(run_name=f"trial_{trial.number}_top{top_k}", nested=True) as nested_run:
        # الـ Tags والـ Params الأساسية
        mlflow.set_tag("data_version", "v1.0_json")
        mlflow.set_tag("git_commit", "main_v2")
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("trial_number", trial.number)
        
        eval_path = Path("data/eval_dataset.json")
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_dataset = json.load(f)
            
        metrics = compute_metrics(eval_dataset, top_k=top_k)
        
        # تسجيل المقاييس الأصلية
        mlflow.log_metric("hit_rate_at_k", metrics["hit_rate_at_k"])
        mlflow.log_metric("mrr_at_k", metrics["mrr_at_k"])
        mlflow.log_metric("avg_latency_sec", metrics["avg_latency_sec"])
        
        # 
        mae_proxy = max(0.0, 1.0 - metrics["hit_rate_at_k"])
        mlflow.log_metric("mae", mae_proxy)
        
        return metrics["hit_rate_at_k"]
def run_optuna_sweep():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("arabic-legal-rag-optuna-sweep")
    
    config = load_config("configs/config.yaml")
    model_name = config["model"]["embedding_model"]
    
   
    with mlflow.start_run(run_name="parent_optuna_sweep_run", log_system_metrics=True) as parent_run:
        mlflow.set_tag("author", "Asmaa")
        mlflow.set_tag("framework", "optuna_mlflow_langchain")
        mlflow.set_tag("git_commit", "main_v2")
        mlflow.log_param("embedding_model", model_name)
        
        # إنشاء وعمل Optuna Study لـ 10 تجارب على الأقل
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        
        # تسجيل أفضل النتائج في الـ Parent Run
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_hit_rate", study.best_value)

       
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        best_params_path = output_dir / "optuna_best_params.json"
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, ensure_ascii=False, indent=4)

        print("\n--- Optuna Hyperparameter Sweep Completed ---")
        print(f"Best Parameters: {study.best_params}")
        print(f"Best Hit Rate: {study.best_value:.4f}")
        print(f"Saved best params to {best_params_path}")
      

if __name__ == "__main__":
    run_optuna_sweep()