# src/arabic_legal_rag/evaluate.py

import argparse
import json
import logging
from pathlib import Path

import mlflow
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """الدالة الرئيسية التي يستدعيها dvc"""
    
    parser = argparse.ArgumentParser(description="Evaluate RAG model")
    parser.add_argument(
        "--eval_dataset_path",
        default="data/eval_dataset.json",
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Number of questions to evaluate"
    )
    parser.add_argument(
        "--experiment_name",
        default="arabic-legal-rag",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--tracking_uri",
        default="http://localhost:5000",
        help="MLflow tracking URI"
    )
    
    args = parser.parse_args()
    
    # ✅ شغّل التقييم
    metrics = run_evaluation(
        eval_dataset_path=args.eval_dataset_path,
        num_questions=args.num_questions,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri
    )
    
    # ✅ احفظ الـ metrics في JSON (DVC يراقبه)
    output_metrics_file = Path("outputs/evaluation_metrics.json")
    output_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Metrics saved to {output_metrics_file}")
    
    # احفظ النتائج المفصلة أيضاً
    detailed_file = Path("outputs/ragas_detailed_results.json")
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "experiment": args.experiment_name
        }, f, ensure_ascii=False, indent=2)
    
    return metrics

def run_evaluation(
    eval_dataset_path: str = "data/eval_dataset.json",
    num_questions: int = 10,
    chunk_size: int = 500,
    overlap: int = 50,
    top_k: int = 5,
    experiment_name: str = "arabic-legal-rag",
    tracking_uri: str = "http://localhost:5000"
) -> dict:
    """
    تشغيل تقييم RAGAS وتسجيله في MLflow
    """
    
    # 1. إعداد MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"Experiment: {experiment_name}")
    
    # 2. حمّل eval dataset
    if not Path(eval_dataset_path).exists():
        logger.error(f"Eval dataset not found: {eval_dataset_path}")
        return {"error": "Dataset not found"}
    
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)
    
    eval_dataset = eval_dataset[:num_questions]
    logger.info(f"Loaded {len(eval_dataset)} questions for evaluation")
    
    # 3. ابدأ MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")
        
        # ✅ سجّل المعاملات
        mlflow.log_params({
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
            "num_questions": num_questions,
            "experiment_name": experiment_name
        })
        logger.info("Logged parameters")
        
        # 4. استخرج البيانات من eval dataset
        questions = [item["question"] for item in eval_dataset]
        contexts = [[c] for c in [item.get("context", "") for item in eval_dataset]]  # ✅ تنسيق صحيح
        ground_truth = [item["answer"] for item in eval_dataset]
        
        # 5. شغّل RAGAS
        logger.info("Running RAGAS evaluation...")
        dataset_dict = {
            "question": questions,
            "contexts": contexts,
            "ground_truth": ground_truth
        }
        
        try:
            result = ragas_evaluate(
                dataset_dict,
                metrics=[
                    faithfulness(),
                    answer_relevancy(),
                    context_precision()
                ],
                llm_client="groq",
                embedding_model="bge-m3"
            )
            
            # 6. استخرج النتائج
            eval_metrics = {
                "ragas_faithfulness": float(result["faithfulness"]),
                "ragas_relevance": float(result["relevance"]),
                "ragas_context_precision": float(result["context_precision"])
            }
            
            logger.info(f"✅ RAGAS Results: {eval_metrics}")
            
            # ✅ سجّل المقاييس في MLflow
            mlflow.log_metrics(eval_metrics)
            logger.info("Logged metrics to MLflow")
            
            # 7. اطبع النتائج
            print("\n" + "="*60)
            print(f"✅ Evaluation Complete - Run ID: {run_id}")
            print("="*60)
            for metric, value in eval_metrics.items():
                print(f"  {metric}: {value:.4f}")
            print("="*60 + "\n")
            
            return eval_metrics
            
        except Exception as e:
            logger.exception("RAGAS evaluation failed")
            mlflow.log_param("error", str(e))
            return {
                "error": str(e),
                "ragas_faithfulness": 0.0,
                "ragas_relevance": 0.0,
                "ragas_context_precision": 0.0
            }

if __name__ == "__main__":
    main()