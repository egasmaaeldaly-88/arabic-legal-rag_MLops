# src/experiments/run_experiment.py
import json
from pathlib import Path
from embedding.indexer import create_index
import mlflow

def run_experiment(
    corpus_path: str = "data/corpus.json",
    chunk_size: int = 500,
    overlap: int = 50,
    top_k: int = 5,
    model_name: str = "bge-m3"
):
    """
    تشغيل تجربة كاملة مع تسجيل المعاملات
    """
    with mlflow.start_run():
        # 1. سجّل المعاملات
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("overlap", overlap)
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("model_name", model_name)
        
        # 2. اقرأ الـ corpus
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        
        # 3. تحقق من الحقول الصحيحة
        # اختر بين text_ar أو text_en حسب اللغة
        sample = corpus[0]
        text_field = "text_ar" if "text_ar" in sample else "text_en"
        
        # 4. أعد الـ index
        index, faiss_index = create_index(
            corpus=corpus,
            text_field=text_field,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            model_name=model_name
        )
        
        # 5. سجّل الإحصائيات
        mlflow.log_metric("corpus_size", len(corpus))
        mlflow.log_metric("index_size", len(faiss_index.index_data))
        
        return index

if __name__ == "__main__":
    run_experiment(chunk_size=500, overlap=50)