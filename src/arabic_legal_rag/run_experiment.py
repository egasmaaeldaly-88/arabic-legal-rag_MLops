import os
import time
import psutil
import mlflow
from loguru import logger
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from arabic_legal_rag.utils import load_config, load_clean_corpus

def build_vector_store_pipeline():
    """Build the FAISS vector store using structural article-based chunking and log metrics to MLflow."""
    # Load configuration
    config = load_config()
    data_path = config["data"]["json_path"]
    model_name = config["model"]["embedding_model"]
    index_dir = config["vector_db"]["index_dir"]
    experiment_name = config.get("mlflow", {}).get("experiment_name", "arabic-legal-rag-retrieval")
    
    # Set MLflow experiment name
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow Experiment set to: {experiment_name}")

    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Load clean corpus (each item represents a complete legal article)
    corpus = load_clean_corpus(data_path)
    logger.info(f"Loaded {len(corpus)} legal articles from corpus.")

    run_name = f"run_article_based_{model_name.split('/')[-1]}"

    with mlflow.start_run(run_name=run_name):
        logger.info(f"=== Starting Article-Based Ingestion Run: {run_name} ===")
        
        # 1. Log Parameters
        mlflow.log_param("embedding_model", model_name)
        mlflow.log_param("chunking_strategy", "article_based_structural")
        mlflow.log_param("total_documents", len(corpus))

        # Monitor system performance
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # in MB

        start_time = time.time()

        # 2. Structural Document Creation (One Document per Legal Article)
        docs = [
            Document(
                page_content=item["text"],
                metadata={
                    "article_number": str(item.get("article_number", "")),
                    "chapter": item.get("chapter", ""),
                    "section": item.get("section", "")
                }
            )
            for item in corpus
        ]

        # 3. Vector Store Creation & Saving Local Index
        logger.info("Generating embeddings and building FAISS vector store...")
        vector_store = FAISS.from_documents(docs, embeddings)
        
        elapsed_time = time.time() - start_time
        mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
        mem_consumed = mem_after - mem_before

        # Save local vector store index directory
        os.makedirs(index_dir, exist_ok=True)
        vector_store.save_local(index_dir)
        logger.info(f"Vector store successfully saved to {index_dir}")

        # 4. Log Metrics
        mlflow.log_metric("total_articles_indexed", len(docs))
        mlflow.log_metric("indexing_time_sec", round(elapsed_time, 2))
        mlflow.log_metric("memory_consumed_mb", round(mem_consumed, 2))

        logger.info(f"Successfully finished and logged Article-Based Ingestion Run.")

if __name__ == "__main__":
    build_vector_store_pipeline()