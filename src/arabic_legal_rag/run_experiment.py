import os
import time
import psutil
import mlflow
from loguru import logger
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.arabic_legal_rag.utils import load_config, load_clean_corpus

def run_mlflow_experiment():
    # Load configuration
    config = load_config()
    data_path = config["data"]["json_path"]
    model_name = config["model"]["embedding_model"]
    
    # Define grid search parameters to test
    grid_params = [
        {"chunk_size": 250, "chunk_overlap": 25},
        {"chunk_size": 500, "chunk_overlap": 50},
        {"chunk_size": 1000, "chunk_overlap": 100},
    ]

    # Set MLflow experiment name once
    experiment_name = "Arabic_Legal_RAG_Chunking_Optimization"
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow Experiment set to: {experiment_name}")

    # Initialize embeddings model once (Singleton approach)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Load corpus
    corpus = load_clean_corpus(data_path)

    # Test queries for model retrieval metrics
    test_queries = [
        "ما هي أحكام التزام المدين بتنفيذ الالتزام عينا؟",
        "ما هي شروط المسؤولية العقدية في القانون المدني؟"
    ]

    for params in grid_params:
        chunk_size = params["chunk_size"]
        chunk_overlap = params["chunk_overlap"]
        run_name = f"chunk_sz_{chunk_size}_ov_{chunk_overlap}"

        # Start MLflow run explicitly under the experiment
        with mlflow.start_run(run_name=run_name):
            logger.info(f"=== Starting Run: {run_name} ===")
            
            # 1. Log Parameters
            mlflow.log_param("chunk_size", chunk_size)
            mlflow.log_param("chunk_overlap", chunk_overlap)
            mlflow.log_param("embedding_model", model_name)
            mlflow.log_param("total_documents", len(corpus))

            # Monitor system performance
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 * 1024) # in MB

            start_time = time.time()

            # 2. Text Splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            docs = text_splitter.create_documents(
                texts=[item["text"] for item in corpus],
                metadatas=[{"article_number": item.get("article_number", "")} for item in corpus]
            )

            # 3. Vector Store Creation & Saving Local Index
            vector_store = FAISS.from_documents(docs, embeddings)
            
            elapsed_time = time.time() - start_time
            mem_after = process.memory_info().rss / (1024 * 1024) # in MB
            mem_consumed = mem_after - mem_before

            index_path = f"data/vector_store_sz_{chunk_size}_ov_{chunk_overlap}"
            vector_store.save_local(index_path)

            # 4. Evaluate Retrieval Metrics
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            retrieval_start = time.time()
            total_retrieved_docs = 0
            for q in test_queries:
                retrieved = retriever.invoke(q)
                total_retrieved_docs += len(retrieved)
            avg_retrieval_latency = (time.time() - retrieval_start) / len(test_queries)

            # 5. Log Metrics
            mlflow.log_metric("total_chunks", len(docs))
            mlflow.log_metric("indexing_time_sec", round(elapsed_time, 2))
            mlflow.log_metric("memory_consumed_mb", round(mem_consumed, 2))
            mlflow.log_metric("avg_retrieval_latency_sec", round(avg_retrieval_latency, 4))
            mlflow.log_metric("avg_retrieved_docs_per_query", total_retrieved_docs / len(test_queries))

            # 6. Log Unique Artifacts per Run (with safe cleanup)
            artifact_file = f"run_info_sz_{chunk_size}_ov_{chunk_overlap}.txt"
            with open(artifact_file, "w", encoding="utf-8") as f:
                f.write(
                    f"Experiment: Arabic Legal RAG Optimization\n"
                    f"Run Name: {run_name}\n"
                    f"Chunk Size: {chunk_size}\n"
                    f"Chunk Overlap: {chunk_overlap}\n"
                    f"Total Chunks Generated: {len(docs)}\n"
                    f"Indexing Time (seconds): {round(elapsed_time, 2)}\n"
                    f"Memory Consumed (MB): {round(mem_consumed, 2)}\n"
                    f"Vector Index Saved Path: {index_path}\n"
                )
            
            # Log artifact to MLflow
            mlflow.log_artifact(artifact_file)
            
            # Safe local cleanup after logging
            if os.path.exists(artifact_file):
                try:
                    os.remove(artifact_file)
                except Exception as e:
                    logger.warning(f"Could not remove temp file: {e}")

            logger.info(f"Successfully finished and logged Run {run_name}")

if __name__ == "__main__":
    run_mlflow_experiment()