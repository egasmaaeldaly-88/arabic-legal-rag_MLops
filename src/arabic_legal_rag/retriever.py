import os
from loguru import logger
import yaml

# Import core functions from model.py to ensure compatibility
from arabic_legal_rag.model import get_embedding_model, load_vector_store

def load_config():
    """Load project settings from the configuration file inside configs folder."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Dynamically locate config.yaml inside the configs folder relative to src/arabic_legal_rag/
    config_path = os.path.abspath(os.path.join(current_dir, "..", "..", "configs", "config.yaml"))
    
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def get_legal_retriever(index_path: str, k: int = 3):
    """
    Initialize the search retriever:
    Loads the embedding model once, reads the vector store from the specified path,
    and prepares it for querying.
    """
    config = load_config()
    model_name = config.get("model", {}).get("embedding_model", "BAAI/bge-m3")

    # 1. Load the embedding model once (saves memory and CPU)
    logger.info(f"Loading embedding model for retrieval: {model_name}")
    embeddings = get_embedding_model(model_name)

    # 2. Ensure the vector store directory exists
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Vector store not found at path: {index_path}")
    
    # 3. Load the saved vector store from experiments
    logger.info(f"Loading FAISS vector store from: {index_path}")
    vector_store = load_vector_store(index_path, embeddings)

    # 4. Convert the vector store into a retriever fetching top k results
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    return retriever

if __name__ == "__main__":
    # Use the article-based vector store path configured or generated
    config = load_config()
    test_index_path = config.get("vector_db", {}).get("index_dir", "data/vector_store_bgem3")
    
    # A real legal query to test article-based retrieval accuracy
    test_query = "ما هي أحكام التقادم الإسقاطي في العقد؟"
    
    try:
        logger.info("Initializing the Retriever...")
        
        # Invoke the retriever to fetch top 3 legal articles
        retriever = get_legal_retriever(index_path=test_index_path, k=3)
        
        logger.info(f"Searching for query: '{test_query}'")
        relevant_docs = retriever.invoke(test_query)
        
        print("\n" + "="*60)
        print(f"Top Retrieved Legal Articles ({len(relevant_docs)} results):")
        print("="*60)
        
        for i, doc in enumerate(relevant_docs, 1):
            # Extract the article number from Metadata (stored during ingest)
            article_num = doc.metadata.get('article_number', 'Not Specified')
            print(f"\n[Result {i}] Article Number: {article_num}")
            print(f"Content: {doc.page_content}")
            print("-" * 60)

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")