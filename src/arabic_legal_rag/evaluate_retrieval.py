import os
from loguru import logger
from arabic_legal_rag.retriever import get_legal_retriever

def evaluate_strategies():
    """Evaluate and compare different chunking strategies using a benchmark query."""
    # Test benchmark query targeting Egyptian Civil Code articles on Statute of Limitations
    test_query = "ما هي أحكام التقادم الإسقاطي في العقد؟"
    
    # Paths to the three vector stores generated during grid search
    strategies = {
        "Chunk Size 250 (Overlap 25)": "data/vector_store_sz_250_ov_25",
        "Chunk Size 500 (Overlap 50)": "data/vector_store_sz_500_ov_50",
        "Chunk Size 1000 (Overlap 100)": "data/vector_store_sz_1000_ov_100"
    }

    logger.info(f"Running evaluation benchmark for query: '{test_query}'\n")
    print("=" * 70)
    print(f"BENCHMARK QUERY: {test_query}")
    print("=" * 70)

    for name, path in strategies.items():
        print(f"\n[Strategy: {name}]")
        print(f"Path: {path}")
        try:
            if not os.path.exists(path):
                print("Status: ❌ Vector store not found.")
                continue
            
            retriever = get_legal_retriever(index_path=path, k=2)
            docs = retriever.invoke(test_query)
            
            print(f"Status: ✅ Successfully retrieved {len(docs)} documents.")
            for i, doc in enumerate(docs, 1):
                art_num = doc.metadata.get('article_number', 'N/A')
                snippet = doc.page_content[:120].replace("\n", " ")
                print(f"  - Result {i} | Article: {art_num} | Preview: {snippet}...")
                
        except Exception as e:
            logger.error(f"Failed to evaluate {name}: {e}")
        print("-" * 70)

if __name__ == "__main__":
    evaluate_strategies()