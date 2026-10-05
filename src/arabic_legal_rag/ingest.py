import os
import json
from pathlib import Path
from langchain_core.documents import Document
from arabic_legal_rag.utils import load_config
from arabic_legal_rag.model import get_embedding_model
from langchain_community.vectorstores import FAISS

def process_corpus(json_path: str = "data/legal_corpus_structured.json") -> list[Document]:
    """
    Load the structured legal corpus, preserving each legal article as a self-contained 
    unit of meaning (Chunk by article, not by arbitrary token count).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        article_text = item.get("ar_text", "")
        article_num = item.get("article_number", 0)
        
        # Skip empty texts safely
        if not article_text.strip():
            continue

        # Create a LangChain Document where the entire article is a single clean chunk
        doc = Document(
            page_content=article_text,
            metadata={
                "article_number": article_num,
                "book": item.get("book", ""),
                "chapter": item.get("chapter", ""),
                "section": item.get("section", ""),
                "topic": item.get("topic", ""),
                "is_repealed": item.get("is_repealed", False),
                "source_page": item.get("source_page", 0),
                "citation": item.get("citation", f"Egyptian Civil Code, Article {article_num}")
            }
        )
        documents.append(doc)

    return documents

def build_vector_database():
    """
    Load configuration, process the corpus by article, build the FAISS index, 
    and save it locally.
    """
    config = load_config("configs/config.yaml")
    
    docs = process_corpus(
        json_path=config["data"].get("structured_json_path", "data/legal_corpus_structured.json")
    )

    print(f"📖 Loaded {len(docs)} legal articles as clean chunks. Building vector database...")
    
    # Get embedding model and build FAISS vector store
    embeddings = get_embedding_model(config["model"]["embedding_model"])
    vector_store = FAISS.from_documents(docs, embeddings)

    # Save the vector store index locally
    output_dir = Path(config["vector_db"]["index_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vector_store.save_local(str(output_dir.resolve()))
    print(f"✅ FAISS index successfully saved at '{output_dir.resolve()}'")
    
if __name__ == "__main__":
    build_vector_database()