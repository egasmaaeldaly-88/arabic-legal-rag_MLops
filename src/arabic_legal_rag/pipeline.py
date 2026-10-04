import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from arabic_legal_rag.utils import load_config, load_clean_corpus
from arabic_legal_rag.model import get_embedding_model, load_vector_store

def build_vector_database(config_path: str = "configs/config.yaml"):
    """قراءة البيانات وبناء الـ FAISS Vector Database والتخزين"""
    config = load_config(config_path)
    corpus = load_clean_corpus(config["data"]["json_path"])

    documents = []
    for item in corpus:
        text = item.get("text", "").strip()
        if text:
            doc = Document(
                page_content=text,
                metadata={"article_number": item.get("article_number", "N/A")}
            )
            documents.append(doc)

    print(f"📖 Loaded {len(documents)} legal articles. Building vector database...")
    embeddings = get_embedding_model(config["model"]["embedding_model"])
    vector_store = FAISS.from_documents(documents, embeddings)

    # التعديل الصحيح لضمان إنشاء الفولدر بالكامل
    os.makedirs(config["vector_db"]["index_dir"], exist_ok=True)
    vector_store.save_local(config["vector_db"]["index_dir"])
    print(f"✅ FAISS index saved successfully at '{config['vector_db']['index_dir']}'")

def run_retrieval(query: str, config_path: str = "configs/config.yaml"):
    """تشغيل الاسترجاع للمواد القانونية بناءً على الاستعلام"""
    config = load_config(config_path)
    embeddings = get_embedding_model(config["model"]["embedding_model"])
    vector_store = load_vector_store(config["vector_db"]["index_dir"], embeddings)

    return vector_store.similarity_search_with_score(
        query, k=config["model"]["top_k"]
    )

if __name__ == "__main__":
    build_vector_database()