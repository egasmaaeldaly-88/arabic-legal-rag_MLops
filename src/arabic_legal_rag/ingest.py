import os
import json
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from arabic_legal_rag.utils import load_config
from arabic_legal_rag.model import get_embedding_model

def process_corpus(json_path: str = "data/legal_corpus_cleaned.json", chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    """
    Load the cleaned legal corpus, clean article numbers, and split text into chunks.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_documents = []
    for item in data:
        article_text = item.get("text", "")
        article_num_raw = str(item.get("article_number", ""))
        
        arabic_to_eng = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        clean_raw = article_num_raw.translate(arabic_to_eng)
        
        match = re.search(r'\d+', clean_raw)
        article_num = match.group(0) if match else clean_raw.strip()

        doc = Document(
            page_content=article_text,
            metadata={
                "article_number": article_num,
                "source_document": item.get("source_document", "")
            }
        )
        raw_documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(raw_documents)
    return documents

def build_vector_database():
    """قراءة الـ Config، معالجة النصوص، بناء الـ FAISS Index وحفظه محلياً"""
    config = load_config("configs/config.yaml")
    
    # استخدام دالة التقطيع
    docs = process_corpus(
        json_path=config["data"]["json_path"],
        chunk_size=config["experimentation"]["chunk_sizes"][1],  # أو الحجم الافتراضي
        chunk_overlap=config["experimentation"]["chunk_overlaps"][1]
    )

    print(f"📖 Loaded & split into {len(docs)} chunks. Building vector database...")
    embeddings = get_embedding_model(config["model"]["embedding_model"])
    vector_store = FAISS.from_documents(docs, embeddings)

    # حفظ الفولدر بالمسار الدقيق المطابق للـ config و dvc.yaml
    output_dir = Path(config["vector_db"]["index_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vector_store.save_local(str(output_dir.resolve()))
    print(f"✅ FAISS index successfully saved at '{output_dir.resolve()}'")

if __name__ == "__main__":
    build_vector_database()