import json
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_corpus(json_path: str = "data/legal_corpus_cleaned.json", chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    """
    Load the cleaned legal corpus, clean article numbers, and split text into chunks.
    Does NOT load the heavy embedding model here, making iterations lightning fast.
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