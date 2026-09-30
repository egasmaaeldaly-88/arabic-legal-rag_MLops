import json
import re
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from arabic_legal_rag.model import get_embedding_model
from arabic_legal_rag.utils import load_config  # دالة قراءة الإعدادات

# 1. تحميل الإعدادات المركزية من ملف الـ YAML لضمان المرونة والـ MLOps
config = load_config("configs/config.yaml")
model_name = config["model"]["embedding_model"]
index_dir = config["vector_db"]["index_dir"]  # سيقرأ المجلد ديناميكياً (سواء mpnet أو minilm)

# 2. التأكد من إنشاء المجلد برمجياً لضمان عدم حدوث خطأ
os.makedirs(index_dir, exist_ok=True)

# 3. تحميل ملف النصوص القانونية المنظف
json_path = "data/legal_corpus_cleaned.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []
for item in data:
    article_text = item.get("text", "")
    article_num_raw = str(item.get("article_number", ""))
    
    # تحويل الأرقام العربية الهندية (٠١٢٣٤٥٦٧٨٩) إلى أرقام إنجليزية لضمان المطابقة مع الـ Ground Truth
    arabic_to_eng = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    clean_raw = article_num_raw.translate(arabic_to_eng)
    
    # استخراج الأرقام فقط (مثال: تحويل "مادة ١" أو "مادة 558" إلى "558" أو "1")
    match = re.search(r'\d+', clean_raw)
    article_num = match.group(0) if match else clean_raw.strip()

    doc = Document(
        page_content=article_text,
        metadata={
            "article_number": article_num,
            "source_document": item.get("source_document", "")
        }
    )
    documents.append(doc)

# 4. تهيئة نموذج الـ Embeddings ديناميكياً بناءً على ما هو محدد في الـ Config
print(f"Loading embedding model: {model_name}...")
embeddings = get_embedding_model(model_name)

# 5. بناء قاعدة المتجهات (Vector Store) من المستندات
print(f"Building FAISS vector store...")
vector_store = FAISS.from_documents(documents, embeddings)

# 6. حفظ المخرجات في المسار الديناميكي الخاص بالنموذج الحالي
vector_store.save_local(index_dir)
print(f"Vector store successfully created and saved at: {index_dir}")