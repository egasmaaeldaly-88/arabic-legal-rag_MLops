from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_embedding_model(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)

def load_vector_store(index_path: str, embeddings):
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )