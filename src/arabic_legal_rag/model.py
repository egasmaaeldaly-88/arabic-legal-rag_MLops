from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_embedding_model(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)

def build_vector_store(chunks, embeddings, output_dir: str):
    """
    Build a FAISS vector store from document chunks using pre-loaded embeddings and save locally.
    """
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(output_dir)
    return vector_store

def load_vector_store(index_path: str, embeddings):
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )