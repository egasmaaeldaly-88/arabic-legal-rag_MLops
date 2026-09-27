"""Arabic Legal RAG Package.

A production-grade RAG pipeline for Egyptian Civil Code retrieval.
"""

from arabic_legal_rag.model import get_embedding_model, load_vector_store
from arabic_legal_rag.pipeline import build_vector_database, run_retrieval
from arabic_legal_rag.utils import load_clean_corpus, load_config

__version__ = "0.1.0"

__all__ = [
    "load_config",
    "load_clean_corpus",
    "get_embedding_model",
    "load_vector_store",
    "build_vector_database",
    "run_retrieval",
]