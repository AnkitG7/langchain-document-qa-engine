"""Vector store and embeddings module for DocMind.

Supports pluggable dedicated embedding models and zero-Docker local vector databases (Chroma and FAISS).
"""

from .embedder import (
    get_embeddings,
    get_fake_embeddings,
    verify_embeddings,
    BaseEmbeddings,
)
from .store import (
    create_vector_store,
    get_or_create_chroma,
    get_or_create_faiss,
    VectorStoreManager,
    delete_documents_by_fingerprint,
    replace_document_vectors,
)
from .retriever import (
    create_retriever,
    similarity_search_with_scores,
    mmr_search,
    threshold_search,
)

__all__ = [
    # Embeddings
    "get_embeddings",
    "get_fake_embeddings",
    "verify_embeddings",
    "BaseEmbeddings",
    # Stores
    "create_vector_store",
    "get_or_create_chroma",
    "get_or_create_faiss",
    "VectorStoreManager",
    "delete_documents_by_fingerprint",
    "replace_document_vectors",
    # Retrievers
    "create_retriever",
    "similarity_search_with_scores",
    "mmr_search",
    "threshold_search",
]
