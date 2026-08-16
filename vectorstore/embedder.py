"""Pluggable Dedicated Embedding Models for DocMind.

Demonstrates:
- Dedicated embedding model architecture (distinct from reasoning LLMs)
- OllamaEmbeddings (nomic-embed-text / bge-small)
- OpenAIEmbeddings & HuggingFaceEmbeddings
- Deterministic FakeEmbeddings for reproducible offline testing
- Embedding health verification utility
"""

import math
from typing import Any, Dict, List, Literal, Optional
from langchain_core.embeddings import Embeddings, DeterministicFakeEmbedding

from config import settings


class BaseEmbeddings(Embeddings):
    """Abstract base type hint for embedding models."""
    pass


class ConsistentFakeEmbeddings(Embeddings):
    """Deterministic, lightweight fake embeddings based on text hash vectors.

    Produces consistent 384-dimensional unit vectors without requiring network or heavy ML models.
    """

    def __init__(self, size: int = 384):
        self.size = size

    def _embed_string(self, text: str) -> List[float]:
        # Generate deterministic vector from text characters
        vec = [0.0] * self.size
        if not text:
            return vec

        for idx, char in enumerate(text):
            pos = (ord(char) * 31 + idx) % self.size
            vec[pos] += 1.0

        # Normalize to unit length (L2 norm)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_string(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_string(text)


def get_fake_embeddings(size: int = 384) -> Embeddings:
    """Returns a deterministic fake embedding model for zero-dependency offline testing."""
    return ConsistentFakeEmbeddings(size=size)


def get_embeddings(
    provider: Optional[Literal["ollama", "openai", "huggingface", "fake"]] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> Embeddings:
    """Factory to retrieve a dedicated embedding model.

    Args:
        provider: 'ollama', 'openai', 'huggingface', or 'fake'. Defaults to settings.default_embedding_provider.
        model_name: Specific model identifier (e.g. 'nomic-embed-text', 'text-embedding-3-small').
        **kwargs: Additional provider-specific kwargs.

    Returns:
        Embeddings: LangChain Embeddings instance.
    """
    selected_provider = provider or settings.default_embedding_provider

    if selected_provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            try:
                from langchain_community.embeddings import OllamaEmbeddings
            except ImportError:
                raise ImportError("Please install langchain-ollama: `pip install langchain-ollama`")

        selected_model = model_name or settings.ollama_embedding_model or "nomic-embed-text"
        return OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=selected_model,
            **kwargs,
        )

    elif selected_provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError("Please install langchain-openai: `pip install langchain-openai`")

        api_key = settings.openai_api_key
        selected_model = model_name or settings.openai_embedding_model or "text-embedding-3-small"
        return OpenAIEmbeddings(
            model=selected_model,
            api_key=api_key if api_key else None,
            **kwargs,
        )

    elif selected_provider == "huggingface":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError("Please install sentence-transformers: `pip install sentence-transformers`")

        selected_model = model_name or settings.huggingface_embedding_model or "BAAI/bge-small-en-v1.5"
        return HuggingFaceEmbeddings(model_name=selected_model, **kwargs)

    elif selected_provider == "fake":
        return get_fake_embeddings()

    else:
        raise ValueError(
            f"Unsupported embedding provider '{selected_provider}'. "
            "Supported: 'ollama', 'openai', 'huggingface', 'fake'."
        )


def verify_embeddings(embedder: Embeddings) -> Dict[str, Any]:
    """Runs a live verification check on an embedding model to validate vector generation and dimensions."""
    sample_query = "DocMind intelligent vector search test"
    sample_docs = ["First test document chunk.", "Second test document chunk."]

    query_vec = embedder.embed_query(sample_query)
    doc_vecs = embedder.embed_documents(sample_docs)

    dimensions = len(query_vec)
    if dimensions == 0:
        raise ValueError("Embedding model returned empty 0-dimensional vector!")

    for idx, dvec in enumerate(doc_vecs):
        if len(dvec) != dimensions:
            raise ValueError(
                f"Dimension mismatch in document {idx}: expected {dimensions}, got {len(dvec)}"
            )

    return {
        "status": "healthy",
        "dimensions": dimensions,
        "query_vector_length": len(query_vec),
        "documents_tested": len(doc_vecs),
    }
