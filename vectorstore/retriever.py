"""Retrieval modes, search strategies, and filtering utilities for DocMind.

Demonstrates:
- Standard Similarity Search (k-NN)
- Maximal Marginal Relevance (MMR) for diversity
- Similarity Score Threshold filtering
- Structured metadata filtering
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever


def create_retriever(
    vectorstore: VectorStore,
    search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "similarity",
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    score_threshold: Optional[float] = None,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> VectorStoreRetriever:
    """Configures and returns a LangChain VectorStoreRetriever.

    Args:
        vectorstore: Instantiated vector store (Chroma or FAISS).
        search_type: 'similarity', 'mmr', or 'similarity_score_threshold'.
        k: Number of relevant documents to return.
        fetch_k: Candidate pool size for MMR reranking.
        lambda_mult: MMR diversity factor (0.0 = maximal diversity, 1.0 = maximal relevance).
        score_threshold: Minimum similarity score threshold for score_threshold search.
        filter_dict: Metadata filter criteria (e.g. {'file_type': 'pdf', 'doc_id': '...'}).
    """
    search_kwargs: Dict[str, Any] = {"k": k}

    if filter_dict:
        search_kwargs["filter"] = filter_dict

    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k
        search_kwargs["lambda_mult"] = lambda_mult

    elif search_type == "similarity_score_threshold":
        if score_threshold is None:
            score_threshold = 0.5
        search_kwargs["score_threshold"] = score_threshold

    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def similarity_search_with_scores(
    vectorstore: VectorStore,
    query: str,
    k: int = 4,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Document, float]]:
    """Performs direct similarity search returning matching Documents alongside their similarity scores."""
    if not query.strip():
        return []

    kwargs: Dict[str, Any] = {"k": k}
    if filter_dict:
        kwargs["filter"] = filter_dict

    try:
        return vectorstore.similarity_search_with_score(query, **kwargs)
    except Exception:
        # Fallback if specific vector store implementation lacks filter in with_score
        docs = vectorstore.similarity_search(query, k=k)
        return [(doc, 1.0) for doc in docs]


def mmr_search(
    vectorstore: VectorStore,
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Performs Maximal Marginal Relevance (MMR) search to return a balanced, non-redundant set of documents."""
    if not query.strip():
        return []

    kwargs: Dict[str, Any] = {
        "k": k,
        "fetch_k": fetch_k,
        "lambda_mult": lambda_mult,
    }
    if filter_dict:
        kwargs["filter"] = filter_dict

    return vectorstore.max_marginal_relevance_search(query, **kwargs)


def threshold_search(
    vectorstore: VectorStore,
    query: str,
    score_threshold: float = 0.6,
    k: int = 4,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Retrieves documents only if their similarity score exceeds the specified threshold."""
    if not query.strip():
        return []

    retriever = create_retriever(
        vectorstore=vectorstore,
        search_type="similarity_score_threshold",
        score_threshold=score_threshold,
        k=k,
        filter_dict=filter_dict,
    )
    return retriever.invoke(query)
