"""Hybrid Retrieval Engine (Dense Vector + Sparse BM25) with Reciprocal Rank Fusion (RRF).

Demonstrates:
- Dense semantic vector search (Chroma / FAISS) + Sparse lexical keyword search (BM25)
- Reciprocal Rank Fusion (RRF) algorithm: RRF(d) = sum(1 / (k + rank(d)))
- Overcoming keyword blind spots of dense vectors and semantic blind spots of keyword search
"""

from typing import Any, Dict, List, Literal, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.retrievers import BM25Retriever

from ingestion.cleaner import calculate_content_hash


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k_rrf: int = 60,
    top_n: int = 4,
) -> List[Document]:
    """Combines multiple ranked document lists into a single consensus ranking using RRF.

    Formula:
        RRF_Score(doc) = sum_{list in ranked_lists} (1 / (k_rrf + rank_in_list))

    Args:
        ranked_lists: List of ranked Document lists (e.g. [dense_results, sparse_results]).
        k_rrf: Constant dampening factor (default: 60).
        top_n: Number of top fused documents to return.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for doc_list in ranked_lists:
        for rank, doc in enumerate(doc_list, start=1):
            h = calculate_content_hash(doc.page_content)
            if h not in doc_map:
                doc_map[h] = doc
                scores[h] = 0.0

            # Add inverse rank score
            scores[h] += 1.0 / (k_rrf + rank)

    # Sort documents by accumulated RRF score descending
    sorted_hashes = sorted(scores.keys(), key=lambda h: scores[h], reverse=True)
    return [doc_map[h] for h in sorted_hashes[:top_n]]


class HybridRetriever:
    """Hybrid Retriever combining Dense (VectorStore) and Sparse (BM25) retrievers."""

    def __init__(
        self,
        dense_retriever: VectorStoreRetriever,
        sparse_retriever: BM25Retriever,
        k: int = 4,
        k_rrf: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.k = k
        self.k_rrf = k_rrf

    def invoke(self, query: str) -> List[Document]:
        """Executes dense + sparse search and fuses results with Reciprocal Rank Fusion."""
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)

        return reciprocal_rank_fusion(
            ranked_lists=[dense_docs, sparse_docs],
            k_rrf=self.k_rrf,
            top_n=self.k,
        )

    def retrieve_with_details(self, query: str) -> Dict[str, Any]:
        """Returns dense, sparse, and fused results for transparency and analysis."""
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)
        fused_docs = reciprocal_rank_fusion(
            ranked_lists=[dense_docs, sparse_docs],
            k_rrf=self.k_rrf,
            top_n=self.k,
        )

        return {
            "query": query,
            "dense_count": len(dense_docs),
            "sparse_count": len(sparse_docs),
            "fused_count": len(fused_docs),
            "dense_docs": dense_docs,
            "sparse_docs": sparse_docs,
            "fused_docs": fused_docs,
        }
