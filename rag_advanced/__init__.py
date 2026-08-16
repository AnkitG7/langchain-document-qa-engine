"""Advanced RAG Module for DocMind.

Demonstrates:
- Query Transformations: Multi-Query, HyDE, Step-Back Prompting
- Sparse Lexical Retrieval: BM25
- Hybrid Retrieval: Dense + Sparse with Reciprocal Rank Fusion (RRF) and Weighted Fusion
- Reranking: Cross-Encoder / LLM Relevance Scoring
- Contextual Compression: Sentence-level extraction and fluff reduction
- End-to-End Advanced RAG Pipeline
"""

from .query_transform import (
    HyDETransformer,
    MultiQueryTransformer,
    StepBackTransformer,
)
from .sparse import BM25Index, create_bm25_retriever
from .hybrid import HybridRetriever, reciprocal_rank_fusion
from .reranker import LLMReranker
from .compression import ContextualCompressor
from .pipeline import AdvancedRAGPipeline

__all__ = [
    # Query Transformations
    "HyDETransformer",
    "MultiQueryTransformer",
    "StepBackTransformer",
    # Sparse & Hybrid Retrieval
    "BM25Index",
    "create_bm25_retriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    # Reranking & Compression
    "LLMReranker",
    "ContextualCompressor",
    # Pipeline
    "AdvancedRAGPipeline",
]
