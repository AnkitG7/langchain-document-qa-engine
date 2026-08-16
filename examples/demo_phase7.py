"""Interactive Phase 7 Demonstration Script for DocMind Advanced RAG Architecture.

Run with:
    python examples/demo_phase7.py

Demonstrates:
1. Baseline Vector Search vs. BM25 Sparse Lexical Search
2. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
3. Query Transformations:
   - HyDE (Hypothetical Document Embeddings)
   - Multi-Query Expansion
   - Step-Back Prompting
4. LLM Relevance Reranking (Cross-Encoder style)
5. Contextual Sentence Compression
6. End-to-End Advanced RAG Pipeline with live Gemma & Nomic embeddings
"""

import sys
import os
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from llm.provider import get_chat_model
from ingestion.pipeline import IngestionPipeline
from vectorstore.embedder import get_embeddings
from vectorstore.store import get_or_create_faiss
from rag_advanced.sparse import BM25Index, create_bm25_retriever
from rag_advanced.hybrid import HybridRetriever
from rag_advanced.query_transform import HyDETransformer, MultiQueryTransformer, StepBackTransformer
from rag_advanced.reranker import LLMReranker
from rag_advanced.compression import ContextualCompressor
from rag_advanced.pipeline import AdvancedRAGPipeline

DATA_DIR = Path(__file__).parent.parent / "data"


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def run_demo():
    print_banner("DocMind Phase 7: Advanced RAG Architecture Demo")

    # 1. Ingestion & Indexing
    print_banner("1. Ingesting Documents for Dense & Sparse Hybrid Indexing")
    pipeline = IngestionPipeline(chunk_size=300, chunk_overlap=50)
    chunks, _ = pipeline.run_batch([
        str(DATA_DIR / "sample_doc.txt"),
        str(DATA_DIR / "sample_guide.md"),
        str(DATA_DIR / "sample_data.csv"),
    ])
    print(f"Total Chunks Ingested: {len(chunks)}")

    embedder = get_embeddings()
    dense_store = get_or_create_faiss(documents=chunks, embeddings=embedder)
    dense_retriever = dense_store.as_retriever(search_kwargs={"k": 4})
    sparse_retriever = create_bm25_retriever(documents=chunks, k=4)
    llm = get_chat_model()

    # 2. Dense vs Sparse vs Hybrid RRF
    print_banner("2. Dense Vector vs. Sparse BM25 vs. Hybrid RRF")
    hybrid = HybridRetriever(dense_retriever=dense_retriever, sparse_retriever=sparse_retriever, k=3)
    test_query = "What is the project_name for id 104 in CSV?"

    details = hybrid.retrieve_with_details(test_query)
    print(f"Query: '{test_query}'")
    print(f"  - Dense Retrieved:  {details['dense_count']} chunks")
    print(f"  - Sparse (BM25):    {details['sparse_count']} chunks")
    print(f"  - Hybrid Fused RRF: {details['fused_count']} chunks")
    print(f"Top Fused Result:\n  {details['fused_docs'][0].page_content[:140]}...")

    # 3. Query Transformations
    print_banner("3. Query Transformations (HyDE & Multi-Query)")
    hyde = HyDETransformer(llm=llm)
    hypo_passage = hyde.generate_hypothetical_document("How does chunking work in DocMind?")
    print(f"HyDE Generated Hypothetical Passage:\n  '{hypo_passage[:180]}...'")

    mq = MultiQueryTransformer(llm=llm, num_queries=3)
    expanded = mq.generate_queries("Explain the system architecture")
    print(f"\nMulti-Query Generated Perspectives ({len(expanded)} variants):")
    for q in expanded:
        print(f"  * {q}")

    # 4. LLM Relevance Reranker
    print_banner("4. Cross-Encoder / LLM Relevance Reranking")
    reranker = LLMReranker(llm=llm, top_n=2)
    rerank_query = "What chunking techniques are supported?"
    candidate_docs = dense_retriever.invoke(rerank_query)
    scored = reranker.rerank(rerank_query, candidate_docs, top_n=2)
    print(f"Query: '{rerank_query}'")
    print("Top Reranked Passages:")
    for doc, score in scored:
        src = doc.metadata.get("filename", doc.metadata.get("source", "doc"))
        print(f"  [Relevance Score: {score}/10] ({src}): {doc.page_content[:100]}...")

    # 5. Full Advanced RAG Pipeline
    print_banner("5. Full End-to-End Advanced RAG Execution (Multi-Query + Hybrid RRF + Rerank + Compress)")
    adv_pipeline = AdvancedRAGPipeline(
        dense_retriever=dense_retriever,
        documents=chunks,
        llm=llm,
    )

    q = "What are the core features and chunking methods in DocMind?"
    print(f"[User Query]: {q}")
    res = adv_pipeline.query(question=q, strategy="full_advanced")

    print(f"\n[Strategy]: {res['strategy']}")
    print(f"[Retrieved Chunks]: {res['retrieved_documents_count']}")
    print(f"[DocMind Advanced Answer]:\n{res['answer']}")

    print_banner("Phase 7 Complete!")
    print("HyDE, Multi-Query, Step-Back, BM25, Hybrid RRF, Reranking, and Compression verified successfully.")


if __name__ == "__main__":
    run_demo()
