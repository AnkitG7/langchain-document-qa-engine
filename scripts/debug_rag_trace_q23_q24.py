"""Debug and Trace Layer-by-Layer Retrieval for Q23 and Q24 on Attention Is All You Need.

Inspects:
1. Dense FAISS top-k
2. BM25 top-k
3. RRF Fused ranking
4. Reranker / Context Compression output
5. Final Prompt Context
6. Evaluates Retrieval Recall vs Reranker Recall vs Generation Reasoning
"""

import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.multimodal_pipeline import MultimodalIngestionPipeline
from vectorstore.embedder import get_embeddings
from vectorstore.store import get_or_create_faiss
from rag_advanced.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag_advanced.reranker import LLMReranker
from rag_advanced.compression import ContextualCompressor
from rag_advanced.pipeline import AdvancedRAGPipeline
from llm.provider import get_chat_model

PDF_PATH = "data/real_pdfs/transformer_paper.pdf"
INDEX_PATH = "data/faiss_transformer_benchmark"

DEBUG_QUESTIONS = [
    {
        "id": 23,
        "question": "Why are residual connections around each sublayer (LayerNorm(x + Sublayer(x))) essential in the 6-layer Transformer stack?",
        "target_evidence": ["LayerNorm(x + Sublayer(x))", "residual connections", "d_model = 512"],
    },
    {
        "id": 24,
        "question": "If you process a sequence of length n = 1000, what is the primary computational bottleneck in the Transformer and why?",
        "target_evidence": ["O(n^2 * d)", "Complexity per Layer", "restricted to considering only a neighborhood of size r", "very long sequences"],
    },
]


def trace_pipeline():
    print("=" * 80)
    print("  LAYER-BY-LAYER RAG RETRIEVAL & REASONING DIAGNOSTIC (Q23 & Q24)")
    print("=" * 80)

    # 1. Ingest & Load Store
    embedder = get_embeddings()
    pipeline = MultimodalIngestionPipeline(chunk_size=600, chunk_overlap=100, enable_vision_processing=False)
    docs, report = pipeline.ingest_pdf(PDF_PATH)

    dense_store = get_or_create_faiss(documents=docs, embeddings=embedder, index_path=INDEX_PATH)
    dense_retriever = dense_store.as_retriever(search_kwargs={"k": 6})
    llm = get_chat_model()

    from rag_advanced.sparse import create_bm25_retriever

    sparse_retriever = create_bm25_retriever(docs, k=6)
    hybrid = HybridRetriever(dense_retriever=dense_retriever, sparse_retriever=sparse_retriever, k=6)
    reranker = LLMReranker(llm=llm, top_n=4)

    for item in DEBUG_QUESTIONS:
        qid = item["id"]
        q = item["question"]
        targets = item["target_evidence"]

        print("\n" + "#" * 80)
        print(f"  DIAGNOSING QUESTION Q{qid}: {q}")
        print("#" * 80)

        # Step 1: Dense Retrieval
        dense_docs = dense_retriever.invoke(q)
        print(f"\n[1. DENSE FAISS RETRIEVAL (k={len(dense_docs)})]:")
        for i, d in enumerate(dense_docs, 1):
            p = d.metadata.get("page", "N/A")
            etype = d.metadata.get("element_type", "text")
            snip = d.page_content[:120].encode("ascii", "replace").decode("ascii").replace("\n", " ")
            has_target = any(t.lower() in d.page_content.lower() for t in targets)
            print(f"  {i}. Page {p:2} | Type: {etype:6} | Match: {has_target} | {snip}...")

        # Step 2: Sparse BM25 Retrieval
        sparse_docs = hybrid.sparse_retriever.invoke(q)
        print(f"\n[2. SPARSE BM25 RETRIEVAL (k={len(sparse_docs)})]:")
        for i, d in enumerate(sparse_docs, 1):
            p = d.metadata.get("page", "N/A")
            etype = d.metadata.get("element_type", "text")
            snip = d.page_content[:120].encode("ascii", "replace").decode("ascii").replace("\n", " ")
            has_target = any(t.lower() in d.page_content.lower() for t in targets)
            print(f"  {i}. Page {p:2} | Type: {etype:6} | Match: {has_target} | {snip}...")

        # Step 3: Reciprocal Rank Fusion (RRF)
        fused_docs = reciprocal_rank_fusion([dense_docs, sparse_docs], top_n=6)
        print(f"\n[3. RRF FUSED RANKING (top_n={len(fused_docs)})]:")
        for i, d in enumerate(fused_docs, 1):
            p = d.metadata.get("page", "N/A")
            etype = d.metadata.get("element_type", "text")
            snip = d.page_content[:120].encode("ascii", "replace").decode("ascii").replace("\n", " ")
            has_target = any(t.lower() in d.page_content.lower() for t in targets)
            print(f"  {i}. Page {p:2} | Type: {etype:6} | Match: {has_target} | {snip}...")

        # Step 4: Reranker Output
        reranked_docs = reranker.compress_documents(q, fused_docs, top_n=4)
        print(f"\n[4. RERANKED CONTEXT (top_n={len(reranked_docs)})]:")
        for i, d in enumerate(reranked_docs, 1):
            p = d.metadata.get("page", "N/A")
            etype = d.metadata.get("element_type", "text")
            snip = d.page_content[:120].encode("ascii", "replace").decode("ascii").replace("\n", " ")
            has_target = any(t.lower() in d.page_content.lower() for t in targets)
            print(f"  {i}. Page {p:2} | Type: {etype:6} | Match: {has_target} | {snip}...")

        # Step 5: Full Generation
        rag = AdvancedRAGPipeline(dense_retriever=dense_retriever, documents=docs, llm=llm)
        res = rag.query(question=q, strategy="hybrid_rrf")
        print(f"\n[5. FINAL GENERATED ANSWER]:\n{res['answer']}")


if __name__ == "__main__":
    trace_pipeline()
