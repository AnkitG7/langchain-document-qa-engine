"""Baseline Text-Only RAG Test on Tesla 2024 Shareholder Report.

Tests:
1. Text question (Revenue)
2. Table question (Vehicle Deliveries)
3. Visual question (Delivery & Production Chart)
Using existing standard PyPDF text extraction.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.pipeline import IngestionPipeline
from vectorstore.embedder import get_embeddings
from vectorstore.store import get_or_create_faiss
from rag_advanced.pipeline import AdvancedRAGPipeline
from llm.provider import get_chat_model

PDF_PATH = "data/real_pdfs/tesla_shareholder_deck.pdf"


def run_baseline():
    print("=" * 80)
    print("  STEP 3: BASELINE TEXT-ONLY RAG EVALUATION (TESLA Q4 2024 UPDATE)")
    print("=" * 80)

    # 1. Ingest with standard PyPDF pipeline
    print("\n1. Ingesting Tesla Shareholder Deck with standard text-only IngestionPipeline...")
    pipeline = IngestionPipeline(chunk_size=600, chunk_overlap=80)
    chunks, stats = pipeline.run_batch([PDF_PATH])
    print(f"Chunks generated: {len(chunks)} across {stats.total_raw_documents} pages")

    # 2. Build index
    embedder = get_embeddings()
    dense_store = get_or_create_faiss(documents=chunks, embeddings=embedder, index_path="data/faiss_tesla_baseline")
    retriever = dense_store.as_retriever(search_kwargs={"k": 4})
    llm = get_chat_model()
    rag = AdvancedRAGPipeline(dense_retriever=retriever, documents=chunks, llm=llm)

    # 3. Test Questions
    questions = [
        ("Text Question", "What was Tesla's total revenue in 2024?"),
        ("Table Question", "How many total vehicles did Tesla deliver in 2024?"),
        ("Visual Question", "What does the vehicle deliveries and production chart show across quarters in 2024?"),
    ]

    for qtype, q in questions:
        print("\n" + "-" * 70)
        print(f"[{qtype}]: {q}")
        t0 = time.time()
        res = rag.query(question=q, strategy="hybrid_rrf")
        dur = (time.time() - t0) * 1000
        print(f"\n[Answer ({dur:.2f} ms)]:\n{res['answer']}")
        print("\n[Retrieved Evidence Sources]:")
        for doc in res.get("documents", [])[:3]:
            page = doc.metadata.get("page", doc.metadata.get("page_number", "N/A"))
            clean_snippet = doc.page_content[:90].replace("\n", " ")
            print(f"  - Page {page} | {clean_snippet}...")


if __name__ == "__main__":
    run_baseline()
