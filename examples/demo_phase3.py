"""Interactive Phase 3 Demonstration Script for DocMind.

Run with:
    python examples/demo_phase3.py

Demonstrates:
1. Dedicated Embedding Model verification (Ollama nomic-embed-text / ConsistentFake)
2. Chroma Vector Store (In-Memory & Persistent SQLite)
3. FAISS Vector Store (In-Memory Index & Serialization)
4. Advanced Retrieval Modes: Similarity, MMR, Score Threshold, and Metadata Filtering
5. End-to-End RAG Q&A Chain (Retrieval + Gemma / LLM Generation)
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from llm.provider import get_chat_model
from ingestion.pipeline import IngestionPipeline
from vectorstore.embedder import get_embeddings, verify_embeddings
from vectorstore.store import get_or_create_chroma, get_or_create_faiss
from vectorstore.retriever import (
    create_retriever,
    similarity_search_with_scores,
    mmr_search,
)
from chains.qa_chain import create_structured_qa_chain

DATA_DIR = Path(__file__).parent.parent / "data"


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def run_demo():
    print_banner("DocMind Phase 3: Embeddings & Vector Stores Demo")

    # 1. Verify Embeddings
    print_banner("1. Dedicated Embedding Model Verification")
    embedding_provider = settings.default_embedding_provider
    print(f"Configured Embedding Provider: {embedding_provider}")
    print(f"Configured Embedding Model:    {settings.ollama_embedding_model if embedding_provider == 'ollama' else settings.openai_embedding_model}")

    try:
        embedder = get_embeddings(provider=embedding_provider)
        report = verify_embeddings(embedder)
        print(f"Embedding Health Check: SUCCESS! Vector Dimensions = {report['dimensions']}")
    except Exception as e:
        print(f"[NOTE] Live embedding provider ({embedding_provider}) unavailable: {e}")
        print("[INFO] Using deterministic ConsistentFakeEmbeddings for demonstration.")
        embedder = get_embeddings(provider="fake")
        report = verify_embeddings(embedder)
        print(f"Embedding Health Check: SUCCESS! Fake Dimensions = {report['dimensions']}")

    # 2. Ingest Sample Documents
    print_banner("2. Ingesting Real Multi-Format Documents")
    pipeline = IngestionPipeline(chunk_size=500, chunk_overlap=50)
    sources = [
        str(DATA_DIR / "sample_doc.txt"),
        str(DATA_DIR / "sample_guide.md"),
        str(DATA_DIR / "sample_data.csv"),
    ]
    chunks, ingest_report = pipeline.run_batch(sources)
    print(f"Ingested {ingest_report.total_raw_documents} documents -> {len(chunks)} chunks generated.")

    # 3. Chroma Vector Store Demo
    print_banner("3. Chroma Vector Store (In-Memory & Persistent)")
    chroma_dir = tempfile.mkdtemp(prefix="chroma_demo_")
    try:
        chroma_store = get_or_create_chroma(
            collection_name="docmind_demo",
            persist_directory=chroma_dir,
            embeddings=embedder,
            documents=chunks,
        )
        print(f"Chroma Store initialized and persisted to SQLite at {chroma_dir}")

        # Metadata Filtering Demo
        print("\nSearching with Metadata Filter (file_type == 'csv'):")
        csv_retriever = create_retriever(
            vectorstore=chroma_store,
            search_type="similarity",
            k=2,
            filter_dict={"file_type": "csv"},
        )
        csv_results = csv_retriever.invoke("budget and completion percentage")
        for idx, doc in enumerate(csv_results, start=1):
            print(f"   [{idx}] Source: {doc.metadata.get('filename')} (Row {doc.metadata.get('row')}):")
            print(f"       {doc.page_content.replace(chr(10), ' | ')}")

    finally:
        shutil.rmtree(chroma_dir, ignore_errors=True)

    # 4. FAISS Vector Store & MMR Retrieval Demo
    print_banner("4. FAISS Vector Store & MMR Diversity Retrieval")
    faiss_store = get_or_create_faiss(documents=chunks, embeddings=embedder)
    query = "What chunking techniques and document types are supported?"

    print(f"Query: '{query}'\n")
    print("A. Standard Similarity Search (Top 2):")
    sim_results = similarity_search_with_scores(faiss_store, query=query, k=2)
    for doc, score in sim_results:
        print(f"   - (Score {score:.4f}) {doc.page_content[:90]}...")

    print("\nB. Maximal Marginal Relevance (MMR) Search (k=2, fetch_k=6, lambda=0.5):")
    mmr_results = mmr_search(faiss_store, query=query, k=2, fetch_k=6, lambda_mult=0.5)
    for doc in mmr_results:
        print(f"   - [Source: {doc.metadata.get('filename')}] {doc.page_content[:90]}...")

    # 5. Full End-to-End RAG with Gemma / LLM Generation
    print_banner("5. Full End-to-End RAG Q&A (Retriever + Gemma / LCEL Chain)")
    retriever = create_retriever(vectorstore=faiss_store, search_type="similarity", k=4)
    retrieved_docs = retriever.invoke(query)
    context_text = "\n\n".join(
        f"--- Source: {d.metadata.get('filename')} (Type: {d.metadata.get('file_type')}) ---\n{d.page_content}"
        for d in retrieved_docs
    )

    try:
        llm = get_chat_model()
        # Test basic invocation
        llm.invoke("Hi")
        use_live_llm = True
    except Exception:
        use_live_llm = False

    if use_live_llm:
        print(f"Generating answer with live {settings.default_llm_provider} ({settings.ollama_model_name})...\n")
        rag_chain = create_structured_qa_chain(llm=llm)
        answer = rag_chain.invoke({"context": context_text, "question": query})
        print(f"Answer: {answer.answer}")
        print(f"Confidence: {answer.confidence_score * 100:.0f}%")
        print(f"Key Takeaways: {answer.key_takeaways}")
        print(f"Citations: {[c.model_dump() for c in answer.citations]}")
    else:
        print("[INFO] Live LLM offline. Showing retrieved context chunks that would be passed to Gemma:")
        print(context_text[:300] + "...\n")

    print_banner("Phase 3 Complete!")
    print("Embeddings, Chroma, FAISS, and MMR retrieval verified successfully.")


if __name__ == "__main__":
    run_demo()
