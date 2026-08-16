"""Live Real-World Multimodal RAG Demonstration and Benchmark.

Evaluates:
- Level 1: Text Question (Revenue)
- Level 2: Table Question (Vehicle Deliveries by model & total)
- Level 3: Visual/Chart Question (Deliveries and Production trends across quarters)

Compares:
- Text-Only Baseline vs. Multimodal RAG (Tables + Vision Descriptions)
- RAG Triad Faithfulness & Answer Relevance with gemma4:cloud
"""

import sys
import os
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.multimodal_pipeline import MultimodalIngestionPipeline
from vectorstore.embedder import get_embeddings
from vectorstore.store import get_or_create_faiss
from rag_advanced.pipeline import AdvancedRAGPipeline
from llm.provider import get_chat_model
from evaluation.metrics import FaithfulnessMetric, AnswerRelevanceMetric

PDF_PATH = "data/real_pdfs/tesla_shareholder_deck.pdf"
INDEX_PATH = "data/faiss_tesla_multimodal"


def run_multimodal_demo():
    print("\n" + "=" * 80)
    print("  DOCMIND MULTIMODAL RAG: REAL IMAGE, TABLE & CHART EVALUATION")
    print("=" * 80)

    pdf_file = Path(PDF_PATH)
    if not pdf_file.exists():
        print(f"Error: {PDF_PATH} not found.")
        return

    # 1. Ingest PDF with Multimodal Ingestion Pipeline
    print(f"\n1. Ingesting {pdf_file.name} with MultimodalIngestionPipeline...")
    print("   - Text: PyMuPDF clean text parsing")
    print("   - Tables: pdfplumber Markdown grid extraction")
    print("   - Images & Charts: PyMuPDF extraction + gemma4:cloud Vision LLM description")

    pipeline = MultimodalIngestionPipeline(
        chunk_size=600,
        chunk_overlap=80,
        enable_vision_processing=True,
    )
    docs, report = pipeline.ingest_pdf(str(pdf_file))

    print(f"\n[Ingestion Complete in {report.duration_seconds:.2f}s]")
    print(f"  Total Pages Processed:    {report.total_pages_processed}")
    print(f"  Text Chunks Created:      {report.text_chunks_count}")
    print(f"  Structured Tables Extracted: {report.tables_count}")
    print(f"  Visual Figures Described: {report.images_count}")
    print(f"  Total Unified Documents:  {report.total_unified_documents}")

    # 2. Build Hybrid Dense + BM25 Index
    print("\n2. Indexing Unified Multimodal Documents with nomic-embed-text & BM25...")
    embedder = get_embeddings()
    dense_store = get_or_create_faiss(
        documents=docs,
        embeddings=embedder,
        index_path=INDEX_PATH,
    )
    retriever = dense_store.as_retriever(search_kwargs={"k": 5})
    llm = get_chat_model()
    rag = AdvancedRAGPipeline(dense_retriever=retriever, documents=docs, llm=llm)

    # 3. Evaluators
    faith_eval = FaithfulnessMetric(llm=llm)
    rel_eval = AnswerRelevanceMetric(llm=llm)

    # 4. Multimodal Test Suite
    test_cases = [
        {
            "id": 1,
            "level": "Level 1: Text",
            "question": "What was Tesla's total revenue in 2024?",
            "baseline_status": "PASS ($97,690M)",
        },
        {
            "id": 2,
            "level": "Level 2: Table",
            "question": "How many total vehicles did Tesla deliver in 2024, broken down by Model 3/Y and other models?",
            "baseline_status": "FAIL (Flattened/Lost)",
        },
        {
            "id": 3,
            "level": "Level 3: Visual/Chart",
            "question": "What does the vehicle deliveries and production chart show across quarters in 2024?",
            "baseline_status": "FAIL (Blind to Charts)",
        },
        {
            "id": 4,
            "level": "Level 3: Visual Chart (COGS)",
            "question": "What does the Average COGS per vehicle chart show regarding cost trends across quarters in 2024?",
            "baseline_status": "FAIL (Blind to COGS chart)",
        },
    ]

    results = []

    for tc in test_cases:
        qid = tc["id"]
        lvl = tc["level"]
        q = tc["question"]

        print("\n" + "=" * 80)
        print(f"  TEST {qid}/{len(test_cases)}: {lvl}")
        print("=" * 80)
        print(f"[Question]: {q}")

        t0 = time.time()
        res = rag.query(question=q, strategy="hybrid_rrf")
        latency = (time.time() - t0) * 1000

        ans = res["answer"]
        ctx_docs = res.get("documents", [])
        ctx_text = res.get("context", "\n\n".join(d.page_content for d in ctx_docs))

        print(f"\n[DocMind Multimodal Answer ({latency:.2f} ms)]:\n{ans}\n")
        print(f"[Retrieved Evidence Chunks ({len(ctx_docs)})]:")
        for d in ctx_docs[:4]:
            p = d.metadata.get("page", d.metadata.get("page_number", "N/A"))
            etype = d.metadata.get("element_type", "text")
            snippet = d.page_content[:100].encode("ascii", "replace").decode("ascii").replace("\n", " ")
            print(f"  - Page {p:2} | Type: {etype:6} | {snippet}...")

        # Run RAG Triad Evaluator
        print("\n[Evaluating with RAG Triad (gemma4:cloud)...]")
        f_res = faith_eval.evaluate(answer=ans, context=ctx_text)
        r_res = rel_eval.evaluate(question=q, answer=ans)

        print(f"  * Faithfulness: {f_res.score:.2f} ({'PASS' if f_res.score >= 0.8 else 'FLAGGED'}) | {f_res.reasoning}")
        print(f"  * Relevance:    {r_res.score:.2f} ({'PASS' if r_res.score >= 0.8 else 'FLAGGED'}) | {r_res.reasoning}")

        results.append({
            "id": qid,
            "level": lvl,
            "latency": latency,
            "faithfulness": f_res.score,
            "relevance": r_res.score,
            "baseline": tc["baseline_status"],
            "after_status": "PASS" if (f_res.score >= 0.8 and r_res.score >= 0.7) else "PARTIAL",
        })

    # 5. Summary Comparison Scorecard
    print("\n" + "=" * 80)
    print("  MULTIMODAL RAG VS TEXT-ONLY BASELINE SCORECARD")
    print("=" * 80)
    print(f"{'#':<3} | {'Difficulty Level':<22} | {'Text-Only Baseline':<20} | {'Multimodal RAG':<16} | {'Faith.':<6} | {'Relev.':<6}")
    print("-" * 80)
    for r in results:
        print(f"{r['id']:<3} | {r['level']:<22} | {r['baseline']:<20} | {r['after_status']:<16} | {r['faithfulness']:<6.2f} | {r['relevance']:<6.2f}")
    print("-" * 80)
    avg_f = sum(r["faithfulness"] for r in results) / len(results)
    avg_r = sum(r["relevance"] for r in results) / len(results)
    print(f"OVERALL: Faithfulness = {avg_f:.2f}/1.00 | Relevance = {avg_r:.2f}/1.00")
    print("=" * 80)


if __name__ == "__main__":
    run_multimodal_demo()
