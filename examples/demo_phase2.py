"""Interactive Phase 2 Demonstration Script for DocMind.

Run with:
    python examples/demo_phase2.py

Demonstrates:
1. Multi-format Document Loading (PDF, TXT, MD, CSV, Web)
2. Text Cleaning & Normalization
3. Metadata Enrichment & Hash-based Deduplication
4. Text Splitting Strategies (Recursive, Token, Markdown Header)
5. End-to-End Ingestion Pipeline Orchestration with Audit Telemetry
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.loaders import load_document
from ingestion.cleaner import clean_text, clean_document, deduplicate_documents
from ingestion.splitters import (
    create_recursive_splitter,
    create_token_splitter,
    create_markdown_header_splitter,
    split_documents,
)
from ingestion.pipeline import IngestionPipeline

DATA_DIR = Path(__file__).parent.parent / "data"


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def run_demo():
    print_banner("DocMind Phase 2: Document Ingestion & Chunking Demo")

    # 1. Multi-Format Loading
    print_banner("1. Multi-Format Loading via Universal Loader")
    files = ["sample_doc.txt", "sample_guide.md", "sample_data.csv"]
    for fname in files:
        fpath = str(DATA_DIR / fname)
        docs = load_document(fpath)
        print(f"File: {fname:<18} | Loaded Docs: {len(docs):<2} | First Doc Type: {docs[0].metadata.get('file_type')}")
        print(f"   Sample Content: {docs[0].page_content[:80]}...")

    # 2. Text Cleaning & Metadata Enrichment
    print_banner("2. Text Cleaning & Metadata Enrichment")
    dirty_snippet = "  DocMind   Engine\n\n\n\n  Provides    clean text.\u200b  "
    cleaned_snippet = clean_text(dirty_snippet)
    print(f"Raw Input:     {repr(dirty_snippet)}")
    print(f"Cleaned Text:  {repr(cleaned_snippet)}")

    sample_doc = load_document(str(DATA_DIR / "sample_doc.txt"))[0]
    enriched = clean_document(sample_doc)
    print("\nEnriched Document Metadata:")
    for k, v in enriched.metadata.items():
        print(f"  - {k}: {v}")

    # 3. Comparing Chunking Strategies
    print_banner("3. Comparing Chunking Strategies on Markdown Guide")
    md_doc = load_document(str(DATA_DIR / "sample_guide.md"))[0]

    # Strategy A: Recursive Character Splitting
    rec_splitter = create_recursive_splitter(chunk_size=150, chunk_overlap=20)
    rec_chunks = split_documents([md_doc], splitter=rec_splitter)
    print(f"A. Recursive Splitter (size=150, overlap=20): {len(rec_chunks)} chunks produced")
    print(f"   Chunk 0: {repr(rec_chunks[0].page_content[:60])}...")
    print(f"   Chunk 0 Metadata: {rec_chunks[0].metadata}")

    # Strategy B: Token-based Splitting
    tok_splitter = create_token_splitter(chunk_size=30, chunk_overlap=5)
    tok_chunks = split_documents([md_doc], splitter=tok_splitter)
    print(f"\nB. Token-based Splitter (tokens=30, overlap=5): {len(tok_chunks)} chunks produced")

    # Strategy C: Markdown Header Splitting
    md_splitter = create_markdown_header_splitter()
    md_chunks = split_documents([md_doc], splitter=md_splitter)
    print(f"\nC. Markdown Header Splitter: {len(md_chunks)} chunks produced")
    for i, c in enumerate(md_chunks):
        headers = {k: v for k, v in c.metadata.items() if "header" in k}
        print(f"   Chunk {i}: Headers={headers} | Chars={len(c.page_content)}")

    # 4. End-to-End Ingestion Pipeline
    print_banner("4. Full Batch Ingestion Pipeline with Telemetry Report")
    pipeline = IngestionPipeline(chunk_size=200, chunk_overlap=30)
    sources = [
        str(DATA_DIR / "sample_doc.txt"),
        str(DATA_DIR / "sample_guide.md"),
        str(DATA_DIR / "sample_data.csv"),
    ]

    final_chunks, report = pipeline.run_batch(sources)

    print("Ingestion Audit Report:")
    print(f"  - Sources Processed:        {len(report.sources_processed)}")
    print(f"  - Total Raw Documents:      {report.total_raw_documents}")
    print(f"  - Total Chunks Created:     {report.total_chunks_created}")
    print(f"  - Duplicates Removed:       {report.duplicate_chunks_removed}")
    print(f"  - Final Chunks for Indexing:{report.final_chunks_count}")
    print(f"  - Total Word Count:         {report.total_words}")
    print(f"  - Avg Chunk Size:           {report.avg_chunk_size_chars} chars")
    print(f"  - Pipeline Execution Time:  {report.duration_seconds * 1000:.2f} ms")

    print_banner("Phase 2 Complete!")
    print("Document ingestion, multi-format loading, cleaning, and chunking verified successfully.")


if __name__ == "__main__":
    run_demo()
