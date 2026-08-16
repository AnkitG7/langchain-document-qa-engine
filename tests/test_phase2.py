"""Unit tests for Phase 2: Document Ingestion, Cleaning, Chunking, and Pipeline."""

import os
import pytest
from pathlib import Path
from langchain_core.documents import Document

from ingestion.loaders import (
    load_document,
    TextDocumentLoader,
    MarkdownLoader,
    CSVDocumentLoader,
)
from ingestion.cleaner import (
    clean_text,
    clean_document,
    enrich_document_metadata,
    deduplicate_documents,
)
from ingestion.splitters import (
    create_recursive_splitter,
    create_token_splitter,
    create_markdown_header_splitter,
    split_documents,
)
from ingestion.pipeline import IngestionPipeline, IngestionReport

DATA_DIR = Path(__file__).parent.parent / "data"


class TestDocumentLoaders:
    def test_text_loader(self):
        txt_path = str(DATA_DIR / "sample_doc.txt")
        loader = TextDocumentLoader(file_path=txt_path)
        docs = loader.load()

        assert len(docs) == 1
        assert "DocMind" in docs[0].page_content
        assert docs[0].metadata["filename"] == "sample_doc.txt"
        assert docs[0].metadata["file_type"] == "txt"
        assert docs[0].metadata["file_size_bytes"] > 0

    def test_markdown_loader(self):
        md_path = str(DATA_DIR / "sample_guide.md")
        loader = MarkdownLoader(file_path=md_path)
        docs = loader.load()

        assert len(docs) == 1
        assert "System Architecture Guide" in docs[0].page_content
        assert docs[0].metadata["filename"] == "sample_guide.md"
        assert docs[0].metadata["file_type"] == "markdown"

    def test_csv_loader(self):
        csv_path = str(DATA_DIR / "sample_data.csv")
        loader = CSVDocumentLoader(file_path=csv_path)
        docs = loader.load()

        assert len(docs) == 5  # 5 rows
        assert "DocMind Core" in docs[0].page_content
        assert docs[0].metadata["row"] == 1
        assert "project_name" in docs[0].metadata["headers"]

    def test_universal_loader_dispatch(self):
        txt_docs = load_document(str(DATA_DIR / "sample_doc.txt"))
        assert len(txt_docs) == 1
        assert txt_docs[0].metadata["file_type"] == "txt"

        csv_docs = load_document(str(DATA_DIR / "sample_data.csv"))
        assert len(csv_docs) == 5
        assert csv_docs[0].metadata["file_type"] == "csv"


class TestCleaningAndEnrichment:
    def test_clean_text_normalization(self):
        dirty_text = "  Hello   world! \n\n\n\n  Multiple    newlines.  \u200b "
        cleaned = clean_text(dirty_text)
        assert cleaned == "Hello world!\n\nMultiple newlines."

    def test_enrich_metadata(self):
        doc = Document(page_content="Sample content for testing.", metadata={"source": "test.txt"})
        enriched = enrich_document_metadata(doc, doc_id="custom_id_123")

        assert enriched.metadata["doc_id"] == "custom_id_123"
        assert "content_hash" in enriched.metadata
        assert enriched.metadata["word_count"] == 4
        assert enriched.metadata["char_count"] == 27
        assert "ingested_at" in enriched.metadata

    def test_deduplicate_documents(self):
        doc1 = Document(page_content="Identical content", metadata={"source": "doc1.txt"})
        doc2 = Document(page_content="Identical content", metadata={"source": "doc2.txt"})
        doc3 = Document(page_content="Unique content", metadata={"source": "doc3.txt"})

        docs = [clean_document(d) for d in [doc1, doc2, doc3]]
        unique_docs = deduplicate_documents(docs)

        assert len(unique_docs) == 2
        assert {d.page_content for d in unique_docs} == {"Identical content", "Unique content"}


class TestSplitters:
    def test_recursive_splitter(self):
        splitter = create_recursive_splitter(chunk_size=50, chunk_overlap=10)
        long_text = "Paragraph one with some interesting text.\n\nParagraph two with more details.\n\nParagraph three."
        doc = Document(page_content=long_text, metadata={"doc_id": "doc_01"})

        chunks = split_documents([doc], splitter=splitter)
        assert len(chunks) >= 2
        assert chunks[0].metadata["parent_doc_id"] == "doc_01"
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == len(chunks)
        assert chunks[0].metadata["chunk_id"] == "doc_01_c000"

    def test_token_splitter(self):
        splitter = create_token_splitter(chunk_size=10, chunk_overlap=2)
        text = "This is a token-based chunking test that divides sentences based on tokenizer limits."
        chunks = splitter.split_text(text)
        assert len(chunks) > 1

    def test_markdown_header_splitter(self):
        splitter = create_markdown_header_splitter()
        md_text = "# Header 1\nContent under header 1.\n\n## Header 2\nContent under header 2."
        doc = Document(page_content=md_text, metadata={"source": "guide.md", "doc_id": "guide_01"})

        chunks = split_documents([doc], splitter=splitter)
        assert len(chunks) >= 2
        assert any("header_1" in c.metadata for c in chunks)


class TestIngestionPipeline:
    def test_pipeline_run_single(self):
        pipeline = IngestionPipeline(chunk_size=200, chunk_overlap=30)
        chunks, report = pipeline.run(str(DATA_DIR / "sample_doc.txt"))

        assert len(chunks) > 0
        assert isinstance(report, IngestionReport)
        assert report.total_raw_documents == 1
        assert report.final_chunks_count == len(chunks)
        assert report.total_words > 0
        assert report.duration_seconds >= 0

    def test_pipeline_batch_run(self):
        pipeline = IngestionPipeline(chunk_size=300, chunk_overlap=30)
        sources = [
            str(DATA_DIR / "sample_doc.txt"),
            str(DATA_DIR / "sample_guide.md"),
            str(DATA_DIR / "sample_data.csv"),
        ]
        chunks, report = pipeline.run_batch(sources)

        assert len(chunks) > 0
        assert report.total_raw_documents == 7  # 1 txt + 1 md + 5 csv rows
        assert len(report.sources_processed) == 3
        assert report.final_chunks_count == len(chunks)
