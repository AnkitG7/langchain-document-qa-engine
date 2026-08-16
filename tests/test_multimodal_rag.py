"""Unit and Integration Tests for Multimodal Document RAG Subsystem.

Tests:
1. Table Markdown formatting and extraction
2. Image byte extraction and PIL handling
3. VisionModelProvider interface and formatting
4. MultimodalIngestionPipeline orchestration
5. Unified Hybrid RAG retrieval across Text, Tables, and Visual descriptions
"""

import io
import pytest
from pathlib import Path
from PIL import Image
from langchain_core.documents import Document

from ingestion.multimodal_parser import MultimodalDocumentParser, ExtractedElement
from ingestion.multimodal_pipeline import MultimodalIngestionPipeline, MultimodalIngestionReport
from llm.vision import VisionModelProvider
from vectorstore.embedder import get_embeddings
from vectorstore.store import get_or_create_faiss
from rag_advanced.hybrid import HybridRetriever
from rag_advanced.pipeline import AdvancedRAGPipeline
from llm.provider import get_chat_model


class TestTableAndParser:
    """Tests table serialization and parser helpers."""

    def test_table_markdown_formatting(self):
        parser = MultimodalDocumentParser()
        raw_table = [
            ["Metric", "Q1 2024", "Q2 2024"],
            ["Deliveries", "386,810", "443,956"],
            ["Production", "433,371", "410,831"],
        ]
        md = parser._format_table_as_markdown(raw_table)
        assert md is not None
        assert "| Metric | Q1 2024 | Q2 2024 |" in md
        assert "| Deliveries | 386,810 | 443,956 |" in md
        assert "| Production | 433,371 | 410,831 |" in md

    def test_table_markdown_handles_empty_and_pipes(self):
        parser = MultimodalDocumentParser()
        raw_table = [
            ["Item | Name", "Value"],
            ["Test\nRow", None],
        ]
        md = parser._format_table_as_markdown(raw_table)
        assert md is not None
        assert "Item / Name" in md  # Pipe sanitized
        assert "Test Row" in md  # Newline collapsed

    def test_extracted_element_model(self):
        elem = ExtractedElement(
            element_id="test_01",
            element_type="table",
            page_number=3,
            source_file="sample.pdf",
            text_content="Table Content",
            table_markdown="| Col1 | Col2 |\n|---|---|\n| A | B |",
        )
        assert elem.element_type == "table"
        assert elem.page_number == 3
        assert elem.table_markdown is not None


class TestVisionModelProvider:
    """Tests vision model byte conversions and mock fallback."""

    def test_to_image_bytes_from_pil(self):
        vision = VisionModelProvider()
        img = Image.new("RGB", (50, 50), color="blue")
        raw_bytes = vision._to_image_bytes(img)
        assert raw_bytes is not None
        assert len(raw_bytes) > 0

    def test_to_image_bytes_from_raw(self):
        vision = VisionModelProvider()
        sample_bytes = b"fake_png_header_and_data_12345"
        converted = vision._to_image_bytes(sample_bytes)
        assert converted == sample_bytes

    def test_describe_image_empty_input_returns_safe_message(self):
        vision = VisionModelProvider()
        res = vision.describe_image(b"")
        assert "Empty" in res or "unreadable" in res


class TestMultimodalIngestionPipeline:
    """Tests unified multimodal pipeline document construction."""

    def test_multimodal_pipeline_synthetic_elements(self, monkeypatch):
        # Mock parser to return synthetic text, table, and image elements
        def fake_parse_pdf(self, path):
            return [
                ExtractedElement(
                    element_id="doc_p001_txt",
                    element_type="text",
                    page_number=1,
                    source_file="test_doc.pdf",
                    text_content="Tesla reported record energy storage deployment in 2024.",
                ),
                ExtractedElement(
                    element_id="doc_p002_tab_01",
                    element_type="table",
                    page_number=2,
                    source_file="test_doc.pdf",
                    text_content="Table on Page 2",
                    table_markdown="| Vehicle | Q4 Deliveries |\n|---|---|\n| Model 3/Y | 461,384 |",
                    metadata={"rows": 2, "columns": 2},
                ),
                ExtractedElement(
                    element_id="doc_p003_img_01",
                    element_type="image",
                    page_number=3,
                    source_file="test_doc.pdf",
                    image_path="data/test_img.png",
                    metadata={"width": 600, "height": 400},
                ),
            ]

        monkeypatch.setattr(MultimodalDocumentParser, "parse_pdf", fake_parse_pdf)

        pipeline = MultimodalIngestionPipeline(enable_vision_processing=False)
        docs, report = pipeline.ingest_pdf("dummy.pdf")

        assert len(docs) == 3
        assert report.text_chunks_count == 1
        assert report.tables_count == 1
        assert report.images_count == 1

        types = [d.metadata.get("element_type") for d in docs]
        assert "text" in types
        assert "table" in types
        assert "image" in types

        # Check table content formatting
        table_doc = next(d for d in docs if d.metadata.get("element_type") == "table")
        assert "| Model 3/Y | 461,384 |" in table_doc.page_content
        assert table_doc.metadata.get("page") == 2


class TestUnifiedMultimodalRetrieval:
    """Tests hybrid search retrieval across mixed text, table, and visual chunks."""

    def test_retrieval_finds_correct_element_types(self):
        docs = [
            Document(
                page_content="Tesla full year total automotive revenue reached 77.1 billion in 2024.",
                metadata={"filename": "tsla.pdf", "page": 4, "element_type": "text"},
            ),
            Document(
                page_content=(
                    "### [Table on Page 5 of tsla.pdf]\n\n"
                    "| Model | Total Deliveries 2024 |\n"
                    "|---|---|\n"
                    "| Model 3 and Model Y | 1,734,000 |\n"
                    "| Other Models | 56,000 |\n"
                    "| Total | 1,790,000 |"
                ),
                metadata={"filename": "tsla.pdf", "page": 5, "element_type": "table"},
            ),
            Document(
                page_content=(
                    "### [Visual Chart on Page 7 of tsla.pdf]\n"
                    "Bar chart showing vehicle deliveries by quarter from Q1 2023 to Q4 2024. "
                    "Q4 2024 represents the highest quarterly peak with 484,000 units delivered."
                ),
                metadata={"filename": "tsla.pdf", "page": 7, "element_type": "image"},
            ),
        ]

        embedder = get_embeddings()
        faiss_store = get_or_create_faiss(documents=docs, embeddings=embedder)
        retriever = faiss_store.as_retriever(search_kwargs={"k": 2})

        # 1. Query for Table Data
        table_results = retriever.invoke("How many Model 3 and Model Y vehicles were delivered?")
        assert len(table_results) > 0
        assert table_results[0].metadata["element_type"] == "table"
        assert "1,734,000" in table_results[0].page_content

        # 2. Query for Visual Chart Data
        chart_results = retriever.invoke("What does the quarterly deliveries bar chart peak show?")
        assert len(chart_results) > 0
        assert any(d.metadata["element_type"] == "image" for d in chart_results)
        assert "Bar chart" in chart_results[0].page_content
