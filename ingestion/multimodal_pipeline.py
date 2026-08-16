"""Unified Multimodal Ingestion Pipeline: Text, Tables, and Visual Elements.

Demonstrates:
- Multi-element orchestration (Text, Tables, Images, Charts)
- Converting Markdown tables into searchable structured chunks
- Enriching visual figures with Vision LLM factual summaries
- Unified document stream with element-type tagging for hybrid RAG
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from .multimodal_parser import MultimodalDocumentParser, ExtractedElement
from .splitters import get_text_splitter, split_documents
from .cleaner import calculate_content_hash
from llm.vision import VisionModelProvider


class MultimodalIngestionReport(BaseModel):
    """Statistics and metrics from multimodal document ingestion."""
    source_files: List[str] = Field(default_factory=list)
    total_pages_processed: int = 0
    text_chunks_count: int = 0
    tables_count: int = 0
    images_count: int = 0
    total_unified_documents: int = 0
    duration_seconds: float = 0.0


class MultimodalIngestionPipeline:
    """Orchestrates extraction of text, tables, and images into unified LangChain documents."""

    def __init__(
        self,
        parser: Optional[MultimodalDocumentParser] = None,
        vision_provider: Optional[VisionModelProvider] = None,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        enable_vision_processing: bool = True,
    ):
        self.parser = parser or MultimodalDocumentParser()
        self.vision_provider = vision_provider or VisionModelProvider()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_vision_processing = enable_vision_processing
        self.text_splitter = get_text_splitter("recursive", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest_pdf(
        self,
        pdf_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Document], MultimodalIngestionReport]:
        """Ingests a PDF into unified text, table, and image documents."""
        start_time = time.perf_counter()
        elements = self.parser.parse_pdf(pdf_path)

        unified_docs: List[Document] = []
        text_count = 0
        table_count = 0
        image_count = 0
        pages_seen = set()

        for elem in elements:
            pages_seen.add(elem.page_number)
            base_meta = {
                "source": elem.source_file,
                "filename": elem.source_file,
                "page": elem.page_number,
                "page_number": elem.page_number,
                "element_type": elem.element_type,
            }
            if extra_metadata:
                base_meta.update(extra_metadata)

            # 1. Text Elements -> Standard Text Chunking
            if elem.element_type in ("text", "scanned_page") and elem.text_content:
                parent_doc = Document(page_content=elem.text_content, metadata=base_meta)
                chunks = split_documents([parent_doc], splitter=self.text_splitter)
                for c in chunks:
                    c.metadata["element_type"] = "text"
                    c.metadata["content_hash"] = calculate_content_hash(c.page_content)
                unified_docs.extend(chunks)
                text_count += len(chunks)

            # 2. Table Elements -> Structured Markdown Documents
            elif elem.element_type == "table" and elem.table_markdown:
                table_meta = dict(base_meta)
                table_meta["element_type"] = "table"
                table_meta["table_rows"] = elem.metadata.get("rows", 0)
                table_meta["table_cols"] = elem.metadata.get("columns", 0)

                table_content = (
                    f"### [Table on Page {elem.page_number} of {elem.source_file}]\n\n"
                    f"{elem.table_markdown}"
                )
                table_meta["content_hash"] = calculate_content_hash(table_content)

                table_doc = Document(page_content=table_content, metadata=table_meta)
                unified_docs.append(table_doc)
                table_count += 1

            # 3. Image & Chart Elements -> Vision LLM Description
            elif elem.element_type in ("image", "chart") and elem.image_path:
                img_meta = dict(base_meta)
                img_meta["element_type"] = elem.element_type
                img_meta["image_path"] = elem.image_path
                img_meta.update(elem.metadata)

                if self.enable_vision_processing:
                    # Generate factual visual description using Vision LLM
                    desc = self.vision_provider.describe_image(
                        elem.image_path,
                        context_hint=f"From {elem.source_file} page {elem.page_number} ({elem.element_type})",
                    )
                else:
                    desc = f"[Visual {elem.element_type} on page {elem.page_number}: {Path(elem.image_path).name}]"

                visual_content = (
                    f"### [Visual {elem.element_type.title()} on Page {elem.page_number} of {elem.source_file}]\n"
                    f"{desc}"
                )
                img_meta["content_hash"] = calculate_content_hash(visual_content)
                img_meta["visual_description"] = desc

                img_doc = Document(page_content=visual_content, metadata=img_meta)
                unified_docs.append(img_doc)
                image_count += 1

        duration = round(time.perf_counter() - start_time, 2)
        report = MultimodalIngestionReport(
            source_files=[Path(pdf_path).name],
            total_pages_processed=len(pages_seen),
            text_chunks_count=text_count,
            tables_count=table_count,
            images_count=image_count,
            total_unified_documents=len(unified_docs),
            duration_seconds=duration,
        )

        return unified_docs, report
