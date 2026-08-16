"""End-to-End Document Ingestion Pipeline Orchestrator.

Demonstrates:
- Full ingestion lifecycle: Load -> Clean -> Enrich -> Split -> Deduplicate -> Validate
- Comprehensive ingestion telemetry (IngestionReport)
- Batch multi-file and multi-source processing
"""

import time
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from config import settings
from .loaders import load_document, load_documents_batch
from .cleaner import clean_document, deduplicate_documents
from .splitters import get_text_splitter, split_documents


from pathlib import Path
from .document_registry import (
    DocumentRegistry,
    get_document_registry,
    calculate_file_sha256,
    compute_config_signature,
    DocumentRegistryEntry,
)


class IngestionReport(BaseModel):
    """Execution metrics and audit statistics for an ingestion run."""
    sources_processed: List[str] = Field(default_factory=list)
    total_raw_documents: int = Field(default=0, description="Raw Document objects loaded from disk/web")
    total_chunks_created: int = Field(default=0, description="Chunks generated after text splitting")
    duplicate_chunks_removed: int = Field(default=0, description="Duplicate chunks filtered out")
    final_chunks_count: int = Field(default=0, description="Total active chunks ready for embedding")
    total_words: int = Field(default=0, description="Total word count across all chunks")
    avg_chunk_size_chars: float = Field(default=0.0, description="Average character length per chunk")
    duration_seconds: float = Field(default=0.0, description="Time taken in seconds")
    is_reused: bool = Field(default=False, description="True if document was already indexed with identical config")
    cache_status: str = Field(default="PROCESSED", description="CACHE_HIT, CONFIG_DRIFT, or NEW_DOCUMENT")
    content_fingerprint: Optional[str] = Field(default=None, description="SHA-256 binary hash of file")
    config_signature: Optional[str] = Field(default=None, description="SHA-256 hash of ingestion configuration")


class IngestionPipeline:
    """Orchestrates the complete document ingestion lifecycle with content-based caching."""

    def __init__(
        self,
        splitter_type: Optional[Literal["recursive", "token", "markdown", "semantic"]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        registry: Optional[DocumentRegistry] = None,
        embedding_model: str = "nomic-embed-text",
    ):
        self.splitter_type = splitter_type or settings.default_splitter_type
        self.chunk_size = chunk_size or settings.default_chunk_size
        self.chunk_overlap = chunk_overlap or settings.default_chunk_overlap
        self.embedding_model = embedding_model
        self.registry = registry or get_document_registry()
        self.splitter = get_text_splitter(
            splitter_type=self.splitter_type,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def get_config_signature(self) -> str:
        """Returns the deterministic configuration signature for this pipeline instance."""
        return compute_config_signature(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            splitter_type=self.splitter_type,
            parser_type="standard",
            embedding_model=self.embedding_model,
        )

    def check_cache(self, source: str) -> Tuple[bool, Optional[DocumentRegistryEntry], str]:
        """Checks whether a local file is already indexed with identical configuration."""
        path = Path(source)
        if not path.exists() or not path.is_file():
            return False, None, "NOT_FOUND"
        fingerprint = calculate_file_sha256(path)
        sig = self.get_config_signature()
        return self.registry.lookup(fingerprint, sig)

    def run(
        self,
        source: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        force_reprocess: bool = False,
    ) -> Tuple[List[Document], IngestionReport]:
        """Ingests a single document source with duplicate content detection."""
        path = Path(source)
        sig = self.get_config_signature()

        # Content-based fingerprint check for local files
        if path.exists() and path.is_file() and not force_reprocess:
            fingerprint = calculate_file_sha256(path)
            is_valid, entry, status = self.registry.lookup(fingerprint, sig)
            if is_valid and entry:
                self.registry.record_alias_filename(fingerprint, path.name)
                # Load or split chunks for caller convenience while signaling reuse
                chunks, report = self.run_batch(sources=[source], extra_metadata=extra_metadata)
                report.is_reused = True
                report.cache_status = "CACHE_HIT"
                report.content_fingerprint = fingerprint
                report.config_signature = sig
                return chunks, report

        # Process new or drifted document
        chunks, report = self.run_batch(sources=[source], extra_metadata=extra_metadata)

        # Register document if local file
        if path.exists() and path.is_file() and chunks:
            fingerprint = calculate_file_sha256(path)
            doc_id = chunks[0].metadata.get("parent_doc_id", chunks[0].metadata.get("doc_id", f"doc_{fingerprint[:12]}"))
            total_chars = sum(len(c.page_content) for c in chunks)
            self.registry.register(
                content_fingerprint=fingerprint,
                config_signature=sig,
                doc_id=doc_id,
                filename=path.name,
                file_type=path.suffix.lstrip(".").lower() or "unknown",
                file_size_bytes=path.stat().st_size,
                chunks_count=len(chunks),
                character_count=total_chars,
                config_details={
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "splitter_type": self.splitter_type,
                    "embedding_model": self.embedding_model,
                },
            )
            report.content_fingerprint = fingerprint
            report.config_signature = sig

        return chunks, report

    def run_batch(
        self,
        sources: List[str],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Document], IngestionReport]:
        """Ingests a batch of document sources through the full pipeline."""
        start_time = time.perf_counter()

        # Step 1: Load raw documents
        raw_docs: List[Document] = []
        for src in sources:
            docs = load_document(src)
            if extra_metadata:
                for doc in docs:
                    doc.metadata.update(extra_metadata)
            raw_docs.extend(docs)

        # Step 2: Clean and enrich raw documents
        cleaned_docs: List[Document] = [clean_document(doc) for doc in raw_docs]

        # Step 3: Split documents into chunks with index tracking
        chunks = split_documents(
            documents=cleaned_docs,
            splitter=self.splitter,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        total_chunks_raw = len(chunks)

        # Step 4: Deduplicate chunks based on content hash
        final_chunks = deduplicate_documents(chunks)
        duplicates_removed = total_chunks_raw - len(final_chunks)

        # Step 5: Compute statistics
        duration = time.perf_counter() - start_time
        total_chars = sum(len(c.page_content) for c in final_chunks)
        total_words = sum(len(c.page_content.split()) for c in final_chunks)
        avg_chunk_size = (total_chars / len(final_chunks)) if final_chunks else 0.0

        report = IngestionReport(
            sources_processed=sources,
            total_raw_documents=len(raw_docs),
            total_chunks_created=total_chunks_raw,
            duplicate_chunks_removed=duplicates_removed,
            final_chunks_count=len(final_chunks),
            total_words=total_words,
            avg_chunk_size_chars=round(avg_chunk_size, 2),
            duration_seconds=round(duration, 4),
            is_reused=False,
            cache_status="PROCESSED",
        )

        return final_chunks, report
