"""Document ingestion, loading, cleaning, chunking, and metadata enrichment module."""

from .loaders import (
    load_document,
    load_documents_batch,
    PDFDocumentLoader,
    TextDocumentLoader,
    MarkdownLoader,
    CSVDocumentLoader,
    WebDocumentLoader,
)
from .cleaner import clean_text, clean_document, enrich_document_metadata, deduplicate_documents
from .splitters import (
    get_text_splitter,
    split_documents,
    create_recursive_splitter,
    create_token_splitter,
    create_markdown_header_splitter,
)
from .pipeline import IngestionPipeline, IngestionReport
from .multimodal_parser import MultimodalDocumentParser, ExtractedElement
from .multimodal_pipeline import MultimodalIngestionPipeline, MultimodalIngestionReport
from .document_registry import (
    DocumentRegistry,
    get_document_registry,
    calculate_file_sha256,
    compute_config_signature,
    DocumentRegistryEntry,
)

__all__ = [
    # Loaders
    "load_document",
    "load_documents_batch",
    "PDFDocumentLoader",
    "TextDocumentLoader",
    "MarkdownLoader",
    "CSVDocumentLoader",
    "WebDocumentLoader",
    # Cleaner
    "clean_text",
    "clean_document",
    "enrich_document_metadata",
    "deduplicate_documents",
    # Splitters
    "get_text_splitter",
    "split_documents",
    "create_recursive_splitter",
    "create_token_splitter",
    "create_markdown_header_splitter",
    # Pipeline & Multimodal
    "IngestionPipeline",
    "IngestionReport",
    "MultimodalDocumentParser",
    "ExtractedElement",
    "MultimodalIngestionPipeline",
    "MultimodalIngestionReport",
    # Registry & Fingerprinting
    "DocumentRegistry",
    "get_document_registry",
    "calculate_file_sha256",
    "compute_config_signature",
    "DocumentRegistryEntry",
]
