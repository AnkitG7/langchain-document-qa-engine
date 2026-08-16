"""Text Cleaning, Metadata Enrichment, and Deduplication Utilities.

Demonstrates:
- Text sanitization for RAG (whitespace normalization, unicode normalization)
- Document metadata enrichment (content hash, word count, token estimation, unique doc_id)
- Content hash-based deduplication
"""

import re
import hashlib
import unicodedata
from datetime import datetime, timezone
from typing import List, Optional, Set
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Sanitizes raw document text for higher vector embedding quality.

    - Normalizes unicode characters (NFKC)
    - Strips zero-width characters and unusual control codes
    - Strips leading/trailing spaces on each line and normalizes internal whitespace
    - Collapses triple+ newlines down to double newlines (preserving paragraphs)
    - Trims leading and trailing whitespace
    """
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove zero-width spaces, soft hyphens, and weird control characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\xad]", "", text)

    # Clean line-by-line: trim whitespace on each line and collapse internal spaces
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]

    # Reconstruct text
    reconstructed = "\n".join(lines)

    # Collapse consecutive blank lines (3 or more newlines -> 2 newlines)
    cleaned = re.sub(r"\n{3,}", "\n\n", reconstructed)

    return cleaned.strip()


def calculate_content_hash(content: str) -> str:
    """Computes a deterministic SHA-256 hash of cleaned text."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def enrich_document_metadata(
    doc: Document,
    doc_id: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> Document:
    """Enriches a LangChain Document with analytical metadata fields.

    Adds:
    - `content_hash`: SHA-256 hash for deduplication and provenance
    - `char_count`: Total character length
    - `word_count`: Total word count
    - `estimated_tokens`: Rough token estimate (~4 chars per token)
    - `doc_id`: Unique document/chunk identifier
    - `ingested_at`: UTC timestamp of ingestion
    """
    content = doc.page_content
    c_hash = calculate_content_hash(content)
    words = len(content.split())
    chars = len(content)
    est_tokens = max(1, chars // 4)

    metadata = dict(doc.metadata)
    metadata["content_hash"] = c_hash
    metadata["char_count"] = chars
    metadata["word_count"] = words
    metadata["estimated_tokens"] = est_tokens
    metadata["doc_id"] = doc_id or f"doc_{c_hash[:12]}"
    metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()

    if extra_metadata:
        metadata.update(extra_metadata)

    return Document(page_content=content, metadata=metadata)


def clean_document(doc: Document) -> Document:
    """Cleans the document's page_content and updates its enriched metadata."""
    cleaned_content = clean_text(doc.page_content)
    cleaned_doc = Document(page_content=cleaned_content, metadata=dict(doc.metadata))
    return enrich_document_metadata(cleaned_doc)


def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """Removes duplicate documents or chunks based on their content hash."""
    seen_hashes: Set[str] = set()
    unique_docs: List[Document] = []

    for doc in docs:
        c_hash = doc.metadata.get("content_hash") or calculate_content_hash(doc.page_content)
        if c_hash not in seen_hashes:
            seen_hashes.add(c_hash)
            unique_docs.append(doc)

    return unique_docs
