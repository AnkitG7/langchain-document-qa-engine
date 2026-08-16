"""Document Splitting and Chunking Strategies.

Demonstrates:
- RecursiveCharacterTextSplitter (hierarchical semantic boundaries)
- TokenTextSplitter (strict token-limit budgeting)
- MarkdownHeaderTextSplitter (structure-aware header hierarchy)
- SemanticChunker (embedding-distance-based semantic boundary splitting)
- Chunk metadata preservation and indexing
"""

from typing import List, Literal, Optional
from langchain_core.documents import Document

try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        TokenTextSplitter,
        MarkdownHeaderTextSplitter,
    )
except ImportError:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        TokenTextSplitter,
        MarkdownHeaderTextSplitter,
    )

from config import settings
from .cleaner import calculate_content_hash


def create_recursive_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None,
) -> RecursiveCharacterTextSplitter:
    """Creates a RecursiveCharacterTextSplitter.

    Splits text by trying larger semantic boundaries first (paragraphs, then newlines, then spaces).
    """
    size = chunk_size or settings.default_chunk_size
    overlap = chunk_overlap or settings.default_chunk_overlap
    default_seps = separators or ["\n\n", "\n", ". ", " ", ""]

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=default_seps,
        length_function=len,
        is_separator_regex=False,
    )


def create_token_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    encoding_name: str = "cl100k_base",
) -> TokenTextSplitter:
    """Creates a TokenTextSplitter measuring chunks by token count rather than character count."""
    size = chunk_size or (settings.default_chunk_size // 4)
    overlap = chunk_overlap or (settings.default_chunk_overlap // 4)

    return TokenTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        encoding_name=encoding_name,
    )


def create_markdown_header_splitter(
    headers_to_split_on: Optional[List[tuple]] = None,
) -> MarkdownHeaderTextSplitter:
    """Creates a MarkdownHeaderTextSplitter that extracts header hierarchies into metadata."""
    default_headers = headers_to_split_on or [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=default_headers,
        strip_headers=False,
    )


def create_semantic_splitter(
    embeddings=None,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 90.0,
):
    """Creates a SemanticChunker using a dedicated embedding model.

    Splits text when consecutive sentences exhibit a significant semantic embedding drop.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker

        if embeddings is None:
            # Fallback to recursive if no embedding model passed
            return create_recursive_splitter()

        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )
    except ImportError:
        # Fallback if langchain_experimental is not installed
        return create_recursive_splitter()


def get_text_splitter(
    splitter_type: Optional[Literal["recursive", "token", "markdown", "semantic"]] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embeddings=None,
):
    """Factory to retrieve a configured text splitter."""
    stype = splitter_type or settings.default_splitter_type

    if stype == "recursive":
        return create_recursive_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif stype == "token":
        return create_token_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif stype == "markdown":
        return create_markdown_header_splitter()
    elif stype == "semantic":
        return create_semantic_splitter(embeddings=embeddings)
    else:
        return create_recursive_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_documents(
    documents: List[Document],
    splitter=None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Splits a list of documents into chunks, enriching each chunk with positional metadata.

    Preserves parent metadata and injects:
    - `chunk_index`: 0-indexed position within parent doc
    - `total_chunks`: total chunks produced for this parent doc
    - `parent_doc_id`: parent document ID
    - `chunk_id`: unique ID '{parent_doc_id}_chunk_{i}'
    """
    active_splitter = splitter or create_recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: List[Document] = []

    for doc in documents:
        # If document is markdown and markdown splitter selected
        if isinstance(active_splitter, MarkdownHeaderTextSplitter):
            chunks = active_splitter.split_text(doc.page_content)
        else:
            chunks = active_splitter.split_documents([doc])

        total = len(chunks)
        parent_id = doc.metadata.get("doc_id", "doc")

        for idx, chunk in enumerate(chunks):
            # Inherit and enrich metadata
            meta = dict(doc.metadata)
            meta.update(chunk.metadata)
            meta["chunk_index"] = idx
            meta["total_chunks"] = total
            meta["parent_doc_id"] = parent_id
            meta["chunk_id"] = f"{parent_id}_c{idx:03d}"
            meta["char_count"] = len(chunk.page_content)
            meta["word_count"] = len(chunk.page_content.split())
            meta["content_hash"] = calculate_content_hash(chunk.page_content)

            all_chunks.append(Document(page_content=chunk.page_content, metadata=meta))

    return all_chunks
