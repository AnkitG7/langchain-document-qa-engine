"""Sparse Lexical Retrieval (BM25) for Exact Keyword Matching.

Demonstrates:
- BM25 term frequency / inverse document frequency scoring
- Exact keyword, acronym, code identifier, and alphanumeric token matching
- Complements dense semantic vector stores where embeddings have keyword blind spots
"""

import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


def default_tokenizer(text: str) -> List[str]:
    """Tokenizes text by lowercase words, numbers, and technical identifiers."""
    return re.findall(r"\w+", text.lower())


def create_bm25_retriever(
    documents: List[Document],
    k: int = 4,
) -> BM25Retriever:
    """Creates a BM25 lexical retriever from a list of Document chunks."""
    if not documents:
        # Create empty placeholder document to avoid empty index errors
        documents = [Document(page_content="empty", metadata={"source": "none"})]

    retriever = BM25Retriever.from_documents(
        documents=documents,
        preprocess_func=default_tokenizer,
        k=k,
    )
    return retriever


class BM25Index:
    """Convenience wrapper for BM25 sparse lexical indexing and retrieval."""

    def __init__(self, documents: Optional[List[Document]] = None, k: int = 4):
        self.k = k
        self.documents = documents or []
        self.retriever = create_bm25_retriever(self.documents, k=self.k) if self.documents else None

    def index_documents(self, documents: List[Document]) -> None:
        """Indexes or re-indexes a list of Document objects."""
        self.documents = documents
        self.retriever = create_bm25_retriever(documents, k=self.k)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Performs BM25 lexical keyword search."""
        if not self.retriever or not self.documents:
            return []
        if k is not None:
            self.retriever.k = k
        return self.retriever.invoke(query)
