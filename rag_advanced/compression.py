"""Contextual Document Compression Module.

Demonstrates:
- Extractive context compression
- Stripping irrelevant sentences from retrieved chunks before prompting the generator
- Maximizing signal-to-noise ratio and token budget efficiency
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from llm.provider import get_chat_model


class ContextualCompressor:
    """Extracts only query-relevant facts from document chunks, dropping irrelevant fluff."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm or get_chat_model()

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert context extractor. Given a question and a document excerpt, "
                "extract ONLY the specific sentences or data points directly relevant to answering the question. "
                "Do NOT paraphrase or add commentary. If nothing in the document is relevant, output 'NO_RELEVANT_CONTENT'.",
            ),
            (
                "human",
                "Question: {question}\n\nDocument Excerpt:\n{context}\n\nRelevant Sentences:",
            ),
        ])
        self.compression_chain = prompt | self.llm | StrOutputParser()

    def compress_document(self, query: str, doc: Document) -> Optional[Document]:
        """Compresses a single Document chunk."""
        try:
            extracted = self.compression_chain.invoke({
                "question": query,
                "context": doc.page_content,
            }).strip()

            if not extracted or "NO_RELEVANT_CONTENT" in extracted:
                return None

            # Create a compressed copy of the document
            return Document(
                page_content=extracted,
                metadata={**doc.metadata, "original_length": len(doc.page_content), "compressed": True},
            )
        except Exception:
            return doc

    def compress(self, query: str, documents: List[Document]) -> List[Document]:
        """Compresses a list of documents, discarding chunks with no relevant information."""
        compressed_docs = []
        for doc in documents:
            compressed = self.compress_document(query, doc)
            if compressed is not None:
                compressed_docs.append(compressed)

        # If all were discarded, return original documents as safe fallback
        return compressed_docs if compressed_docs else documents
