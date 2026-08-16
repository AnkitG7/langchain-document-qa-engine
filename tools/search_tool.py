"""Vector Document Search Tool for DocMind Agents.

Demonstrates:
- @tool decorator with explicit Pydantic args_schema
- Vector retrieval tool integration (Chroma / FAISS)
- Metadata filtering via tool arguments
- Detailed result formatting with citations
"""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import VectorStore

from vectorstore.retriever import create_retriever


class SearchDocumentsInput(BaseModel):
    """Schema for document search tool arguments."""
    query: str = Field(
        description="The semantic search query describing the information you need to find in uploaded documents."
    )
    k: int = Field(
        default=4,
        description="Number of relevant document snippets to retrieve (between 1 and 10).",
        ge=1,
        le=10,
    )
    file_type: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict search to a specific file type (e.g. 'pdf', 'csv', 'markdown', 'txt').",
    )


def create_search_tool(vectorstore: VectorStore, default_k: int = 4) -> BaseTool:
    """Creates a LangChain BaseTool wrapping the vector store retriever."""

    @tool("search_documents", args_schema=SearchDocumentsInput)
    def search_documents(
        query: str,
        k: int = default_k,
        file_type: Optional[str] = None,
    ) -> str:
        """Search the indexed document knowledge base for relevant facts, excerpts, and tabular data."""
        if not query or not query.strip():
            return "Search query was empty. Please provide a specific search query."

        filter_dict = {"file_type": file_type.lower()} if file_type else None

        retriever = create_retriever(
            vectorstore=vectorstore,
            search_type="similarity",
            k=k,
            filter_dict=filter_dict,
        )

        docs = retriever.invoke(query)
        if not docs:
            filter_info = f" with file_type='{file_type}'" if file_type else ""
            return f"No relevant documents found for query: '{query}'{filter_info}."

        formatted_results = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("filename", doc.metadata.get("source", f"Document {idx}"))
            ftype = doc.metadata.get("file_type", "unknown")
            page_or_row = ""
            if "page" in doc.metadata:
                page_or_row = f", Page {doc.metadata['page']}"
            elif "row" in doc.metadata:
                page_or_row = f", Row {doc.metadata['row']}"

            header_info = f"[{idx}] Source: {source} (Type: {ftype}{page_or_row})"
            content = doc.page_content.strip()
            formatted_results.append(f"{header_info}\n{content}")

        return "\n\n".join(formatted_results)

    return search_documents
