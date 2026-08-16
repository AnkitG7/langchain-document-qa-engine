"""Tools module for DocMind agents.

Exports:
- DocumentSearchTool: Vector store similarity and MMR search tool
- CalculatorTool: Safe mathematical and quantitative analysis tool
- MetadataCatalogTool: Document provenance and catalog querying tool
- get_docmind_tools: Factory to assemble the complete tool suite
"""

from typing import List, Optional
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore

from .search_tool import create_search_tool, SearchDocumentsInput
from .calculator_tool import calculator_tool, CalculatorInput
from .metadata_tool import metadata_catalog_tool, DocumentCatalogInput


def get_docmind_tools(vectorstore: Optional[VectorStore] = None) -> List[BaseTool]:
    """Returns the full suite of specialized tools available to the DocMind agent."""
    tools: List[BaseTool] = [
        calculator_tool,
        metadata_catalog_tool,
    ]

    if vectorstore is not None:
        tools.append(create_search_tool(vectorstore))

    return tools


__all__ = [
    "create_search_tool",
    "SearchDocumentsInput",
    "calculator_tool",
    "CalculatorInput",
    "metadata_catalog_tool",
    "DocumentCatalogInput",
    "get_docmind_tools",
]
