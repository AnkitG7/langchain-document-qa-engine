"""Document Catalog and Metadata Inspection Tool for DocMind Agents.

Demonstrates:
- Querying document metadata, filenames, file types, and index inventory
- Pydantic args_schema
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class DocumentCatalogInput(BaseModel):
    """Schema for document catalog tool arguments."""
    action: Literal["list_all", "find_by_type", "get_file_info"] = Field(
        default="list_all",
        description="The action to perform: 'list_all' to list all files, 'find_by_type' to find files of a specific type, 'get_file_info' for detailed info on a specific file.",
    )
    file_type: Optional[str] = Field(
        default=None,
        description="File extension or type to filter by (e.g. 'csv', 'pdf', 'md', 'txt') when action='find_by_type'.",
    )
    filename: Optional[str] = Field(
        default=None,
        description="Target filename (e.g. 'sample_data.csv') when action='get_file_info'.",
    )


@tool("query_document_catalog", args_schema=DocumentCatalogInput)
def metadata_catalog_tool(
    action: Literal["list_all", "find_by_type", "get_file_info"] = "list_all",
    file_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Inspect the document catalog, list uploaded files, and query file properties and types."""
    data_dir = Path("data")
    if not data_dir.exists():
        return "Catalog is empty: No 'data' directory found."

    files = [f for f in data_dir.iterdir() if f.is_file()]
    if not files:
        return "Catalog is empty: No document files uploaded yet."

    if action == "list_all":
        lines = ["Document Catalog Inventory:"]
        for f in sorted(files, key=lambda x: x.name):
            size_kb = f.stat().st_size / 1024
            ext = f.suffix.lstrip(".").lower() or "unknown"
            lines.append(f"- {f.name} (Type: {ext}, Size: {size_kb:.1f} KB)")
        return "\n".join(lines)

    elif action == "find_by_type":
        if not file_type:
            return "Please provide 'file_type' parameter to search by type."
        target_ext = f".{file_type.lstrip('.').lower()}"
        matching = [f for f in files if f.suffix.lower() == target_ext]
        if not matching:
            return f"No documents found with file type '{file_type}'."
        lines = [f"Documents matching type '{file_type}':"]
        for f in matching:
            size_kb = f.stat().st_size / 1024
            lines.append(f"- {f.name} ({size_kb:.1f} KB)")
        return "\n".join(lines)

    elif action == "get_file_info":
        if not filename:
            return "Please provide 'filename' parameter to inspect file info."
        target = data_dir / filename
        if not target.exists():
            return f"File '{filename}' not found in catalog."
        stat = target.stat()
        return (
            f"File Details for '{filename}':\n"
            f"- Full Path: {target.resolve()}\n"
            f"- Extension: {target.suffix}\n"
            f"- Size: {stat.st_size} bytes ({stat.st_size / 1024:.2f} KB)\n"
            f"- Last Modified Timestamp: {stat.st_mtime}"
        )

    return f"Unknown catalog action: {action}"
