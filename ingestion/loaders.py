"""Multi-format Document Loaders for DocMind.

Demonstrates:
- LangChain Document schema (page_content + metadata)
- PDF loading with page tracking (PyPDF / pypdf)
- Plain text and Markdown loading
- CSV tabular data loading with row-level metadata
- Web URL loading with HTML tag cleaning (WebBaseLoader / BeautifulSoup)
- Universal loader factory dispatching on source type
"""

import os
import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# 1. Text & Markdown Loader
# ---------------------------------------------------------------------------
class TextDocumentLoader:
    """Loads plain text files (.txt, .log, .rst, etc.) with encoding detection."""

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {self.file_path}")

        try:
            with open(path, "r", encoding=self.encoding, errors="replace") as f:
                content = f.read()
        except Exception as e:
            raise RuntimeError(f"Error reading {self.file_path}: {e}")

        metadata = {
            "source": str(path.resolve()),
            "filename": path.name,
            "file_type": path.suffix.lower().lstrip(".") or "text",
            "file_size_bytes": path.stat().st_size,
        }
        return [Document(page_content=content, metadata=metadata)]


class MarkdownLoader:
    """Loads Markdown documents with structural metadata."""

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {self.file_path}")

        with open(path, "r", encoding=self.encoding, errors="replace") as f:
            content = f.read()

        metadata = {
            "source": str(path.resolve()),
            "filename": path.name,
            "file_type": "markdown",
            "file_size_bytes": path.stat().st_size,
        }
        return [Document(page_content=content, metadata=metadata)]


# ---------------------------------------------------------------------------
# 2. PDF Loader
# ---------------------------------------------------------------------------
class PDFDocumentLoader:
    """High-fidelity PDF document loader using pypdf or PyPDFLoader.

    Extracts page-by-page text, preserving page numbers in metadata.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        documents: List[Document] = []

        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            total_pages = len(reader.pages)

            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                metadata = {
                    "source": str(path.resolve()),
                    "filename": path.name,
                    "file_type": "pdf",
                    "page": page_idx,
                    "total_pages": total_pages,
                    "file_size_bytes": path.stat().st_size,
                }
                documents.append(Document(page_content=page_text, metadata=metadata))

        except ImportError:
            # Fallback to langchain_community PyPDFLoader if available
            try:
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader(str(path))
                documents = loader.load()
            except ImportError:
                raise ImportError("Please install pypdf: `pip install pypdf`")

        return documents


# ---------------------------------------------------------------------------
# 3. CSV Tabular Loader
# ---------------------------------------------------------------------------
class CSVDocumentLoader:
    """Loads CSV files, converting rows into structured text with metadata.

    Supports custom row formatting or converting tabular datasets into readable narratives.
    """

    def __init__(
        self,
        file_path: str,
        delimiter: str = ",",
        content_columns: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ):
        self.file_path = file_path
        self.delimiter = delimiter
        self.content_columns = content_columns
        self.encoding = encoding

    def load(self) -> List[Document]:
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        documents: List[Document] = []

        with open(path, mode="r", encoding=self.encoding, errors="replace") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            headers = reader.fieldnames or []

            for row_idx, row in enumerate(reader, start=1):
                # Build narrative content for this row
                if self.content_columns:
                    items = [f"{col}: {row[col]}" for col in self.content_columns if col in row]
                else:
                    items = [f"{k}: {v}" for k, v in row.items() if v]

                row_content = "\n".join(items)
                metadata = {
                    "source": str(path.resolve()),
                    "filename": path.name,
                    "file_type": "csv",
                    "row": row_idx,
                    "headers": list(headers),
                    "file_size_bytes": path.stat().st_size,
                }
                documents.append(Document(page_content=row_content, metadata=metadata))

        return documents


# ---------------------------------------------------------------------------
# 4. Web URL Loader
# ---------------------------------------------------------------------------
class WebDocumentLoader:
    """Loads web pages from URLs, stripping HTML boilerplate and scripts."""

    def __init__(self, url: str, timeout: int = 15):
        self.url = url
        self.timeout = timeout

    def load(self) -> List[Document]:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36 DocMind/1.0"
            )
        }

        response = requests.get(self.url, headers=headers, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Strip scripts, styles, navbars, and footers for clean RAG ingestion
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.decompose()

        page_title = soup.title.string.strip() if soup.title and soup.title.string else self.url
        body_text = soup.get_text(separator="\n", strip=True)

        metadata = {
            "source": self.url,
            "title": page_title,
            "file_type": "web",
            "status_code": response.status_code,
        }

        return [Document(page_content=body_text, metadata=metadata)]


# ---------------------------------------------------------------------------
# 5. Universal Loader Factory & Batch Loader
# ---------------------------------------------------------------------------
def load_document(source: str, **kwargs: Any) -> List[Document]:
    """Universal loader factory that routes to the appropriate loader based on source.

    Supports:
    - URLs: http:// or https:// -> WebDocumentLoader
    - PDF: .pdf -> PDFDocumentLoader
    - CSV: .csv -> CSVDocumentLoader
    - Markdown: .md, .markdown -> MarkdownLoader
    - Text: .txt, .log, .json, .yaml, .rst -> TextDocumentLoader
    """
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        return WebDocumentLoader(url=source, **kwargs).load()

    path = Path(source)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return PDFDocumentLoader(file_path=source).load()
    elif suffix == ".csv":
        return CSVDocumentLoader(file_path=source, **kwargs).load()
    elif suffix in (".md", ".markdown"):
        return MarkdownLoader(file_path=source, **kwargs).load()
    elif suffix in (".txt", ".log", ".rst", ".json", ".yaml", ".yml", ".py", ".html"):
        return TextDocumentLoader(file_path=source, **kwargs).load()
    else:
        # Default fallback to TextDocumentLoader
        return TextDocumentLoader(file_path=source, **kwargs).load()


def load_documents_batch(sources: List[str], **kwargs: Any) -> List[Document]:
    """Loads a list of heterogeneous document sources (files and URLs) into a single Document list."""
    all_docs: List[Document] = []
    for src in sources:
        docs = load_document(src, **kwargs)
        all_docs.extend(docs)
    return all_docs
