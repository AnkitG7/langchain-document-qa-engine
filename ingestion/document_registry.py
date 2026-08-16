"""Production Document Fingerprinting & Ingestion Cache Registry.

Demonstrates:
- True content-based binary SHA-256 document fingerprinting (content identity, not filename)
- Ingestion Configuration Signatures: Invalidation when chunk_size, overlap, splitter, parser, or embedding model changes
- Persistent Registry on disk surviving application restarts
- Prevention of redundant PDF parsing, tokenization, chunking, and embedding generation
"""

import os
import json
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


def calculate_file_sha256(file_path_or_bytes: Union[str, Path, bytes]) -> str:
    """Computes deterministic SHA-256 hash of raw file bytes."""
    hasher = hashlib.sha256()
    if isinstance(file_path_or_bytes, bytes):
        hasher.update(file_path_or_bytes)
    else:
        path = Path(file_path_or_bytes)
        if not path.exists():
            raise FileNotFoundError(f"File not found for hash calculation: {path}")
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def compute_config_signature(
    chunk_size: int,
    chunk_overlap: int,
    splitter_type: str = "recursive",
    parser_type: str = "standard",
    embedding_model: str = "nomic-embed-text",
    extra_options: Optional[Dict[str, Any]] = None,
) -> str:
    """Generates a deterministic hash representing the exact ingestion configuration.
    
    If any parameter that affects chunk boundaries, parsing logic, or vector embeddings changes,
    the signature changes, triggering safe cache invalidation.
    """
    config_dict = {
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "splitter_type": str(splitter_type).lower(),
        "parser_type": str(parser_type).lower(),
        "embedding_model": str(embedding_model).lower(),
        "extra_options": extra_options or {},
    }
    serialized = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class DocumentRegistryEntry(BaseModel):
    """Metadata record for an indexed document in the persistent registry."""
    content_fingerprint: str = Field(description="SHA-256 hash of raw file bytes")
    config_signature: str = Field(description="SHA-256 hash of the ingestion configuration")
    doc_id: str = Field(description="Unique parent document identifier")
    filenames: List[str] = Field(default_factory=list, description="Known filenames for this content")
    file_type: str = Field(default="pdf", description="File extension without dot")
    file_size_bytes: int = Field(default=0, description="Size of file in bytes")
    chunks_count: int = Field(default=0, description="Total chunks created in index")
    character_count: int = Field(default=0, description="Total characters across chunks")
    config_details: Dict[str, Any] = Field(default_factory=dict, description="Human-readable config snapshot")
    indexed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_accessed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DocumentRegistry:
    """Thread-safe persistent registry of ingested documents and their config signatures."""

    def __init__(self, registry_file: Optional[Union[str, Path]] = None):
        self.registry_file = Path(registry_file or "data/document_registry.json")
        self._lock = threading.Lock()
        self._entries: Dict[str, DocumentRegistryEntry] = {}
        self._load()

    def _load(self) -> None:
        """Loads persistent registry from disk if it exists."""
        with self._lock:
            if self.registry_file.exists():
                try:
                    with open(self.registry_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._entries = {
                            k: DocumentRegistryEntry.model_validate(v)
                            for k, v in data.items()
                        }
                except Exception:
                    self._entries = {}
            else:
                self._entries = {}

    def _save(self) -> None:
        """Saves registry to disk."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.model_dump() for k, v in self._entries.items()}
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_entry(self, content_fingerprint: str) -> Optional[DocumentRegistryEntry]:
        """Retrieves entry by content fingerprint."""
        with self._lock:
            return self._entries.get(content_fingerprint)

    def lookup(
        self,
        content_fingerprint: str,
        config_signature: str,
    ) -> Tuple[bool, Optional[DocumentRegistryEntry], str]:
        """Checks if a document with exact content and matching config is already indexed.
        
        Returns:
            (is_valid_cache, entry, reason)
            - (True, entry, "MATCH"): Exact content and matching config -> reuse index.
            - (False, entry, "CONFIG_DRIFT"): Content exists but config changed -> re-process.
            - (False, None, "NOT_FOUND"): Content has never been indexed -> process normally.
        """
        with self._lock:
            entry = self._entries.get(content_fingerprint)
            if entry is None:
                return False, None, "NOT_FOUND"

            if entry.config_signature == config_signature:
                entry.last_accessed_at = datetime.now(timezone.utc).isoformat()
                self._save()
                return True, entry, "MATCH"
            else:
                return False, entry, "CONFIG_DRIFT"

    def register(
        self,
        content_fingerprint: str,
        config_signature: str,
        doc_id: str,
        filename: str,
        file_type: str,
        file_size_bytes: int,
        chunks_count: int,
        character_count: int,
        config_details: Optional[Dict[str, Any]] = None,
    ) -> DocumentRegistryEntry:
        """Registers a newly indexed document or updates an existing entry."""
        with self._lock:
            existing = self._entries.get(content_fingerprint)
            filenames = existing.filenames if existing else []
            if filename not in filenames:
                filenames.append(filename)

            entry = DocumentRegistryEntry(
                content_fingerprint=content_fingerprint,
                config_signature=config_signature,
                doc_id=doc_id,
                filenames=filenames,
                file_type=file_type,
                file_size_bytes=file_size_bytes,
                chunks_count=chunks_count,
                character_count=character_count,
                config_details=config_details or {},
                indexed_at=datetime.now(timezone.utc).isoformat(),
                last_accessed_at=datetime.now(timezone.utc).isoformat(),
            )
            self._entries[content_fingerprint] = entry
            self._save()
            return entry

    def record_alias_filename(self, content_fingerprint: str, filename: str) -> None:
        """Associates a new filename with an already indexed content fingerprint."""
        with self._lock:
            entry = self._entries.get(content_fingerprint)
            if entry and filename not in entry.filenames:
                entry.filenames.append(filename)
                entry.last_accessed_at = datetime.now(timezone.utc).isoformat()
                self._save()

    def list_entries(self) -> List[DocumentRegistryEntry]:
        """Lists all registered document entries."""
        with self._lock:
            return list(self._entries.values())

    def delete(self, content_fingerprint: str) -> bool:
        """Deletes an entry by fingerprint."""
        with self._lock:
            if content_fingerprint in self._entries:
                del self._entries[content_fingerprint]
                self._save()
                return True
            return False

    def clear(self) -> None:
        """Clears all registry entries."""
        with self._lock:
            self._entries.clear()
            self._save()


# Global default registry instance
_default_registry: Optional[DocumentRegistry] = None
_registry_lock = threading.Lock()


def get_document_registry(registry_file: Optional[Union[str, Path]] = None) -> DocumentRegistry:
    """Returns singleton document registry instance."""
    global _default_registry
    with _registry_lock:
        if _default_registry is None or registry_file is not None:
            _default_registry = DocumentRegistry(registry_file=registry_file)
        return _default_registry
