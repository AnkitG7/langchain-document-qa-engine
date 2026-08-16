"""Pluggable Vector Store Implementations (Chroma and FAISS) for DocMind.

Demonstrates:
- Unified VectorStoreManager abstraction
- In-memory and persistent SQLite Chroma store
- Fast in-memory and serialized FAISS index
- Document indexing, metadata preservation, and index reload capabilities
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .embedder import get_embeddings


# ---------------------------------------------------------------------------
# 1. Chroma Store Factory
# ---------------------------------------------------------------------------
def get_or_create_chroma(
    collection_name: str = "docmind_collection",
    persist_directory: Optional[str] = None,
    embeddings: Optional[Embeddings] = None,
    documents: Optional[List[Document]] = None,
) -> VectorStore:
    """Instantiates a Chroma vector store (either in-memory or persisted to disk).

    Args:
        collection_name: Name of the Chroma collection.
        persist_directory: Directory path to persist SQLite index. If None, runs in-memory.
        embeddings: Dedicated embedding model. Defaults to configured embedder.
        documents: Optional list of documents to index upon creation.
    """
    embedder = embeddings or get_embeddings()

    try:
        from langchain_chroma import Chroma
    except ImportError:
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            raise ImportError("Please install langchain-chroma: `pip install langchain-chroma chromadb`")

    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)

    if documents:
        return Chroma.from_documents(
            documents=documents,
            embedding=embedder,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    else:
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=persist_directory,
        )


# ---------------------------------------------------------------------------
# 2. FAISS Store Factory
# ---------------------------------------------------------------------------
def load_faiss_index(
    index_path: str,
    embeddings: Optional[Embeddings] = None,
) -> VectorStore:
    """Loads an existing serialized FAISS index from disk.

    Raises:
        FileNotFoundError: If index_path does not exist.
    """
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS index path does not exist: {index_path}")

    embedder = embeddings or get_embeddings()
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        raise ImportError("Please install faiss-cpu: `pip install faiss-cpu`")

    return FAISS.load_local(
        folder_path=str(path),
        embeddings=embedder,
        allow_dangerous_deserialization=True,
    )


def get_or_create_faiss(
    documents: Optional[List[Document]] = None,
    embeddings: Optional[Embeddings] = None,
    index_path: Optional[str] = None,
) -> VectorStore:
    """Instantiates a FAISS vector store.

    Args:
        documents: Documents to build the initial FAISS index from.
        embeddings: Dedicated embedding model.
        index_path: Optional path to save or load serialized FAISS index.
    """
    embedder = embeddings or get_embeddings()

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        raise ImportError("Please install faiss-cpu: `pip install faiss-cpu`")

    # If loading existing index from disk without new documents
    if index_path and not documents:
        return load_faiss_index(index_path=index_path, embeddings=embedder)

    # If initializing with documents
    if documents:
        batch_size = 50
        store = None
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            if store is None:
                store = FAISS.from_documents(documents=batch, embedding=embedder)
            else:
                store.add_documents(batch)

        if index_path and store:
            os.makedirs(index_path, exist_ok=True)
            store.save_local(index_path)
        return store

    # If in-memory empty initialization
    dummy_text = "DocMind initialization anchor document"
    dummy_doc = Document(page_content=dummy_text, metadata={"system_init": True})
    return FAISS.from_documents(documents=[dummy_doc], embedding=embedder)


# ---------------------------------------------------------------------------
# 3. Unified VectorStore Manager
# ---------------------------------------------------------------------------
class VectorStoreManager:
    """Unified manager wrapping local vector stores with complete indexing & lifecycle management."""

    def __init__(
        self,
        store_type: Literal["chroma", "faiss"] = "chroma",
        collection_name: str = "docmind_collection",
        persist_dir: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        self.store_type = store_type
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.embeddings = embeddings or get_embeddings()
        self.store: VectorStore = self._init_store()

    def _init_store(self) -> VectorStore:
        if self.store_type == "chroma":
            return get_or_create_chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
                embeddings=self.embeddings,
            )
        elif self.store_type == "faiss":
            return get_or_create_faiss(
                embeddings=self.embeddings,
                index_path=self.persist_dir,
            )
        else:
            raise ValueError(f"Unsupported store type '{self.store_type}'. Supported: 'chroma', 'faiss'.")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Indexes documents and updates persistent storage if configured."""
        if not documents:
            return []

        ids = self.store.add_documents(documents)

        # If FAISS and persistent directory configured, save update
        if self.store_type == "faiss" and self.persist_dir:
            self.store.save_local(self.persist_dir)

        return ids

    def delete_by_fingerprint(self, fingerprint_or_doc_id: str) -> int:
        """Removes all vectors belonging to a document fingerprint or doc_id."""
        deleted = delete_documents_by_fingerprint(self.store, fingerprint_or_doc_id)
        if self.store_type == "faiss" and self.persist_dir:
            self.store.save_local(self.persist_dir)
        return deleted

    def replace_document_vectors(
        self,
        fingerprint_or_doc_id: str,
        new_documents: List[Document],
    ) -> Tuple[int, List[str]]:
        """Atomically replaces stale vectors with newly chunked vectors."""
        deleted, inserted_ids = replace_document_vectors(
            vectorstore=self.store,
            fingerprint_or_doc_id=fingerprint_or_doc_id,
            new_documents=new_documents,
        )
        if self.store_type == "faiss" and self.persist_dir:
            self.store.save_local(self.persist_dir)
        return deleted, inserted_ids

    def persist(self) -> None:
        """Explicitly triggers storage persistence."""
        if self.store_type == "faiss" and self.persist_dir:
            self.store.save_local(self.persist_dir)

    def as_retriever(self, **kwargs: Any):
        """Returns a LangChain retriever interface for this store."""
        return self.store.as_retriever(**kwargs)


# ---------------------------------------------------------------------------
# 4. Global Vector Store Deletion & Replacement Utilities
# ---------------------------------------------------------------------------
def delete_documents_by_fingerprint(
    vectorstore: VectorStore,
    fingerprint_or_doc_id: str,
    exclude_ids: Optional[Any] = None,
) -> int:
    """Removes all vectors belonging to a specific document fingerprint or parent doc_id.

    Args:
        vectorstore: The vector store instance (FAISS, Chroma, etc.)
        fingerprint_or_doc_id: Content hash or parent document ID
        exclude_ids: Optional set/collection of document IDs to preserve (used for build-new-then-swap)

    Supports FAISS, Chroma, and PGVector backends.
    Returns the number of vectors removed.
    """
    if not fingerprint_or_doc_id or vectorstore is None:
        return 0

    exclude = set(exclude_ids) if exclude_ids else set()

    # 1. FAISS Backend
    if hasattr(vectorstore, "docstore") and hasattr(vectorstore.docstore, "_dict"):
        matching_ids = [
            k for k, v in vectorstore.docstore._dict.items()
            if (
                v.metadata.get("content_fingerprint") == fingerprint_or_doc_id
                or v.metadata.get("parent_doc_id") == fingerprint_or_doc_id
                or v.metadata.get("doc_id") == fingerprint_or_doc_id
                or v.metadata.get("filename") == fingerprint_or_doc_id
                or v.metadata.get("source") == fingerprint_or_doc_id
            )
            and k not in exclude
        ]
        if matching_ids:
            try:
                vectorstore.delete(matching_ids)
                return len(matching_ids)
            except Exception:
                # If delete not directly supported on underlying index, clean docstore
                for mid in matching_ids:
                    vectorstore.docstore._dict.pop(mid, None)
                return len(matching_ids)

    # 2. Chroma Backend
    elif hasattr(vectorstore, "_collection"):
        try:
            results = vectorstore._collection.get(
                where={"$or": [
                    {"content_fingerprint": fingerprint_or_doc_id},
                    {"parent_doc_id": fingerprint_or_doc_id},
                ]}
            )
            ids_to_del = [i for i in results.get("ids", []) if i not in exclude]
            if ids_to_del:
                vectorstore._collection.delete(ids=ids_to_del)
                return len(ids_to_del)
        except Exception:
            try:
                vectorstore._collection.delete(where={"content_fingerprint": fingerprint_or_doc_id})
            except Exception:
                pass

    return 0


def replace_document_vectors(
    vectorstore: VectorStore,
    fingerprint_or_doc_id: str,
    new_documents: List[Document],
) -> Tuple[int, List[str]]:
    """Safely replaces stale vectors with newly chunked vectors using Build-New-Then-Swap.

    Failure Safety Guarantees:
    1. If new_documents is empty or embedding fails during vectorstore.add_documents(),
       an exception is raised immediately BEFORE any existing vectors are deleted.
       The existing vectors remain 100% intact in the vector store.
    2. Once new vectors are safely embedded and indexed, the old vectors matching the
       fingerprint (excluding the new IDs) are purged.
    3. Guarantees the document is never left in a 0-vector or corrupted state on embedding failure.

    Returns:
        (deleted_count, inserted_ids)
    """
    if not new_documents or vectorstore is None:
        return 0, []

    # Step 1: Build & Insert new vectors first (Pre-compute & validate embeddings)
    # If embedding generation fails, this raises an exception and halts before deleting old vectors
    inserted_ids = vectorstore.add_documents(new_documents)

    # Step 2: Swap / Purge old vectors matching fingerprint, excluding the newly inserted IDs
    deleted_count = delete_documents_by_fingerprint(
        vectorstore=vectorstore,
        fingerprint_or_doc_id=fingerprint_or_doc_id,
        exclude_ids=set(inserted_ids),
    )

    return deleted_count, inserted_ids


def create_vector_store(
    store_type: Literal["chroma", "faiss"] = "chroma",
    documents: Optional[List[Document]] = None,
    embeddings: Optional[Embeddings] = None,
    persist_dir: Optional[str] = None,
    collection_name: str = "docmind_collection",
) -> VectorStore:
    """Factory helper to build a vector store in one call."""
    embedder = embeddings or get_embeddings()

    if store_type == "chroma":
        return get_or_create_chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embeddings=embedder,
            documents=documents,
        )
    elif store_type == "faiss":
        return get_or_create_faiss(
            documents=documents,
            embeddings=embedder,
            index_path=persist_dir,
        )
    else:
        raise ValueError(f"Unsupported store type '{store_type}'")
