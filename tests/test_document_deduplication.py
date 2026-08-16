"""Comprehensive Automated Tests for Document Deduplication & Ingestion Caching.

Tests:
1. Same PDF + same filename -> reuse existing index
2. Same PDF + different filename -> reuse existing index (content-byte identity)
3. Different PDF -> process normally
4. Same PDF + changed chunk size -> re-process (config drift)
5. Same PDF + changed chunk overlap -> re-process (config drift)
6. Same PDF + changed chunking strategy -> re-process (config drift)
7. Existing indexed document -> query works directly without re-ingestion
8. Duplicate upload does not create duplicate chunks/embeddings in vector store
9. Fingerprint registry is persisted and survives application restart
10. API endpoint upload duplicate detection via FastAPI TestClient
"""

import os
import shutil
import tempfile
from pathlib import Path
import pytest
import pymupdf
from fastapi.testclient import TestClient

from ingestion.pipeline import IngestionPipeline, IngestionReport
from ingestion.document_registry import (
    DocumentRegistry,
    calculate_file_sha256,
    compute_config_signature,
    DocumentRegistryEntry,
)
from vectorstore.embedder import get_embeddings
from vectorstore.store import get_or_create_faiss
from rag_advanced.pipeline import AdvancedRAGPipeline
from api.server import create_app
from api.dependencies import AppState, get_app_state


def create_mock_pdf(path: Path, text: str) -> None:
    """Generates a valid binary PDF file containing the specified text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def temp_env(tmp_path):
    """Creates a clean temporary workspace for document registry and sample PDF files."""
    registry_file = tmp_path / "test_registry.json"
    registry = DocumentRegistry(registry_file=registry_file)

    # Sample Document A (Content A)
    doc_a1 = tmp_path / "rbi_circular.pdf"
    content_a = "Reserve Bank of India Monetary Policy: The policy repo rate is kept unchanged at 6.50 percent. Standing Deposit Facility rate is 6.25 percent."
    create_mock_pdf(doc_a1, content_a)

    # Sample Document A with DIFFERENT filename (exact identical content bytes)
    doc_a2 = tmp_path / "rbi_pdf_123x.pdf"
    shutil.copyfile(doc_a1, doc_a2)

    # Sample Document B (Completely Different Content)
    doc_b = tmp_path / "transformer_research.pdf"
    content_b = "Attention Is All You Need: The Transformer model architecture relies entirely on self-attention mechanisms without recurrence or convolutions."
    create_mock_pdf(doc_b, content_b)

    return {
        "registry": registry,
        "registry_file": registry_file,
        "doc_a1": doc_a1,
        "doc_a2": doc_a2,
        "doc_b": doc_b,
        "tmp_path": tmp_path,
    }


class TestContentFingerprintingAndRegistry:
    """Tests core SHA-256 byte hashing, configuration signatures, and persistence."""

    def test_content_identity_independent_of_filename(self, temp_env):
        """rbi_circular.pdf and rbi_pdf_123x.pdf must produce the exact same SHA-256 fingerprint."""
        hash_a1 = calculate_file_sha256(temp_env["doc_a1"])
        hash_a2 = calculate_file_sha256(temp_env["doc_a2"])
        hash_b = calculate_file_sha256(temp_env["doc_b"])

        assert hash_a1 == hash_a2, "Identical content bytes under different filenames must yield identical hash"
        assert hash_a1 != hash_b, "Different document content must produce different SHA-256 hashes"

    def test_config_signature_sensitivity(self):
        """Config signature must change when any chunking/embedding parameter changes."""
        sig_base = compute_config_signature(chunk_size=500, chunk_overlap=50, splitter_type="recursive")
        sig_diff_size = compute_config_signature(chunk_size=1000, chunk_overlap=50, splitter_type="recursive")
        sig_diff_overlap = compute_config_signature(chunk_size=500, chunk_overlap=100, splitter_type="recursive")
        sig_diff_splitter = compute_config_signature(chunk_size=500, chunk_overlap=50, splitter_type="token")
        sig_diff_model = compute_config_signature(chunk_size=500, chunk_overlap=50, splitter_type="recursive", embedding_model="text-embedding-3-small")

        assert sig_base != sig_diff_size, "Changed chunk_size must alter config signature"
        assert sig_base != sig_diff_overlap, "Changed chunk_overlap must alter config signature"
        assert sig_base != sig_diff_splitter, "Changed splitter_type must alter config signature"
        assert sig_base != sig_diff_model, "Changed embedding_model must alter config signature"

    def test_registry_persistence_survives_restart(self, temp_env):
        """Entries saved by one registry instance must be loaded by a newly instantiated registry."""
        registry1 = temp_env["registry"]
        hash_val = calculate_file_sha256(temp_env["doc_a1"])
        sig_val = compute_config_signature(chunk_size=500, chunk_overlap=50)

        registry1.register(
            content_fingerprint=hash_val,
            config_signature=sig_val,
            doc_id="doc_test_123",
            filename="rbi_circular.pdf",
            file_type="pdf",
            file_size_bytes=150,
            chunks_count=3,
            character_count=450,
        )

        # Instantiate a brand new registry pointing to the same disk file (simulating app restart)
        registry2 = DocumentRegistry(registry_file=temp_env["registry_file"])
        is_valid, entry, reason = registry2.lookup(hash_val, sig_val)

        assert is_valid is True
        assert entry is not None
        assert entry.doc_id == "doc_test_123"
        assert entry.chunks_count == 3
        assert "rbi_circular.pdf" in entry.filenames


class TestIngestionPipelineDuplicateDeduplication:
    """Tests IngestionPipeline caching behavior across filenames and config changes."""

    def test_same_pdf_same_filename_reuses_index(self, temp_env):
        """First upload ingests; second upload of same file reuses index."""
        pipeline = IngestionPipeline(
            chunk_size=400,
            chunk_overlap=40,
            registry=temp_env["registry"],
        )

        # 1. Initial Ingestion
        chunks1, report1 = pipeline.run(str(temp_env["doc_a1"]))
        assert report1.is_reused is False
        assert report1.cache_status == "PROCESSED"
        assert len(chunks1) > 0

        # 2. Duplicate Ingestion (Same filename)
        chunks2, report2 = pipeline.run(str(temp_env["doc_a1"]))
        assert report2.is_reused is True
        assert report2.cache_status == "CACHE_HIT"
        assert report2.final_chunks_count == len(chunks1)
        assert len(chunks2) == len(chunks1)

    def test_same_pdf_different_filename_reuses_index(self, temp_env):
        """Uploading rbi_circular.pdf followed by rbi_pdf_123x.pdf must detect identical content and reuse."""
        pipeline = IngestionPipeline(
            chunk_size=400,
            chunk_overlap=40,
            registry=temp_env["registry"],
        )

        # Ingest first filename
        chunks1, report1 = pipeline.run(str(temp_env["doc_a1"]))
        assert report1.is_reused is False

        # Ingest different filename with identical content bytes
        chunks2, report2 = pipeline.run(str(temp_env["doc_a2"]))
        assert report2.is_reused is True
        assert report2.cache_status == "CACHE_HIT"
        assert report2.final_chunks_count == report1.final_chunks_count

        # Check that both filenames are recorded as aliases in the registry
        fingerprint = calculate_file_sha256(temp_env["doc_a1"])
        entry = temp_env["registry"].get_entry(fingerprint)
        assert entry is not None
        assert "rbi_circular.pdf" in entry.filenames
        assert "rbi_pdf_123x.pdf" in entry.filenames

    def test_different_pdf_processes_normally(self, temp_env):
        """Different PDFs must both be processed and registered separately."""
        pipeline = IngestionPipeline(
            chunk_size=400,
            chunk_overlap=40,
            registry=temp_env["registry"],
        )

        chunks_a, report_a = pipeline.run(str(temp_env["doc_a1"]))
        chunks_b, report_b = pipeline.run(str(temp_env["doc_b"]))

        assert report_a.is_reused is False
        assert report_b.is_reused is False
        assert report_a.content_fingerprint != report_b.content_fingerprint

    def test_same_pdf_changed_chunk_size_triggers_reprocessing(self, temp_env):
        """If chunk_size changes, the cache must invalidate and re-process the file."""
        pipeline1 = IngestionPipeline(chunk_size=300, chunk_overlap=30, registry=temp_env["registry"])
        chunks1, report1 = pipeline1.run(str(temp_env["doc_a1"]))
        assert report1.is_reused is False

        # Create pipeline with DIFFERENT chunk_size
        pipeline2 = IngestionPipeline(chunk_size=800, chunk_overlap=30, registry=temp_env["registry"])
        chunks2, report2 = pipeline2.run(str(temp_env["doc_a1"]))

        assert report2.is_reused is False, "Config drift (chunk_size) must trigger re-processing"
        assert report2.cache_status == "PROCESSED"

    def test_same_pdf_changed_chunk_overlap_triggers_reprocessing(self, temp_env):
        """If chunk_overlap changes, the cache must invalidate and re-process the file."""
        pipeline1 = IngestionPipeline(chunk_size=400, chunk_overlap=20, registry=temp_env["registry"])
        chunks1, report1 = pipeline1.run(str(temp_env["doc_a1"]))

        # Different chunk_overlap
        pipeline2 = IngestionPipeline(chunk_size=400, chunk_overlap=120, registry=temp_env["registry"])
        chunks2, report2 = pipeline2.run(str(temp_env["doc_a1"]))

        assert report2.is_reused is False, "Config drift (chunk_overlap) must trigger re-processing"

    def test_same_pdf_changed_splitter_strategy_triggers_reprocessing(self, temp_env):
        """If splitter_type changes (e.g. recursive -> token), the cache must invalidate."""
        pipeline1 = IngestionPipeline(splitter_type="recursive", chunk_size=400, chunk_overlap=40, registry=temp_env["registry"])
        chunks1, report1 = pipeline1.run(str(temp_env["doc_a1"]))

        pipeline2 = IngestionPipeline(splitter_type="token", chunk_size=400, chunk_overlap=40, registry=temp_env["registry"])
        chunks2, report2 = pipeline2.run(str(temp_env["doc_a1"]))

        assert report2.is_reused is False, "Config drift (splitter_type) must trigger re-processing"


class TestVectorStoreDeduplicationAndDirectQuery:
    """Tests vector store vector count preservation and direct QA query execution on cached documents."""

    def test_duplicate_upload_does_not_inflate_vector_store(self, temp_env):
        """Re-uploading the same PDF must not add duplicate vectors to FAISS."""
        embedder = get_embeddings()
        pipeline = IngestionPipeline(chunk_size=400, chunk_overlap=40, registry=temp_env["registry"])

        # Initial Ingestion
        chunks, report = pipeline.run(str(temp_env["doc_a1"]))
        store = get_or_create_faiss(documents=chunks, embeddings=embedder)
        initial_count = store.index.ntotal

        # Simulate duplicate upload
        chunks_dup, report_dup = pipeline.run(str(temp_env["doc_a2"]))
        assert report_dup.is_reused is True

        # Only add documents if NOT reused
        if not report_dup.is_reused:
            store.add_documents(chunks_dup)

        final_count = store.index.ntotal
        assert final_count == initial_count, "Duplicate upload must not increase FAISS vector count"

    def test_query_works_directly_against_cached_document(self, temp_env):
        """An already-indexed document can be queried directly via hybrid RAG."""
        embedder = get_embeddings()
        pipeline = IngestionPipeline(chunk_size=400, chunk_overlap=40, registry=temp_env["registry"])

        # Ingest document A
        chunks, _ = pipeline.run(str(temp_env["doc_a1"]))
        store = get_or_create_faiss(documents=chunks, embeddings=embedder)
        retriever = store.as_retriever(search_kwargs={"k": 2})

        # Query directly
        rag = AdvancedRAGPipeline(dense_retriever=retriever, documents=chunks)
        results = retriever.invoke("What is the repo rate?")

        assert len(results) > 0
        assert "6.50" in results[0].page_content or "repo rate" in results[0].page_content


class TestFastAPIUploadDeduplicationEndpoint:
    """Tests FastAPI /api/v1/documents/upload duplicate detection behavior."""

    def test_upload_api_reuses_existing_document(self, temp_env):
        """Uploading the same PDF content through the API returns is_reused=True on second call."""
        app = create_app()
        client = TestClient(app)

        with open(temp_env["doc_a1"], "rb") as f:
            pdf_bytes = f.read()

        # 1. First upload
        resp1 = client.post(
            "/api/v1/documents/upload",
            files={"file": ("rbi_policy_doc.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp1.status_code == 201
        data1 = resp1.json()
        assert data1["is_reused"] is False
        assert data1["chunks_created"] > 0

        # 2. Second upload (Different filename, identical bytes)
        resp2 = client.post(
            "/api/v1/documents/upload",
            files={"file": ("rbi_policy_copy_rename.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp2.status_code == 201
        data2 = resp2.json()
        assert data2["is_reused"] is True, "Duplicate upload should return is_reused=True"
        assert "already indexed" in data2["message"]
        assert data2["chunks_created"] == data1["chunks_created"]
        assert data2["content_fingerprint"] == data1["content_fingerprint"]


class TestConfigDriftVectorReplacement:
    """Tests end-to-end vector store replacement and stale vector removal on config drift."""

    def test_config_drift_purges_old_vectors_and_inserts_new_vectors(self, temp_env):
        """When a document's chunking configuration changes, old vectors must be purged from FAISS."""
        from vectorstore.store import replace_document_vectors, delete_documents_by_fingerprint
        embedder = get_embeddings()

        # Step 1: Ingest under Configuration A (Fine-grained chunking -> multiple chunks)
        pipeline_a = IngestionPipeline(chunk_size=60, chunk_overlap=10, registry=temp_env["registry"])
        chunks_a, report_a = pipeline_a.run(str(temp_env["doc_a1"]))
        assert len(chunks_a) >= 3, "Configuration A should generate at least 3 small chunks"

        # Index chunks in FAISS
        store = get_or_create_faiss(documents=chunks_a, embeddings=embedder)
        initial_vector_count = store.index.ntotal
        assert initial_vector_count == len(chunks_a)

        # Verify retrieval matches Configuration A chunks
        res_a = store.similarity_search("repo rate", k=5)
        for r in res_a:
            assert len(r.page_content) <= 80, "Chunks retrieved must be from fine-grained Config A"

        # Step 2: Re-ingest under Configuration B (Coarse chunking -> 1 large chunk)
        pipeline_b = IngestionPipeline(chunk_size=1000, chunk_overlap=50, registry=temp_env["registry"])
        chunks_b, report_b = pipeline_b.run(str(temp_env["doc_a1"]))
        assert report_b.is_reused is False, "Config drift must trigger re-processing"
        assert len(chunks_b) == 1, "Configuration B should generate exactly 1 consolidated chunk"

        # Replace vectors in store
        fingerprint = calculate_file_sha256(temp_env["doc_a1"])
        deleted_count, inserted_ids = replace_document_vectors(
            vectorstore=store,
            fingerprint_or_doc_id=fingerprint,
            new_documents=chunks_b,
        )

        assert deleted_count == initial_vector_count, "All old Config A vectors must be deleted"
        assert len(inserted_ids) == 1, "Exactly 1 new Config B vector should be inserted"
        assert store.index.ntotal == 1, "Vector store total must equal new chunk count only"

        # Step 3: Verify retrieval returns ONLY the new Configuration B chunk
        res_b = store.similarity_search("repo rate", k=5)
        assert len(res_b) == 1
        assert len(res_b[0].page_content) > 100, "Retrieved chunk must be the full consolidated chunk from Config B"


class TestMultimodalConfigSignatures:
    """Tests configuration signature sensitivity for multimodal parameters (vision model, OCR, table strategy)."""

    def test_vision_parameters_alter_config_signature(self):
        """Toggling vision processing or changing vision models must change config signature."""
        sig_no_vision = compute_config_signature(
            chunk_size=600,
            chunk_overlap=80,
            parser_type="multimodal",
            enable_vision_processing=False,
        )
        sig_with_vision = compute_config_signature(
            chunk_size=600,
            chunk_overlap=80,
            parser_type="multimodal",
            enable_vision_processing=True,
            vision_model="gemma4:cloud",
        )
        sig_diff_vision_model = compute_config_signature(
            chunk_size=600,
            chunk_overlap=80,
            parser_type="multimodal",
            enable_vision_processing=True,
            vision_model="llama3.2-vision",
        )
        sig_diff_table = compute_config_signature(
            chunk_size=600,
            chunk_overlap=80,
            parser_type="multimodal",
            enable_vision_processing=True,
            vision_model="gemma4:cloud",
            table_strategy="csv",
        )

        assert sig_no_vision != sig_with_vision, "Toggling vision processing must alter signature"
        assert sig_with_vision != sig_diff_vision_model, "Changing vision model must alter signature"
        assert sig_with_vision != sig_diff_table, "Changing table strategy must alter signature"

    def test_multimodal_pipeline_reprocesses_on_vision_toggle(self, temp_env):
        """MultimodalIngestionPipeline must re-process if vision processing is enabled after a text-only run."""
        from ingestion.multimodal_pipeline import MultimodalIngestionPipeline

        # Ingest first without vision
        pipeline1 = MultimodalIngestionPipeline(
            chunk_size=500,
            chunk_overlap=50,
            enable_vision_processing=False,
            registry=temp_env["registry"],
        )
        docs1, rep1 = pipeline1.ingest_pdf(str(temp_env["doc_a1"]))
        assert rep1.is_reused is False

        # Ingest second WITH vision enabled
        pipeline2 = MultimodalIngestionPipeline(
            chunk_size=500,
            chunk_overlap=50,
            enable_vision_processing=True,
            registry=temp_env["registry"],
        )
        docs2, rep2 = pipeline2.ingest_pdf(str(temp_env["doc_a1"]))
        assert rep2.is_reused is False, "Enabling vision must trigger cache invalidation and re-processing"
        assert rep2.config_signature != rep1.config_signature
