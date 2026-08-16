"""Unit and integration tests for Phase 3: Embeddings and Vector Stores.

Covers:
- Dedicated Embeddings (deterministic fake & live verification)
- Chroma Vector Store (in-memory, persistence, reload, similarity, MMR, metadata filter)
- FAISS Vector Store (in-memory, save_local, load_local, similarity, MMR)
- Full End-to-End RAG Pipeline
- Negative and edge case handling
"""

import os
import shutil
import tempfile
import pytest
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from vectorstore.embedder import (
    get_embeddings,
    get_fake_embeddings,
    verify_embeddings,
    ConsistentFakeEmbeddings,
)
from vectorstore.store import (
    create_vector_store,
    get_or_create_chroma,
    get_or_create_faiss,
    VectorStoreManager,
)
from vectorstore.retriever import (
    create_retriever,
    similarity_search_with_scores,
    mmr_search,
    threshold_search,
)
from ingestion.pipeline import IngestionPipeline
from chains.qa_chain import create_basic_qa_chain, create_structured_qa_chain


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"doc_id": "doc_1", "file_type": "markdown", "topic": "framework"},
        ),
        Document(
            page_content="Chroma and FAISS provide efficient local vector similarity search without requiring cloud servers.",
            metadata={"doc_id": "doc_2", "file_type": "text", "topic": "vectorstore"},
        ),
        Document(
            page_content="DocMind supports PDF, Markdown, CSV, and Web ingestion pipelines.",
            metadata={"doc_id": "doc_3", "file_type": "pdf", "topic": "ingestion"},
        ),
    ]


class TestDedicatedEmbeddings:
    def test_fake_embeddings_determinism(self):
        embedder = get_fake_embeddings(size=384)
        vec1 = embedder.embed_query("DocMind test query")
        vec2 = embedder.embed_query("DocMind test query")

        assert len(vec1) == 384
        assert vec1 == vec2  # Exact deterministic match

    def test_verify_embeddings(self):
        embedder = get_fake_embeddings()
        report = verify_embeddings(embedder)
        assert report["status"] == "healthy"
        assert report["dimensions"] == 384
        assert report["documents_tested"] == 2

    def test_unsupported_provider_raises_error(self):
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            get_embeddings(provider="invalid_provider_name")


class TestChromaStore:
    def test_chroma_in_memory_and_search(self, sample_documents):
        embedder = get_fake_embeddings()
        store = get_or_create_chroma(
            collection_name="test_mem_chroma",
            embeddings=embedder,
            documents=sample_documents,
        )

        results = store.similarity_search("Tell me about vector stores", k=2)
        assert len(results) > 0
        assert any("vector" in r.page_content.lower() for r in results)

    def test_chroma_persistence_and_reload(self, sample_documents):
        temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
        try:
            embedder = get_fake_embeddings()
            # 1. Create and persist
            store1 = get_or_create_chroma(
                collection_name="test_persist_chroma",
                persist_directory=temp_dir,
                embeddings=embedder,
                documents=sample_documents,
            )
            assert store1 is not None

            # 2. Reload from disk
            store2 = get_or_create_chroma(
                collection_name="test_persist_chroma",
                persist_directory=temp_dir,
                embeddings=embedder,
            )
            results = store2.similarity_search("LangChain framework", k=1)
            assert len(results) == 1
            assert "LangChain" in results[0].page_content
            assert results[0].metadata["doc_id"] == "doc_1"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_chroma_metadata_filtering(self, sample_documents):
        embedder = get_fake_embeddings()
        store = get_or_create_chroma(
            collection_name="test_filter_chroma",
            embeddings=embedder,
            documents=sample_documents,
        )

        # Filter by file_type == 'pdf'
        retriever = create_retriever(
            vectorstore=store,
            search_type="similarity",
            k=3,
            filter_dict={"file_type": "pdf"},
        )
        results = retriever.invoke("What does DocMind support?")
        assert len(results) == 1
        assert results[0].metadata["file_type"] == "pdf"
        assert "PDF, Markdown" in results[0].page_content


class TestFAISSStore:
    def test_faiss_in_memory_and_search(self, sample_documents):
        embedder = get_fake_embeddings()
        store = get_or_create_faiss(
            documents=sample_documents,
            embeddings=embedder,
        )

        results = store.similarity_search("Chroma and FAISS search", k=2)
        assert len(results) == 2
        assert any("FAISS" in r.page_content for r in results)

    def test_faiss_save_and_load_local(self, sample_documents):
        temp_dir = tempfile.mkdtemp(prefix="faiss_test_")
        try:
            embedder = get_fake_embeddings()
            # Save index
            store1 = get_or_create_faiss(
                documents=sample_documents,
                embeddings=embedder,
                index_path=temp_dir,
            )
            assert Path(temp_dir).exists()

            # Reload index
            store2 = get_or_create_faiss(
                embeddings=embedder,
                index_path=temp_dir,
            )
            results = store2.similarity_search("DocMind ingestion", k=1)
            assert len(results) == 1
            assert "DocMind" in results[0].page_content
            assert results[0].metadata["doc_id"] == "doc_3"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_faiss_mmr_search(self, sample_documents):
        embedder = get_fake_embeddings()
        store = get_or_create_faiss(
            documents=sample_documents,
            embeddings=embedder,
        )

        results = mmr_search(vectorstore=store, query="Framework and vector search", k=2, fetch_k=3)
        assert len(results) == 2


class TestEndToEndRAG:
    def test_full_e2e_rag_workflow(self):
        """End-to-end test: Sample doc -> Pipeline -> VectorStore -> Retriever -> QA Chain -> Response."""
        data_dir = Path(__file__).parent.parent / "data"
        doc_path = str(data_dir / "sample_doc.txt")

        # 1. Ingest
        pipeline = IngestionPipeline(chunk_size=200, chunk_overlap=30)
        chunks, report = pipeline.run(doc_path)
        assert len(chunks) > 0

        # 2. Vector Indexing with Chroma
        embedder = get_fake_embeddings()
        vectorstore = get_or_create_chroma(
            collection_name="test_e2e_rag",
            embeddings=embedder,
            documents=chunks,
        )

        # 3. Retrieve
        retriever = create_retriever(vectorstore=vectorstore, search_type="similarity", k=2)
        retrieved_docs = retriever.invoke("What does DocMind support?")
        assert len(retrieved_docs) > 0

        # 4. Generate Answer via LCEL Chain
        context_str = "\n\n".join(d.page_content for d in retrieved_docs)
        mock_llm = FakeListChatModel(responses=[
            "DocMind supports multi-format document ingestion (PDF, CSV, TXT, MD) and hybrid retrieval."
        ])
        qa_chain = create_basic_qa_chain(llm=mock_llm)

        answer = qa_chain.invoke({
            "context": context_str,
            "question": "What does DocMind support?",
        })
        assert "multi-format" in answer


class TestNegativeCases:
    def test_empty_query_returns_empty_results(self, sample_documents):
        embedder = get_fake_embeddings()
        store = get_or_create_faiss(documents=sample_documents, embeddings=embedder)

        res = similarity_search_with_scores(store, query="")
        assert res == []

        mmr_res = mmr_search(store, query="   ")
        assert mmr_res == []

    def test_vectorstore_manager_empty_docs(self):
        embedder = get_fake_embeddings()
        mgr = VectorStoreManager(store_type="faiss", embeddings=embedder)
        ids = mgr.add_documents([])
        assert ids == []

    def test_nonexistent_faiss_load_fails(self):
        embedder = get_fake_embeddings()
        fake_path = str(Path(tempfile.gettempdir()) / "definitely_nonexistent_faiss_path_987654")
        if Path(fake_path).exists():
            shutil.rmtree(fake_path)

        with pytest.raises(FileNotFoundError):
            get_or_create_faiss(
                embeddings=embedder,
                index_path=fake_path,
            )
