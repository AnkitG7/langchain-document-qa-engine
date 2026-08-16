"""Unit and integration tests for Phase 7: Advanced RAG Architecture.

Covers:
- Query Transformations: HyDE, Multi-Query Expansion, Step-Back Prompting
- Sparse Lexical Retrieval: BM25 keyword matching
- Hybrid Retrieval: Reciprocal Rank Fusion (RRF)
- Cross-Encoder / LLM Reranking
- Contextual Compression
- End-to-End Advanced RAG Pipeline with multiple strategies
"""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_advanced.query_transform import (
    HyDETransformer,
    MultiQueryTransformer,
    StepBackTransformer,
)
from rag_advanced.sparse import BM25Index, create_bm25_retriever
from rag_advanced.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag_advanced.reranker import LLMReranker
from rag_advanced.compression import ContextualCompressor
from rag_advanced.pipeline import AdvancedRAGPipeline
from vectorstore.embedder import get_fake_embeddings
from vectorstore.store import get_or_create_faiss


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="DocMind architecture features modular Ingestion, VectorStore, and Agent layers.",
            metadata={"source": "architecture.md", "file_type": "md", "id": "doc1"},
        ),
        Document(
            page_content="System Error Code ERR_AUTH_99 indicates expired JWT token during API gateway routing.",
            metadata={"source": "troubleshooting.txt", "file_type": "txt", "id": "doc2"},
        ),
        Document(
            page_content="Project budget for FY2026 allocates $85,000 for evaluation and benchmarks.",
            metadata={"source": "finance.csv", "file_type": "csv", "id": "doc3"},
        ),
    ]


@pytest.fixture
def test_dense_retriever(sample_documents):
    embedder = get_fake_embeddings()
    store = get_or_create_faiss(documents=sample_documents, embeddings=embedder)
    return store.as_retriever(search_kwargs={"k": 2})


class TestQueryTransformations:
    def test_hyde_transformer(self, test_dense_retriever):
        mock_llm = FakeListChatModel(responses=[
            "DocMind is an intelligent document analysis engine providing multi-format parsing."
        ])
        hyde = HyDETransformer(llm=mock_llm)
        hypo_doc = hyde.generate_hypothetical_document("What is DocMind?")
        assert "intelligent document analysis engine" in hypo_doc

        docs = hyde.retrieve("What is DocMind?", test_dense_retriever)
        assert len(docs) > 0

    def test_multi_query_transformer(self, test_dense_retriever):
        mock_llm = FakeListChatModel(responses=[
            "How does DocMind architecture work?\nWhat are the internal layers of DocMind?\nExplain DocMind system design."
        ])
        mq = MultiQueryTransformer(llm=mock_llm, num_queries=3)
        queries = mq.generate_queries("DocMind architecture")
        assert len(queries) >= 3

        docs = mq.retrieve("DocMind architecture", test_dense_retriever)
        assert len(docs) > 0

    def test_step_back_transformer(self, test_dense_retriever):
        mock_llm = FakeListChatModel(responses=[
            "What is the system design and layer structure of document processing engines?"
        ])
        step_back = StepBackTransformer(llm=mock_llm)
        res = step_back.retrieve("What is the chunk overlap in DocMind?", test_dense_retriever)
        assert "specific_docs" in res
        assert "step_back_docs" in res
        assert "step_back_query" in res


class TestSparseAndHybridRetrieval:
    def test_bm25_exact_keyword_matching(self, sample_documents):
        bm25 = BM25Index(documents=sample_documents, k=2)
        # Search for exact error code that embeddings might diffuse
        results = bm25.retrieve("ERR_AUTH_99")
        assert len(results) > 0
        assert "ERR_AUTH_99" in results[0].page_content

    def test_reciprocal_rank_fusion(self, sample_documents):
        doc1, doc2, doc3 = sample_documents[0], sample_documents[1], sample_documents[2]

        list1 = [doc1, doc2]
        list2 = [doc2, doc3]

        fused = reciprocal_rank_fusion([list1, list2], top_n=3)
        assert len(fused) == 3
        # doc2 appears in both lists, so it should be ranked first by RRF
        assert fused[0].metadata["id"] == "doc2"

    def test_hybrid_retriever(self, test_dense_retriever, sample_documents):
        sparse = create_bm25_retriever(sample_documents, k=2)
        hybrid = HybridRetriever(dense_retriever=test_dense_retriever, sparse_retriever=sparse, k=3)

        details = hybrid.retrieve_with_details("ERR_AUTH_99")
        assert details["dense_count"] > 0
        assert details["sparse_count"] > 0
        assert details["fused_count"] > 0


class TestRerankerAndCompression:
    def test_llm_reranker(self, sample_documents):
        # Mock scores: doc1 -> 9, doc2 -> 2, doc3 -> 5
        mock_llm = FakeListChatModel(responses=["9", "2", "5"])
        reranker = LLMReranker(llm=mock_llm, top_n=2)

        reranked = reranker.compress_documents("DocMind system architecture", sample_documents)
        assert len(reranked) == 2
        # doc1 scored highest (9)
        assert reranked[0].metadata["id"] == "doc1"

    def test_contextual_compressor(self, sample_documents):
        mock_llm = FakeListChatModel(responses=[
            "DocMind architecture features modular Ingestion and VectorStore layers.",
            "NO_RELEVANT_CONTENT",
        ])
        compressor = ContextualCompressor(llm=mock_llm)
        compressed = compressor.compress("What is the architecture?", sample_documents[:2])
        assert len(compressed) == 1
        assert "architecture features" in compressed[0].page_content


class TestAdvancedRAGPipeline:
    def test_pipeline_strategies(self, test_dense_retriever, sample_documents):
        mock_llm = FakeListChatModel(responses=[
            "Answer from baseline strategy.",
            # HyDE
            "Hypothetical document content.",
            "Answer from HyDE strategy.",
            # Multi-Query
            "Query 1\nQuery 2\nQuery 3",
            "Answer from Multi-Query strategy.",
            # Step-Back
            "Step back query concept.",
            "Answer from Step-Back strategy.",
            # Hybrid RRF
            "Answer from Hybrid RRF strategy.",
            # Reranked
            "9", "7", "4",
            "Answer from Reranked strategy.",
            # Full Advanced
            "MQ 1\nMQ 2\nMQ 3",
            "9", "8", "3",
            "Extracted relevant fact 1.",
            "Extracted relevant fact 2.",
            "Extracted relevant fact 3.",
            "Answer from Full Advanced strategy.",
        ])

        pipeline = AdvancedRAGPipeline(
            dense_retriever=test_dense_retriever,
            documents=sample_documents,
            llm=mock_llm,
        )

        # 1. Baseline
        res_base = pipeline.query("What is the architecture?", strategy="baseline")
        assert "Answer from baseline" in res_base["answer"]

        # 2. HyDE
        res_hyde = pipeline.query("What is the architecture?", strategy="hyde")
        assert "Answer from HyDE" in res_hyde["answer"]

        # 3. Multi-Query
        res_mq = pipeline.query("What is the architecture?", strategy="multi_query")
        assert "Answer from Multi-Query" in res_mq["answer"]

        # 4. Step-Back
        res_sb = pipeline.query("What is the architecture?", strategy="step_back")
        assert "Answer from Step-Back" in res_sb["answer"]

        # 5. Hybrid RRF
        res_hybrid = pipeline.query("What is ERR_AUTH_99?", strategy="hybrid_rrf")
        assert "Answer from Hybrid" in res_hybrid["answer"]

        # 6. Reranked
        res_rerank = pipeline.query("What is the architecture?", strategy="reranked")
        assert "Answer from Reranked" in res_rerank["answer"]

        # 7. Full Advanced
        res_full = pipeline.query("Explain DocMind system architecture", strategy="full_advanced")
        assert "Answer from Full Advanced" in res_full["answer"]
