"""Unit and integration tests for Phase 10: Production Architecture & Engineering.

Covers:
- Production Caching Subsystem (InMemoryTTLCache, RedisCacheBackend fallback, CachedRAGService)
- Cache Hit vs Miss and TTL expiration
- Production Vector Store with PGVector fallback to FAISS
- Kubernetes Liveness, Readiness, and Metrics Probes
- Request Correlation Middleware (X-Request-ID, X-Trace-ID, X-Response-Time-Ms)
- Production App Factory
"""

import time
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from production.cache import (
    InMemoryTTLCache,
    RedisCacheBackend,
    CachedRAGService,
    get_cache_backend,
)
from production.pgvector_store import (
    PGVectorStoreManager,
    ProductionVectorStore,
)
from production.app import create_production_app
from vectorstore.embedder import get_fake_embeddings


@pytest.fixture
def prod_client():
    app = create_production_app()
    return TestClient(app)


class TestProductionCaching:
    def test_in_memory_ttl_cache_operations(self):
        cache = InMemoryTTLCache(default_ttl=2)

        # 1. Set and Get
        cache.set("query_01", {"answer": "DocMind uses LangChain.", "citations": []})
        cached = cache.get("query_01")
        assert cached is not None
        assert cached["answer"] == "DocMind uses LangChain."

        # 2. Miss
        assert cache.get("nonexistent_key") is None

        # 3. Stats
        stats = cache.stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["hit_rate_pct"] > 0

        # 4. TTL Expiration
        time.sleep(2.1)
        assert cache.get("query_01") is None

    def test_redis_backend_graceful_fallback(self):
        # Redis offline on random port -> should fall back gracefully to InMemoryTTLCache
        redis_cache = RedisCacheBackend(redis_url="redis://localhost:9999/0", default_ttl=5)
        redis_cache.set("fallback_test", {"value": "cached_via_fallback"})

        val = redis_cache.get("fallback_test")
        assert val is not None
        assert val["value"] == "cached_via_fallback"

        stats = redis_cache.stats()
        assert "fallback_to_in_memory" in stats["backend"]

    def test_cached_rag_service_wrapper(self):
        call_count = 0

        def dummy_rag(question: str, strategy: str = "default"):
            nonlocal call_count
            call_count += 1
            return {"answer": f"Answer for: {question}", "strategy": strategy}

        cache = InMemoryTTLCache(default_ttl=10)
        cached_service = CachedRAGService(rag_func=dummy_rag, cache=cache)

        # 1. First Call: Miss -> executes dummy_rag
        res1 = cached_service.query("What is RAG?")
        assert res1["cache_hit"] is False
        assert call_count == 1

        # 2. Second Call: Hit -> returns cached result without calling dummy_rag
        res2 = cached_service.query("What is RAG?")
        assert res2["cache_hit"] is True
        assert res2["cached"] is True
        assert call_count == 1


class TestProductionVectorStore:
    def test_production_vectorstore_faiss_fallback(self):
        embedder = get_fake_embeddings()
        sample_docs = [
            Document(page_content="Production vector store fallback test.", metadata={"source": "prod.txt"})
        ]

        # Request PGVector when Postgres is not running -> Falls back to FAISS seamlessly
        prod_store = ProductionVectorStore(
            documents=sample_docs,
            embeddings=embedder,
            use_pgvector=True,
        )

        status = prod_store.get_status()
        assert "backend" in status

        retriever = prod_store.as_retriever(search_kwargs={"k": 1})
        docs = retriever.invoke("fallback test")
        assert len(docs) == 1
        assert "fallback test" in docs[0].page_content


class TestProductionProbesAndMiddleware:
    def test_production_root(self, prod_client):
        res = prod_client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["version"] == "1.0.0"
        assert data["environment"] == "production"

    def test_liveness_probe(self, prod_client):
        res = prod_client.get("/api/v1/health/live")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "alive"
        assert "uptime_seconds" in data

    def test_readiness_probe(self, prod_client):
        res = prod_client.get("/api/v1/health/ready")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ready"
        assert data["vector_store"] == "healthy"
        assert data["cache"] == "healthy"

    def test_system_metrics_endpoint(self, prod_client):
        res = prod_client.get("/api/v1/metrics")
        assert res.status_code == 200
        data = res.json()
        assert "uptime_seconds" in data
        assert "cache_metrics" in data

    def test_request_correlation_headers(self, prod_client):
        custom_headers = {"X-Request-ID": "test-custom-req-id"}
        res = prod_client.get("/api/v1/health/live", headers=custom_headers)
        assert res.status_code == 200
        assert res.headers.get("X-Request-ID") == "test-custom-req-id"
        assert "X-Trace-ID" in res.headers
        assert "X-Response-Time-Ms" in res.headers
