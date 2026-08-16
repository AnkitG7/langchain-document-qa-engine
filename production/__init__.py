"""Production Architecture & Engineering Module for DocMind.

Demonstrates:
- High-Performance Caching Layer (In-Memory TTL & Production Redis)
- Persistent Vector Storage with PGVector and Local Fallbacks
- Enterprise Health Probes (Liveness, Readiness, Metrics)
- Production Multi-Worker Application Factory
- Containerization (Dockerfile & Docker Compose)
"""

from .cache import (
    CacheBackend,
    InMemoryTTLCache,
    RedisCacheBackend,
    CachedRAGService,
    get_cache_backend,
)
from .pgvector_store import (
    PGVectorStoreManager,
    ProductionVectorStore,
)
from .probes import (
    LivenessResponse,
    ReadinessResponse,
    probes_router,
)
from .app import create_production_app

__all__ = [
    # Caching
    "CacheBackend",
    "InMemoryTTLCache",
    "RedisCacheBackend",
    "CachedRAGService",
    "get_cache_backend",
    # Vector Storage
    "PGVectorStoreManager",
    "ProductionVectorStore",
    # Probes & App
    "LivenessResponse",
    "ReadinessResponse",
    "probes_router",
    "create_production_app",
]
