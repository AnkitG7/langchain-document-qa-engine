# Production Considerations and Limitations

This document outlines the current state of the DocMind production system. It provides an honest evaluation of implemented features, architectural limitations, and necessary future improvements required before deploying to a true high-availability, multi-tenant environment.

## 1. Current Production Features
The following features are fully implemented and available in the `production/` subsystem:
- **Content-based document fingerprinting and ingestion cache:** Uses `calculate_content_hash` to detect reused content.
- **Build-New-Then-Swap failure-safe vector replacement:** Ensures application-level atomicity during index builds.
- **In-memory TTL cache and Redis cache backend with graceful fallback:** Implemented in `production/cache.py` (`InMemoryTTLCache` and `RedisCacheBackend`).
- **PGVector production storage with automatic FAISS fallback:** Integrated via `production/pgvector_store.py`.
- **Kubernetes liveness and readiness health probes:** Exposed via `/api/v1/health/live` and `/api/v1/health/ready` endpoints in `production/probes.py`.
- **Request correlation headers (`X-Request-ID`, `X-Trace-ID`, `X-Response-Time-Ms`):** Injected via `RequestCorrelationMiddleware` in `production/app.py`.
- **Telemetry callbacks with token/cost tracking and p50/p95 latency:** Embedded into the RAG pipeline telemetry.
- **Structured JSON trace logging:** Implemented as part of the core logging system.
- **Multi-worker ASGI deployment via Docker:** Support for Uvicorn multi-worker configurations.
- **CORS middleware:** Enabled in `production/app.py` allowing all origins.
- **Input validation and error handling:** Handled via FastAPI and Pydantic.

## 2. JSON Registry Limitations
- The document registry is stored as a JSON file on local disk (`data/document_registry.json`).
- **No ACID guarantees** exist for concurrent writes from multiple workers or processes.
- `threading.Lock` protects in-process access only, meaning it provides no safety across different processes.
- **Risk:** Multi-worker deployments (e.g., Uvicorn with multiple workers) may cause registry corruption due to race conditions.
- **Mitigation:** Move to a database-backed registry (e.g., PostgreSQL, SQLite, or Redis) before scaling horizontally.

## 3. Multi-Worker / Multi-Process Concurrency
- The local FAISS index is strictly in-memory and single-process.
- `DocumentRegistry` utilizes `threading.Lock`, which only synchronizes threads within the same process.
- There is no distributed locking mechanism implemented.
- **Recommendation:** Use PGVector for all multi-worker deployments, or introduce Redis-based distributed locks if a local vector store is mandated.

## 4. Object Storage
- The system currently stores uploaded files directly on the local disk (in the `data/` directory).
- There is no native integration with S3, GCS, or Azure Blob Storage.
- Docker volume mounts are required to ensure data persistence across container restarts.

## 5. Database-Backed Document Registry
- **Current state:** The system uses a flat JSON file (`data/document_registry.json`).
- **Production requirement:** The registry must migrate to a relational database like PostgreSQL or a robust key-value store like Redis to provide ACID guarantees and safe concurrent access.

## 6. Idempotency
- The document upload endpoint is idempotent for the same content and identical configuration (it correctly returns `is_reused=True`).
- **Limitation:** It is not idempotent across concurrent, simultaneous uploads of the same file. A race condition is possible where both requests process the file independently.

## 7. Crash Recovery
- **FAISS:** The in-memory index is permanently lost upon crash unless `save_local` was explicitly called. Save is invoked after `add_documents`.
- **Chroma:** Persistent collections can survive crashes, but in-memory collections are lost.
- **PGVector:** Full ACID crash recovery is provided via the PostgreSQL WAL (Write-Ahead Log).
- **Registry:** Persisted to disk on every write.
- **Risk:** If a system crash occurs strictly between a vector insertion and the subsequent registry save, orphaned vectors may exist in the store.

## 8. Transactional Guarantees
- **FAISS/Chroma:** The Build-New-Then-Swap strategy provides application-level atomicity, not true database-level atomicity.
- **PGVector:** Benefits from true ACID transaction atomicity provided by PostgreSQL.
- **Cross-system limitation:** There are no distributed transactions coordinating states between the vector store and the JSON document registry.

## 9. Embedding Model/Version Changes
- Dimension mismatches between old and new embedding models cause critical application crashes or return garbage similarity results.
- The configuration signature mechanism detects this and successfully triggers re-indexing.
- There is no automatic migration of existing vector stores when the global embedding model changes.
- **Recommendation:** Explicitly version embedding models in the configuration and aggressively re-index affected documents upon any model change.

## 10. Prompt/Version Management
- `prompt_version` is embedded as part of the configuration signature.
- Changing vision LLM prompts automatically triggers re-processing of visual elements.
- There is no formal prompt versioning system or A/B testing framework in place.

## 11. Observability Gaps
- `TraceManager` stores performance metrics entirely in-memory.
- There is no standard Prometheus exporter (the `/api/v1/metrics` endpoint returns custom JSON, not Prometheus exposition format).
- No native integration exists with Datadog, New Relic, or Grafana.
- LangSmith integration is available but requires external configuration and an API key.

## 12. Rate Limiting
- No request-level rate limiting is enforced on the API endpoints.
- No embedding API rate limiting, backoff, or retry logic is implemented.
- While local models (e.g., Ollama) do not natively rate limit, utilizing cloud embedding APIs under high load without rate limiting will rapidly exhaust quotas.

## 13. Garbage Collection
- There is no automatic cleanup of orphaned vectors (e.g., resulting from failed uploads or deleted documents).
- There is no TTL (Time-To-Live) on vector store entries.
- Old registry entries are not automatically purged.

## 14. Security
- There is no authentication or authorization layer on any API endpoints.
- No file content scanning is performed; consequently, uploaded PDFs could contain malicious payloads.
- The CORS middleware is extremely permissive, allowing all origins (`allow_origins=["*"]`).
- No input sanitization is performed beyond basic file extension validation.

## 15. Important Design Rule

> [!IMPORTANT]
> **Never hardcode the answer or benchmark behavior to pass a test.**

- All behavior must be entirely generic and driven directly by the actual document and configuration.
- If the PDF changes, the indexed representation and resulting answer must change accordingly.
- If the chunking strategy changes, the system must detect the configuration drift and fully re-index.
- If the same binary document is uploaded under a completely different filename, the system must detect the hash match and reuse the existing representation.
- If the embedding, vision, table, or parser configuration changes, the system must invalidate the cache and rebuild the index.
