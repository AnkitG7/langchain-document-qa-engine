# Testing

This document details the complete test suite for the DocMind project, which currently consists of **124 passing tests**.

## 1. Test Philosophy

- **Offline by Default:** All tests run offline without external API dependencies. We use Fake LLMs and Fake Embeddings.
- **Behavior over Implementation:** Tests verify the behavior and outcomes rather than the specific implementation details.
- **No Hardcoding:** No tests are hardcoded to pass specific benchmark questions. 
- **Genericity:** Tests are generic. If the input document changes, the behavior should adapt accordingly without requiring test rewrites.

## 2. Test Categories

### Unit Tests
- Isolated testing of components such as embeddings, splitters, cleaners, tools, and metrics.
- **Examples:** `test_fake_embeddings_determinism`, `test_clean_text_normalization`, `test_basic_arithmetic`.

### Integration Tests
- Multi-component pipeline testing including E2E RAG workflows, conversational RAG, and agent tool-calling.
- **Examples:** `test_full_e2e_rag_workflow`, `test_multi_turn_conversational_rag`, `test_pipeline_strategies`.

### Regression Tests
- Ensures that new phases do not break older implementations.
- The full 124-test suite runs across all 10 phases, plus multimodal and deduplication capabilities.

### Failure Recovery Tests
- `test_failed_reindex_preserves_old_vectors_on_embedding_exception`: Simulates an embedding API failure during re-indexing and verifies that the old vectors remain intact.
- `test_chroma_build_new_then_swap_failure_recovery`: Validates the same failure safety mechanisms for the Chroma backend.

### Deduplication Tests (17 tests in `test_document_deduplication.py`)

1. `test_content_identity_independent_of_filename` — Same bytes, different names = same hash.
2. `test_config_signature_sensitivity` — Changing any config param changes signature.
3. `test_registry_persistence_survives_restart` — Disk persistence across restarts.
4. `test_same_pdf_same_filename_reuses_index` — Cache HIT on re-upload.
5. `test_same_pdf_different_filename_reuses_index` — Alias filename handling.
6. `test_different_pdf_processes_normally` — Different content = different fingerprint.
7. `test_same_pdf_changed_chunk_size_triggers_reprocessing` — Config drift detection.
8. `test_same_pdf_changed_chunk_overlap_triggers_reprocessing` — Overlap drift.
9. `test_same_pdf_changed_splitter_strategy_triggers_reprocessing` — Splitter drift.
10. `test_duplicate_upload_does_not_inflate_vector_store` — Zero index bloat.
11. `test_query_works_directly_against_cached_document` — Cached docs are queryable.
12. `test_upload_api_reuses_existing_document` — FastAPI duplicate detection.
13. `test_config_drift_purges_old_vectors_and_inserts_new_vectors` — Physical vector replacement.
14. `test_failed_reindex_preserves_old_vectors_on_embedding_exception` — Failure safety.
15. `test_chroma_build_new_then_swap_failure_recovery` — Chroma failure safety.
16. `test_vision_parameters_alter_config_signature` — Multimodal config sensitivity.
17. `test_multimodal_pipeline_reprocesses_on_vision_toggle` — Vision toggle drift.

### Multimodal Tests (8 tests in `test_multimodal_rag.py`)

- Covers table markdown formatting, image bytes handling, Vision model provider, multimodal ingestion pipeline, and unified hybrid retrieval across element types.

## 3. Test File Summary Table

| File | Tests | Focus |
|---|---|---|
| [`test_phase1.py`](file:///c:/FS/langchain_document_qa/tests/test_phase1.py) | 9 tests | LLM providers, QA/summary/compare chains |
| [`test_phase2.py`](file:///c:/FS/langchain_document_qa/tests/test_phase2.py) | 12 tests | Loaders, cleaners, splitters, pipeline |
| [`test_phase3.py`](file:///c:/FS/langchain_document_qa/tests/test_phase3.py) | 13 tests | Embeddings, Chroma/FAISS, E2E RAG |
| [`test_phase4.py`](file:///c:/FS/langchain_document_qa/tests/test_phase4.py) | 8 tests | Memory, session history, conversational RAG |
| [`test_phase5.py`](file:///c:/FS/langchain_document_qa/tests/test_phase5.py) | 13 tests | Tools, calculator security, agent |
| [`test_phase6.py`](file:///c:/FS/langchain_document_qa/tests/test_phase6.py) | 10 tests | FastAPI endpoints, SSE streaming |
| [`test_phase7.py`](file:///c:/FS/langchain_document_qa/tests/test_phase7.py) | 9 tests | HyDE, BM25, RRF, reranker, compression |
| [`test_phase8.py`](file:///c:/FS/langchain_document_qa/tests/test_phase8.py) | 9 tests | RAG triad metrics, evaluator, benchmark |
| [`test_phase9.py`](file:///c:/FS/langchain_document_qa/tests/test_phase9.py) | 7 tests | Telemetry, tracing, logging |
| [`test_phase10.py`](file:///c:/FS/langchain_document_qa/tests/test_phase10.py) | 9 tests | Production caching, PGVector fallback, K8s probes |
| [`test_multimodal_rag.py`](file:///c:/FS/langchain_document_qa/tests/test_multimodal_rag.py) | 8 tests | Multimodal extraction and retrieval |
| [`test_document_deduplication.py`](file:///c:/FS/langchain_document_qa/tests/test_document_deduplication.py) | 17 tests | Fingerprinting, cache, vector replacement |

## 4. What 124/124 Passing Means

**What it proves:**
- All components function correctly in isolation.
- Integration between components works as designed.
- Regression protection: changes in one phase don't break other phases.
- Failure recovery mechanisms work as designed.
- Cache and deduplication logic is correct.

**What it does NOT prove:**
- Performance under load (no load tests).
- Correctness with real LLMs (tests use fake LLMs).
- Security hardening (no penetration tests).
- Multi-user concurrency (tests are single-threaded).
- Production deployment reliability (no infrastructure tests).
- Actual RAG answer quality (that requires the benchmark, not unit tests).

> [!NOTE]
> Static analysis might find 115 tests from method signatures. The actual test run reports 124 tests. The difference likely comes from parametrized tests or test fixtures. The actual pytest output is 124 passed.
