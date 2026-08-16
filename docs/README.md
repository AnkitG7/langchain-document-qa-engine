# DocMind Documentation Index

> Comprehensive technical documentation for the DocMind Document Q&A Engine.
> Every major concept, architecture decision, and implementation detail is documented here.

---

## How to Read This Documentation

Start with [Architecture Overview](architecture.md) for a high-level understanding of the system, then dive into specific areas of interest. The documentation is organized by concept, not by source file — each document explains the *why*, *what*, *how*, trade-offs, and limitations.

---

## Documentation Map

### 📐 System Architecture
- **[Architecture Overview](architecture.md)** — Top-level system design, 10-phase progression, key design decisions, directory structure, and the foundational design rule

### 📄 Document Processing
- **[Document Fingerprinting & Ingestion Cache](document-fingerprinting.md)** — Binary SHA-256 identity, why filenames fail, configuration signatures, registry lifecycle (CACHE HIT / MISS / DRIFT), alias tracking, and persistence
- **[Upload Lifecycle](upload-lifecycle.md)** — End-to-end upload flow from HTTP POST through fingerprinting, caching, parsing, chunking, embedding, and vector indexing
- **[Multimodal RAG](multimodal-rag.md)** — Text, table, image, and chart extraction; Vision LLM integration; why image extraction ≠ image understanding; element-aware citations

### 🗄️ Vector Storage & Consistency
- **[Vector Store Consistency](vector-store-consistency.md)** — Duplicate/stale/contaminated vector problems; FAISS, Chroma, PGVector implementations; Build-New-Then-Swap atomic replacement; `exclude_ids` protection; backend atomicity comparison

### 🔍 Retrieval & Search
- **[Hybrid Retrieval](hybrid-retrieval.md)** — Dense retrieval, BM25, Reciprocal Rank Fusion (RRF), query transformations (HyDE, Multi-Query, Step-Back), reranking, compression, 8 pipeline strategies, Q23/Q24 diagnostic case study
- **[Query Lifecycle](query-lifecycle.md)** — End-to-end query flow from user question through session history, contextualization, retrieval, reranking, LLM generation, and SSE streaming

### 📊 Evaluation & Benchmarking
- **[RAG Evaluation](rag-evaluation.md)** — RAG Triad metrics (Faithfulness, Relevance, Precision, Recall), 30-question Transformer benchmark, ablation study, what 28/30 proves and does NOT prove

### 🧪 Testing
- **[Testing](testing.md)** — Test philosophy, categories (unit/integration/regression/failure-recovery), 124-test suite breakdown, what passing tests prove and do NOT prove

### 🏭 Production & Operations
- **[Production Considerations](production-considerations.md)** — Current production features, JSON registry limitations, concurrency, security, observability gaps, crash recovery, garbage collection, and remaining gaps for true production readiness

---

## Supporting Documentation

| Document | Location | Purpose |
| :--- | :--- | :--- |
| **Project README** | [README.md](../README.md) | Quick start, phase module reference, environment setup |
| **Engineering Learnings** | [LEARNINGS.md](../LEARNINGS.md) | Detailed case studies, empirical benchmarks, architectural lessons |
| **Environment Template** | [.env.example](../.env.example) | Configuration reference |

---

## Cross-Reference: Source Code → Documentation

| Source Directory | Primary Documentation |
| :--- | :--- |
| `ingestion/` | [Document Fingerprinting](document-fingerprinting.md), [Upload Lifecycle](upload-lifecycle.md), [Multimodal RAG](multimodal-rag.md) |
| `vectorstore/` | [Vector Store Consistency](vector-store-consistency.md) |
| `rag_advanced/` | [Hybrid Retrieval](hybrid-retrieval.md) |
| `evaluation/` | [RAG Evaluation](rag-evaluation.md) |
| `api/` | [Upload Lifecycle](upload-lifecycle.md), [Query Lifecycle](query-lifecycle.md) |
| `memory/` | [Query Lifecycle](query-lifecycle.md) |
| `production/` | [Production Considerations](production-considerations.md) |
| `observability/` | [Production Considerations](production-considerations.md) |
| `tests/` | [Testing](testing.md) |

---

## Design Principles

> **"Never hardcode the answer or benchmark behavior to pass a test."**

All ingestion, retrieval, caching, multimodal processing, and evaluation behavior is generic and driven by the actual uploaded document and configuration. See [Architecture Overview § Important Design Rule](architecture.md) for the full explanation.
