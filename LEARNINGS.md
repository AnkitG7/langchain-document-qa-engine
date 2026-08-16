# 📘 DocMind: Engineering Learnings & Architecture Case Studies

> A practical guide and case study compendium on building, testing, and productionizing a Document Q&A and Advanced RAG Engine using modern LangChain, FastAPI, and local LLMs.

---

## 🗺️ Table of Contents
1. [The 10-Phase Architecture Progression](#1-the-10-phase-architecture-progression)
2. [Core Engineering Principles & Trade-offs](#2-core-engineering-principles--trade-offs)
3. [Case Study: The "Parent-Hash Inheritance" Retrieval Bug](#3-case-study-the-parent-hash-inheritance-retrieval-bug)
4. [RAG Triad Metrics: Why Groundedness Precedes Relevance](#4-rag-triad-metrics-why-groundedness-precedes-relevance)
5. [Production Architecture vs. Local Prototyping](#5-production-architecture-vs-local-prototyping)
6. [Checklist for Production RAG Systems](#6-checklist-for-production-rag-systems)

---

## 1. The 10-Phase Architecture Progression

This codebase was designed as a versioned learning curriculum. Each phase is preserved under a permanent Git tag:

```text
v0.1-phase-01  ──►  Phase 1:  LLM Abstraction, Fallbacks & LCEL Chains
      │
v0.2-phase-02  ──►  Phase 2:  Document Loaders, Cleaners & Splitters
      │
v0.3-phase-03  ──►  Phase 3:  Dedicated Embeddings, Chroma & FAISS Stores
      │
v0.4-phase-04  ──►  Phase 4:  Session History & Conversational RAG
      │
v0.5-phase-05  ──►  Phase 5:  Tools, Safety & Tool-Calling Agents
      │
v0.6-phase-06  ──►  Phase 6:  FastAPI Backend & Real-Time SSE Streaming
      │
v0.7-phase-07  ──►  Phase 7:  Advanced RAG (HyDE, BM25, RRF & Rerankers)
      │
v0.8-phase-08  ──►  Phase 8:  RAG Triad Evaluation & Benchmarking
      │
v0.9-phase-09  ──►  Phase 9:  Observability, Callbacks & Tracing
      │
v1.0-production ──► Phase 10: Production Architecture, Caching & Probes
```

---

## 2. Core Engineering Principles & Trade-offs

### A. Dedicated Embeddings vs. Reasoning LLMs
* **Rule**: Never use a chat LLM for embedding generation.
* **Why**: Chat models produce non-normalized, high-variance representations. Dedicated models like `nomic-embed-text` (768 dimensions) are contrastively trained for cosine similarity search, yielding higher retrieval accuracy at a fraction of the compute cost.

### B. Hybrid Search (Dense + Sparse BM25 + RRF)
* **Problem**: Pure dense vector search frequently fails on exact keywords, acronyms, product IDs, and model numbers (e.g., "Transformer (big)", "Section 104", "NIFTY24DEC").
* **Solution**: Combine Dense Semantic Embeddings (`nomic-embed-text` in FAISS/PGVector) with Sparse Lexical Retrieval (`BM25Index`) using **Reciprocal Rank Fusion (RRF)**:
  $$RRF(d) = \sum_{r \in \text{Rankings}} \frac{1}{60 + \text{rank}(d)}$$
* **Result**: Robust retrieval that captures both conceptual meaning and exact keyword hits.

---

## 3. Case Study: The "Parent-Hash Inheritance" Retrieval Bug

During real-world stress testing on 3 multi-domain workbooks (Attention Is All You Need, NISM Equity Derivatives, and NSE Capital Markets), we encountered a subtle retrieval defect that illustrates the danger of improper metadata inheritance.

### The Symptom
* When asked: *"What BLEU score did the Transformer (big) achieve on the English-to-German translation task in the Attention Is All You Need paper?"*
* **DocMind Answer**: *"The document mentions a base model score of 25.8 in Table 3, but contains no mention of the 'Transformer (big)' model."*
* **Metrics**: Faithfulness = **1.00** (No hallucination), but Relevance = **0.00** (Failed to answer).

### The Root Cause Investigation
We traced the document lifecycle:
1. `clean_document(doc)` calculated a SHA-256 `content_hash` on the **entire raw page** (Page 8).
2. When `split_documents(docs)` split Page 8 into 3 chunks (Overview, Table 2 with Big model BLEU 28.4, and Table 3), it copied the parent page's metadata to every child chunk.
3. As a result, **Chunk 0, Chunk 1, and Chunk 2 all possessed the exact same `content_hash`**.
4. When `deduplicate_documents` ran, it observed that Chunk 1 and Chunk 2 had the same hash as Chunk 0, and **discarded them as duplicate text**.
5. **Outcome**: 1,531 chunks were unintentionally trimmed to 270 chunks (only chunk 0 of every page survived). Table 2 was silently dropped!

```text
                                Raw PDF (Page 8)
                     ┌───────────────────────────────────┐
                     │ • Paragraph 1 (Overview)          │
                     │ • Table 2 (BLEU 28.4 for Big)     │
                     │ • Table 3 (Base Model Scores)     │
                     └─────────────────┬─────────────────┘
                                       │
                      clean_document() computes page hash:
                              content_hash = "a7b3...89"
                                       │
                                       ▼
                        split_documents() Chunks Page 8
 ┌───────────────────────┬───────────────────────┬───────────────────────┐
 │ Chunk 0 (Overview)    │ Chunk 1 (Table 2 BLEU)│ Chunk 2 (Table 3)     │
 │ Inherited: "a7b3...89"│ Inherited: "a7b3...89"│ Inherited: "a7b3...89"│
 └───────────┬───────────┴───────────┬───────────┴───────────┬───────────┘
             │                       │                       │
             ▼                       ▼                       ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │                        deduplicate_documents()                        │
 │  Seen "a7b3...89"? First chunk kept. Chunks 1 & 2 dropped as "dupes"! │
 └───────────────────────────────────────────────────────────────────────┘
```

### The Fix
1. **Per-Chunk Hash Computation** ([`ingestion/splitters.py`](file:///c:/FS/langchain_document_qa/ingestion/splitters.py)):
   ```python
   meta["content_hash"] = calculate_content_hash(chunk.page_content)
   ```
2. **Embedding Batching** ([`vectorstore/store.py`](file:///c:/FS/langchain_document_qa/vectorstore/store.py)):
   Batched 1,498 chunks in sets of 50 to prevent payload exhaustion on local Ollama sockets.

### The Result
* **Chunk Retention**: Increased from 270 chunks to **1,498 clean chunks**.
* **Test 1 Relevance**: Increased from **0.00 to 0.80 (PASS)**:
  > *"On the WMT 2014 English-to-German translation task, the model achieved a BLEU score of 28.4 [Source: transformer_paper.pdf]."*
* **Overall Relevance**: Rose from **0.56 to 0.82 (+46% increase)** while maintaining **1.00 Faithfulness**.

---

## 4. RAG Triad Metrics: Why Groundedness Precedes Relevance

Evaluation is the compass of RAG engineering:

```text
               ┌───────────────────────────────────────────────┐
               │                  User Query                   │
               └───────────────┬───────────────┬───────────────┘
                               │               │
                     Answer Relevance    Context Precision
                               │               │
                               ▼               ▼
                       ┌──────────────┐ ┌──────────────┐
                       │  Generated   │ │  Retrieved   │
                       │    Answer    │ │   Context    │
                       └──────┬───────┘ └──────┬───────┘
                              │                │
                              └───────┬────────┘
                                      │
                                 Faithfulness
                                (Groundedness)
```

1. **Faithfulness (Groundedness)**:
   - Does every claim in the answer exist in the retrieved context?
   - Score = 1.00 means the model will honestly report "Information not present" rather than fabricating plausible-sounding hallucinations.
2. **Answer Relevance**:
   - Does the answer directly address the user's intent?
3. **Diagnostic Power**:
   - High Faithfulness (1.00) + Low Relevance (0.56) = **Retrieval Problem (Ingestion / Chunking / Search)**.
   - Low Faithfulness (<0.70) + High Relevance = **LLM Hallucination / Prompt Drift Problem**.

---

## 5. Production Architecture vs. Local Prototyping

| Dimension | Local Prototype | Production Implementation |
| :--- | :--- | :--- |
| **Response Caching** | None (Every query calls LLM ~866ms) | **`CachedRAGService`**: In-Memory TTL / Redis cache serving identical queries in **<1ms with 0 tokens**. |
| **Vector Storage** | Ephemeral FAISS directory | **`ProductionVectorStore`**: PGVector with automatic fallback to local FAISS in zero-Docker mode. |
| **Orchestration Health**| None | **`LivenessProbe` (`/health/live`)** and **`ReadinessProbe` (`/health/ready`)** for Kubernetes routing. |
| **Telemetry & Auditing**| Print statements | **`DocMindTelemetryCallback`**: Hierarchical spans, token sums, p50/p95 latency metrics, and structured JSON logs. |
| **Concurrency** | Single-threaded script | **Multi-worker ASGI cluster** with `X-Request-ID` and `X-Trace-ID` correlation middleware. |

---

## 6. The Next Frontier: "Document Monopoly" in Multi-Aspect Synthesis

While fixing the chunk deduplication bug boosted overall relevance from **0.56 to 0.82 (+46%)**, Test 5 (*Cross-Document Synthesis between Primary Capital Markets and Derivatives Risk Hedging*) revealed the next frontier of RAG optimization: **0.30 relevance despite 1.00 faithfulness**.

### Why Single-Stage Top-N Reranking Fails on Multi-Topic Queries
When a user asks a cross-document comparative question:
> *"Compare the primary market (book building for IPOs) and the derivatives market in terms of how capital is raised versus how risk is hedged."*

The query spans **two completely disjoint semantic concepts**:
1. Concept A: Primary Equity Issuance / IPOs (`nse_financial_markets.pdf`)
2. Concept B: Derivatives Risk Management / Hedging (`nism_derivatives.pdf`)

```text
                                  User Query
               (Concept A: IPOs / Primary) + (Concept B: Derivatives / Hedging)
                                       │
                                       ▼
                       Hybrid RRF Retrieval (top_n=6)
               ┌───────────────────────────────────────────────┐
               │ 1. NSE Primary Market Overview (Score: 0.88)  │
               │ 2. NSE IPO Book Building Process (Score: 0.84)│
               │ 3. NSE Capital Mobilization (Score: 0.81)     │
               │ 4. NISM Futures Hedging Basics (Score: 0.79)  │
               │ 5. NISM Options Risk Protection (Score: 0.76) │
               │ 6. NSE Issue Pricing (Score: 0.74)            │
               └───────────────────────┬───────────────────────┘
                                       │
                                       ▼
                     LLMReranker / ContextCompressor (top_n=3)
                                       │
                                       ▼
            ┌─────────────────────────────────────────────────────┐
            │ 1. NSE Primary Market Overview                      │
            │ 2. NSE IPO Book Building Process                    │
            │ 3. NSE Capital Mobilization                         │
            │ ❌ NISM Derivatives Chunks PRUNED OUT (Starved)!    │
            └──────────────────────────┬──────────────────────────┘
                                       │
                                       ▼
                       LLM Synthesis Prompt (Gemma 4)
             "The provided text only discusses primary markets.
              Derivatives information is absent from the context."
                   (Faithfulness: 1.00 | Relevance: 0.30)
```

### The "Document Monopoly" Problem
When standard top-$N$ rerankers (like cross-encoders or LLM rankers) evaluate candidate chunks against a multi-concept query, chunks closely matching **Concept A** often crowd out all top-$N$ slots, causing **Document Monopoly**:
* Document A fills slots 1, 2, and 3.
* Document B (which contains the critical second half of the comparison) is dropped during reranking or contextual compression.
* The LLM truthfully and faithfully reports that Concept B was not present in the context.

### Architectural Solution: Aspect-Decomposed Multi-Source Retrieval
To achieve 1.00 relevance on cross-document comparative tasks:
1. **Query Decomposition**: Decompose the user question into sub-queries:
   - Sub-query 1: *"How does capital raising and book building work in primary markets?"*
   - Sub-query 2: *"How does risk hedging and mitigation work in derivatives markets?"*
2. **Quota-Based Multi-Source Reranking**: Allocate at least $\lfloor N / 2 \rfloor$ context slots to each distinct sub-query or document source.
3. **Structured Comparison Prompting**: Feed both partition contexts to the synthesis prompt.

---

---

## 8. Multimodal Document RAG: Solving Table Flattening & Vector Chart Blindness

When transitioning DocMind from text-only RAG to **True Multimodal Document RAG**, testing against real investor decks (Tesla Q4 2024 Shareholder Update Deck, 36 pages) revealed two critical architectural barriers in traditional RAG pipelines.

### The Two Major Failures of Text-Only RAG

```text
1. The Table Flattening Problem:
   Raw 2D PDF Grid:                    Standard Text Extractor:
   ┌───────────┬─────────┬─────────┐   "Deliveries Model 3/Y 1,704,093 Other 85,133
   │ Model     │ 2023    │ 2024    │    1,789,226 Production 1,679,338 -5%..."
   ├───────────┼─────────┼─────────┤   
   │ Model 3/Y │ 1.77M   │ 1.70M   │   Result: Column/row alignments are destroyed.
   │ Other     │ 65K     │ 85K     │   Retrieval fails to answer structured queries.
   └───────────┴─────────┴─────────┘

2. The Vector Chart Blindness Problem:
   Slide Deck Chart (Page 14):         Standard Image Extractor (doc.extract_image):
   ┌───────────────────────────────┐   "Found 0 raster images on Page 14."
   │ 📊 Average COGS per vehicle   │   
   │ Q1 $36.8k -> Q4 $34.8k (Low)  │   Result: Because charts are drawn with vector
   │ (Drawn with vector paths)     │   paths, pure image extractors are completely blind.
   └───────────────────────────────┘
```

---

### What Was Added & The Engineering Rationale ("Why")

```text
                     PDF Document
                          │
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
  Text Blocks        Tables (2D)        Visual Charts
 (PyMuPDF Clean)   (pdfplumber MD)    (Vector + Pixmap)
       │                  │                  │
       │                  │                  ▼
       │                  │          Vision LLM Service
       │                  │        (gemma4:cloud / Ollama)
       │                  │        "Extract: Type, Axes,
       │                  │         Numbers & Trends"
       │                  │                  │
       │                  │                  ▼
       │                  │          Visual Description
       │                  │       [Page 14, Type: chart]
       └──────────────────┼──────────────────┘
                          ▼
                Unified Multi-Element
                 LangChain Documents
                          │
                          ▼
            Hybrid Dense + Sparse Indexing
              (nomic-embed-text + BM25)
                          │
                          ▼
                     RRF + Rerank
                          │
                          ▼
             gemma4:cloud Answer Engine
                          │
                          ▼
               Element-Aware Citations
        [Source: file.pdf, Page 5, Type: table]
        [Source: file.pdf, Page 14, Type: chart]
```

1. **`pdfplumber` Table Grid Serialization (`ingestion/multimodal_parser.py`)**:
   - **Why**: Extracts 2D coordinate-bounded tables and serializes them into aligned Markdown tables with preserved column headers and row borders (`| Model | Deliveries 2024 |`). This preserves table relationships for both dense vector embeddings and BM25 exact keyword matching.
2. **Vector Drawing Detection & Pixmap Rendering (`ingestion/multimodal_parser.py`)**:
   - **Why**: Detects pages with vector graphics (`page.get_drawings() >= 8`) and renders 150 DPI high-resolution page pixmaps so that non-bitmap charts (bar charts, line graphs, cost curves) can be visually analyzed.
3. **Native Vision LLM Understanding (`llm/vision.py`)**:
   - **Why**: Sends visual images to `gemma4:cloud` via Ollama with an anti-hallucination prompt extracting: Visual Type, Title/Caption, Axes & Units, Key Data Points, and Trends.
4. **Element-Aware Metadata & Citations (`rag_advanced/pipeline.py`)**:
   - **Why**: Documents retain `element_type: "table" | "chart" | "text"`. Final answers explicitly cite where evidence originated:
     > *"Tesla's total revenue was $97,690M [Source: tesla_shareholder_deck.pdf, Page 5, Type: table]."*
     > *"Average COGS reached an all-time low in Q4 2024 at $34,800 [Source: tesla_shareholder_deck.pdf, Page 14, Type: chart]."*

---

### Empirical Before vs. After Benchmark (Tesla Shareholder Report)

| Test # | Difficulty Level | Test Question | Text-Only Baseline | **Multimodal RAG** | **Faithfulness** | **Relevance** |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Level 1: Text** | What was Tesla's total revenue in 2024? | PASS ($97,690M) | **PASS ($97,690M)** `[Page 5, Type: table]` | **1.00** | **1.00** |
| **2** | **Level 2: Table** | How many total vehicles did Tesla deliver in 2024 by model? | ❌ FAIL (Flattened/Lost) | **PASS (1,789,226 total; Model 3/Y: 1,704,093; Other: 85,133)** `[Page 8, Type: table]` | **1.00** | **1.00** |
| **3** | **Level 3: Visual Chart** | What does the vehicle deliveries and production chart show across quarters in 2024? | ❌ FAIL (Blind to Charts) | **PASS (4Q peak 0.50M; +2% deliveries vs -7% production divergence)** `[Pages 7, 26, Type: chart]` | **1.00** | **1.00** |
| **4** | **Level 3: Visual Chart** | What does the Average COGS per vehicle chart show regarding cost trends in 2024? | ❌ FAIL (Blind to COGS chart) | **PASS (Q1 $36.8k, Q2 $36.9k, Q3 $35.2k, Q4 $34.8k all-time low)** `[Page 14, Type: chart]` | **1.00** | **1.00** |
| --- | --- | --- | --- | --- | --- | --- |
| **AVG** | **Overall Performance** | **Real Multi-Page Deck** | **25% Success** | **🚀 100% Success (4/4 PASS)** | **🛡️ 1.00 / 1.00** | **🎯 1.00 / 1.00** |

---

## 9. Checklist for Production RAG Systems

- [x] **Strict Model Separation**: Dedicated embeddings for retrieval; reasoning models for synthesis.
- [x] **Hybrid Retrieval**: Dense vectors for concepts + BM25 for keywords fused via RRF.
- [x] **Safe Chunk Deduplication**: Compute hashes strictly on chunk text, never inherited page metadata.
- [x] **Batch Vector Generation**: Never send unbatched 1,000+ chunk payloads to embedding endpoints.
- [x] **Sub-Millisecond Query Caching**: Cache deterministic query hashes to eliminate latency and API costs.
- [x] **Multi-Tier Fallbacks**: Provide zero-Docker local operation alongside containerized production deployments.
- [x] **Automated Triad Benchmarking**: Track Faithfulness, Relevance, Precision, and Recall across releases.
- [x] **Independent Layer Diagnostics**: Evaluate retrieval recall, reranker diversity, and LLM groundedness separately.
- [x] **Multimodal Ingestion**: Table Markdown serialization + Vector Chart Pixmap rendering + Vision LLM understanding.
- [x] **Element-Aware Citations**: Explicitly tag whether retrieved evidence originated from text, table, or chart.
