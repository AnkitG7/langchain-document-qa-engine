# 🧠 DocMind: Intelligent Document Q&A Engine

> **A comprehensive, production-grade Document Q&A and Analysis Engine built on modern LangChain.**

---

## 🗺️ Progressive Architecture Roadmap

| Phase | Module | Concepts Mastered | Status |
| :--- | :--- | :--- | :---: |
| **Phase 1** | `llm/`, `chains/` | Multi-Provider LLMs, LCEL Pipes, Output Parsers, Parallel Runnables, Summarization, Comparison, Composition | ✅ **Done** |
| **Phase 2** | `ingestion/` | Multi-Format Loaders (PDF, Web, CSV, MD), Chunking Strategies, Metadata Enrichment, Cleaning Pipeline | ✅ **Done** |
| **Phase 3** | `vectorstore/` | Pluggable Dedicated Embeddings, Chroma & FAISS Stores, Similarity, MMR & Threshold Retrieval | ✅ **Done** |
| **Phase 4** | `memory/` | Modern Message History (`RunnableWithMessageHistory`), Sliding Windows, History-Aware Contextualization | ✅ **Done** |
| **Phase 5** | `tools/`, `agent/` | Tool-Calling Agents, Dynamic Tool Routing, Safe Math & Catalog Tools, Multi-Step Reasoning | ✅ **Done** |
| **Phase 6** | `api/` | FastAPI REST API, Server-Sent Events (SSE) Streaming | ⏳ *Next* |
| **Phase 7** | `rag_advanced/` | Query Transformation (HyDE, Multi-Query, Step-Back), Contextual Compression, Reranking | ⏳ *Upcoming* |
| **Phase 8** | `evaluation/` | RAG Evaluation Metrics (Faithfulness, Relevancy, Precision), Automated Benchmarking | ⏳ *Upcoming* |
| **Phase 9** | `observability/` | Tracing, LangSmith integration, Token & Latency tracking | ⏳ *Upcoming* |
| **Phase 10** | `production/` | Optional Docker Compose, PostgreSQL + PGVector, Redis Caching | ⏳ *Upcoming* |

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone/Open repository
cd langchain_document_qa

# Activate Python Virtual Environment
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 3. Run Interactive Demos
```bash
# Phase 1: LLM & LCEL Chains Demo
python examples/demo_phase1.py

# Phase 2: Ingestion & Chunking Demo
python examples/demo_phase2.py

# Phase 3: Embeddings, Vector Stores & Retrieval Demo
python examples/demo_phase3.py

# Phase 4: Modern Message History & Conversational RAG Demo
python examples/demo_phase4.py

# Phase 5: Tools & Tool-Calling Agents Demo
python examples/demo_phase5.py
```

### 4. Run Automated Test Suite
```bash
# Run all 55 regression tests across all phases
pytest tests/ -v

---

## 🧩 Phase 1 Concepts & Modules (`llm/`, `chains/`)
- **LLM Abstraction (`llm/provider.py`)**: Multi-provider support (Ollama Gemma, OpenAI, Anthropic, Fake) + `.with_fallbacks()`.
- **LCEL Question Answering (`chains/qa_chain.py`)**: Pipe syntax `prompt | llm | parser`, Pydantic validation, and `RunnableParallel`.
- **Summarization (`chains/summary_chain.py`)**: Stuff, Map-Reduce (`.batch()`), and Refine patterns.
- **Comparison (`chains/compare_chain.py`)**: Multi-document contrast & trade-off reporting.
- **Composition (`chains/composition.py`)**: 3-step cognitive pipeline (Extract ➔ Critique ➔ Synthesize).

---

## 📄 Phase 2 Concepts & Modules (`ingestion/`)
- **Multi-Format Loaders (`ingestion/loaders.py`)**:
  - `PDFDocumentLoader`: PDF parsing with page-level metadata.
  - `TextDocumentLoader` & `MarkdownLoader`: Encoding-resilient document loaders.
  - `CSVDocumentLoader`: Tabular row-by-row parsing with column headers.
  - `WebDocumentLoader`: Web scraping with HTML boilerplate stripping via BeautifulSoup.
  - `load_document()` / `load_documents_batch()`: Universal auto-dispatch factory.
- **Cleaning & Enrichment (`ingestion/cleaner.py`)**:
  - `clean_text()`: Unicode NFKC normalization, whitespace & control code cleaning.
  - `enrich_document_metadata()`: SHA-256 content hashing, word/token estimation, unique doc IDs.
  - `deduplicate_documents()`: Content hash-based chunk deduplication.
- **Chunking Strategies (`ingestion/splitters.py`)**:
  - `RecursiveCharacterTextSplitter`: Paragraph/sentence hierarchical chunking.
  - `TokenTextSplitter`: Token-budget chunking via `tiktoken`.
  - `MarkdownHeaderTextSplitter`: Header-hierarchy aware splitting.
  - `split_documents()`: Preserves parent lineage with chunk indexing (`chunk_index`, `parent_doc_id`, `chunk_id`).
- **Ingestion Orchestrator (`ingestion/pipeline.py`)**:
  - `IngestionPipeline`: Complete batch pipeline generating structured `IngestionReport` audit telemetry.

---

## 🗄️ Phase 3 Concepts & Modules (`vectorstore/`)
- **Dedicated Embedding Separation (`vectorstore/embedder.py`)**:
  - Distinct embedding factory `get_embeddings()` using `nomic-embed-text` (Ollama), `text-embedding-3-small` (OpenAI), or deterministic fake embeddings.
  - `verify_embeddings()` health check.
- **Pluggable Local Vector Stores (`vectorstore/store.py`)**:
  - `Chroma`: In-memory & SQLite-backed persistent collections.
  - `FAISS`: In-memory similarity indexing, serialized index saving (`save_local`), and loading (`load_local`).
  - `VectorStoreManager`: Unified storage adapter.
- **Search & Retrieval Modes (`vectorstore/retriever.py`)**:
  - `similarity`: Standard k-NN search.
  - `mmr`: Maximal Marginal Relevance for balancing relevance and diversity.
  - `similarity_score_threshold`: Score threshold filtering out low-similarity noise.
  - Metadata filtering by `file_type`, `doc_id`, etc.

---

## 💬 Phase 4 Concepts & Modules (`memory/`)
- **Session History Manager (`memory/history_store.py`)**:
  - `SessionHistoryManager`: Multi-session isolation with `InMemoryChatMessageHistory` and `FileSessionHistory` (JSON persistence).
- **Message Windowing & Trimming (`memory/trimmer.py`)**:
  - `trim_conversation_history()` & `create_message_trimmer()`: Sliding conversation window keeping system instructions while bounding token growth.
- **History-Aware Conversational RAG (`memory/conversational_rag.py`)**:
  - `create_contextualize_question_chain()`: Reformulates ambiguous follow-up turns into standalone search queries.
  - `RunnableWithMessageHistory`: Modern LangChain wrapper binding pure LCEL pipelines to dynamic session stores.
- **Progressive Summarization (`memory/summary_memory.py`)**:
  - `ProgressiveConversationSummary`: Periodically summarizes older turns into a condensed narrative when conversation grows.
- **Legacy vs. Modern Architecture (`memory/legacy_comparison.py`)**:
  - Educational explanation of why modern LangChain moved away from mutable `ConversationBufferMemory` to pure stateless LCEL runnables + external message stores.

---

## 🛠️ Phase 5 Concepts & Modules (`tools/`, `agent/`)
- **Specialized Tool Definitions (`tools/`)**:
  - `calculator_tool`: Safe mathematical AST evaluator for arithmetic, sums, averages, and percentages without unsafe `eval()`.
  - `metadata_catalog_tool`: File inventory inspector, type filtering, and document metadata lookups.
  - `create_search_tool`: Dynamic vector retrieval tool with source attribution and metadata filters.
  - `get_docmind_tools`: Dynamic tool suite registry.
- **Modern Tool-Calling Agent (`agent/doc_agent.py`)**:
  - `create_agent`: LangChain graph-based tool-calling agent with native model tool binding.
  - `DocMindAgent`: Multi-step reasoning engine (Thought ➔ Tool Call ➔ Observation ➔ Final Answer) with multi-turn session persistence.
  - Step tracing: Extraction of intermediate tool calls and observations for transparency.
