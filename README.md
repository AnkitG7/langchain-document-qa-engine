# 🧠 DocMind: Intelligent Document Q&A Engine

> **A comprehensive, production-grade Document Q&A and Analysis Engine built on modern LangChain.**

---

## 🗺️ Progressive Architecture Roadmap

| Phase | Module | Concepts Mastered | Status |
| :--- | :--- | :--- | :---: |
| **Phase 1** | `llm/`, `chains/` | Multi-Provider LLMs, LCEL Pipes, Output Parsers, Parallel Runnables, Summarization, Comparison, Composition | ✅ **Done** |
| **Phase 2** | `ingestion/` | Multi-Format Loaders (PDF, Web, CSV, MD), Chunking Strategies, Metadata Enrichment, Cleaning Pipeline | ✅ **Done** |
| **Phase 3** | `vectorstore/` | Pluggable Embeddings, Local In-Memory / Disk Stores (Chroma, FAISS), Similarity, MMR & Threshold Retrieval | ⏳ *Next* |
| **Phase 4** | `memory/` | Modern Message History (`RunnableWithMessageHistory`), Sliding Windows, Token-Aware State | ⏳ *Upcoming* |
| **Phase 5** | `tools/`, `agent/` | Tool-Calling Agents, ReAct Loop, Document Search & Numeric Analysis Tools | ⏳ *Upcoming* |
| **Phase 6** | `api/` | FastAPI REST API, Server-Sent Events (SSE) Streaming | ⏳ *Upcoming* |
| **Phase 7** | `rag_advanced/` | Query Transformation (HyDE, Multi-Query, Step-Back), Contextual Compression, Reranking | ⏳ *Upcoming* |
| **Phase 8** | `evaluation/` | RAG Evaluation Metrics (Faithfulness, Relevancy, Precision), Automated Benchmarking | ⏳ *Upcoming* |
| **Phase 9** | `observability/` | Tracing, LangSmith integration, Token & Latency tracking | ⏳ *Upcoming* |
| **Phase 10** | `production/` | Optional Docker Compose, PostgreSQL + PGVector, Redis Caching | ⏳ *Upcoming* |

---

## 🚀 Quick Start (Phase 1)

### 1. Environment Setup
```bash
# Clone/Open repository
cd langchain_document_qa

# Create Python Virtual Environment (Local, No Docker required)
python -m venv venv
# On Windows:
.\venv\Scripts\activate

# Install Phase 1 dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env` and set your API keys:
```bash
cp .env.example .env
```
*(Note: If no API key is provided, the system automatically runs in simulated offline mode for testing!)*

### 3. Run Interactive Demos
```bash
# Phase 1: LLM & LCEL Chains Demo
python examples/demo_phase1.py

# Phase 2: Ingestion & Chunking Demo
python examples/demo_phase2.py
```

### 4. Run Automated Unit Tests
```bash
# Run all unit tests
pytest tests/ -v

# Or run Phase 2 specifically
pytest tests/test_phase2.py -v
```

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
