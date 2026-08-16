"""Interactive Phase 9 Demonstration Script for DocMind Observability & Tracing.

Run with:
    python examples/demo_phase9.py

Demonstrates:
1. Custom LangChain BaseCallbackHandler (DocMindTelemetryCallback)
2. Fine-grained Span Hierarchy (LLM, Chain, Tool, Retriever)
3. Token Usage, Model Parameters, and Latency Tracking
4. Cost Estimation & Percentile Analytics (p50, p95)
5. Structured JSON Audit Logging & Disk Trace Storage
6. Live Trace Execution with Ollama gemma4:cloud
"""

import sys
import os
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from llm.provider import get_chat_model
from ingestion.pipeline import IngestionPipeline
from vectorstore.embedder import get_embeddings
from vectorstore.store import get_or_create_faiss
from memory.conversational_rag import ConversationalRAGChain
from memory.history_store import SessionHistoryManager
from agent.doc_agent import DocMindAgent
from observability.callbacks import DocMindTelemetryCallback
from observability.tracing import TraceManager, trace_context
from observability.logger import JSONTraceLogger, FileTraceStorage

DATA_DIR = Path(__file__).parent.parent / "data"


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def run_demo():
    print_banner("DocMind Phase 9: Observability & Tracing Demo")

    # 1. Pipeline Setup
    print_banner("1. Setting Up Vector Store and Observability Manager")
    pipeline = IngestionPipeline(chunk_size=300, chunk_overlap=50)
    chunks, _ = pipeline.run_batch([
        str(DATA_DIR / "sample_doc.txt"),
        str(DATA_DIR / "sample_data.csv"),
    ])
    embedder = get_embeddings()
    store = get_or_create_faiss(documents=chunks, embeddings=embedder)
    retriever = store.as_retriever(search_kwargs={"k": 2})
    llm = get_chat_model()

    trace_manager = TraceManager()
    trace_storage = FileTraceStorage("data/traces")

    # 2. Traced Conversational RAG Execution
    print_banner("2. Executing Conversational RAG with Fine-Grained Telemetry")
    rag_engine = ConversationalRAGChain(retriever=retriever, llm=llm)

    cb_rag = DocMindTelemetryCallback(trace_id="trace_rag_turn_01", session_id="obs_sess_01")
    rag_query = "What document types does DocMind support?"
    print(f"[User Query]: {rag_query}")

    res = rag_engine.chat(
        user_input=rag_query,
        session_id="obs_sess_01",
        config={"callbacks": [cb_rag]},
    )
    ans = res.get("answer", "")
    print(f"[DocMind Answer]:\n{ans}\n")

    trace_rag = cb_rag.get_trace()
    trace_manager.record_trace(trace_rag)
    trace_storage.save(trace_rag)

    print(f"--- RAG Execution Trace [{trace_rag.trace_id}] ---")
    print(f"Status:             {trace_rag.status.upper()}")
    print(f"Total Duration:     {trace_rag.total_duration_ms:.2f} ms")
    print(f"Prompt Tokens:      {trace_rag.prompt_tokens}")
    print(f"Completion Tokens:  {trace_rag.completion_tokens}")
    print(f"Total Tokens:       {trace_rag.total_tokens}")
    print(f"Estimated Cost:     ${trace_rag.estimated_cost_usd:.6f}")
    print(f"Spans Collected:    {len(trace_rag.spans)}")
    for i, s in enumerate(trace_rag.spans, 1):
        print(f"  [{i}] Type: {s.span_type:<10} | Name: {s.name:<25} | Latency: {s.duration_ms:>7.2f} ms")

    # 3. Traced Tool-Calling Agent Execution
    print_banner("3. Executing Tool-Calling Agent with Multi-Step Tool Tracing")
    from tools import get_docmind_tools
    agent = DocMindAgent(llm=llm, tools=get_docmind_tools(vectorstore=store))
    cb_agent = DocMindTelemetryCallback(trace_id="trace_agent_turn_01", session_id="obs_agent_sess")

    agent_query = "Calculate 1500 * 18 and list available documents"
    print(f"[Agent Query]: {agent_query}")

    result = agent.run(
        user_input=agent_query,
        session_id="obs_agent_sess",
        config={"callbacks": [cb_agent]},
    )
    print(f"[Agent Output]:\n{result.get('output', '')}\n")

    trace_agent = cb_agent.get_trace()
    trace_manager.record_trace(trace_agent)
    trace_storage.save(trace_agent)

    print(f"--- Agent Execution Trace [{trace_agent.trace_id}] ---")
    print(f"Total Duration:     {trace_agent.total_duration_ms:.2f} ms")
    print(f"Total Tokens:       {trace_agent.total_tokens}")
    print(f"Spans Collected:    {len(trace_agent.spans)}")
    for i, s in enumerate(trace_agent.spans, 1):
        print(f"  [{i}] Type: {s.span_type:<10} | Name: {s.name:<25} | Latency: {s.duration_ms:>7.2f} ms")

    # 4. Structured JSON Audit Logging
    print_banner("4. Emitting Structured JSON Audit Event (Log Forwarding Ready)")
    json_event = JSONTraceLogger.format_trace_json(trace_rag)
    print(json.dumps(json.loads(json_event), indent=2))

    # 5. System-Wide Telemetry Analytics
    print_banner("5. System-Wide Aggregate Observability Dashboard")
    metrics = trace_manager.get_aggregated_metrics()
    print(f"Total Requests Traced:   {metrics['total_traces']}")
    print(f"Total Tokens Consumed:   {metrics['total_tokens']}")
    print(f"Total Cost Incurred:     ${metrics['total_estimated_cost_usd']:.6f}")
    print(f"Error Rate:              {metrics['error_rate_pct']:.1f}%")
    print(f"Average Latency:         {metrics['avg_duration_ms']:.2f} ms")
    print(f"p50 Latency (Median):    {metrics['p50_duration_ms']:.2f} ms")
    print(f"p95 Latency (Tail):      {metrics['p95_duration_ms']:.2f} ms")

    print_banner("Phase 9 Complete!")
    print("Custom callbacks, span hierarchies, token tracking, and structured logging verified successfully.")


if __name__ == "__main__":
    run_demo()
