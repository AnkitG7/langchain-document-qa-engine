"""Observability & Tracing Module for DocMind.

Demonstrates:
- Hierarchical Execution Spans and Traces
- Custom LangChain Telemetry Callbacks (LLM, Chain, Tool, Retriever)
- Request Correlation, Span Hierarchies, and Latency/Token Profiling
- Cost Estimation & Percentile Latencies
- Optional LangSmith Environment Configuration
- Structured JSON Audit Logging and Disk Persistence
"""

from .callbacks import (
    ExecutionSpan,
    ExecutionTrace,
    DocMindTelemetryCallback,
)
from .tracing import (
    TraceManager,
    trace_context,
    get_current_trace_id,
    configure_langsmith,
)
from .logger import (
    JSONTraceLogger,
    FileTraceStorage,
    get_global_trace_manager,
)

__all__ = [
    # Callbacks & Data Models
    "ExecutionSpan",
    "ExecutionTrace",
    "DocMindTelemetryCallback",
    # Tracing & Context
    "TraceManager",
    "trace_context",
    "get_current_trace_id",
    "configure_langsmith",
    # Logging & Persistence
    "JSONTraceLogger",
    "FileTraceStorage",
    "get_global_trace_manager",
]
