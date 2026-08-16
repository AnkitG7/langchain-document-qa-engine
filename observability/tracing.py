"""Tracing Manager, Context Propagation, and Optional LangSmith Configuration.

Demonstrates:
- ContextVars-based distributed trace propagation across async tasks
- TraceManager for in-memory span indexing and percentile latency analytics
- Seamless environment configuration for LangSmith tracing
"""

import os
import time
import math
import uuid
import contextvars
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .callbacks import ExecutionTrace, DocMindTelemetryCallback

_current_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_trace_id", default=None)
_current_callback_var: contextvars.ContextVar[Optional[DocMindTelemetryCallback]] = contextvars.ContextVar("current_callback", default=None)


def get_current_trace_id() -> Optional[str]:
    """Returns the trace_id of the currently active context."""
    return _current_trace_id_var.get()


def get_current_callback() -> Optional[DocMindTelemetryCallback]:
    """Returns the DocMindTelemetryCallback attached to current context."""
    return _current_callback_var.get()


@contextmanager
def trace_context(trace_id: Optional[str] = None, session_id: Optional[str] = None):
    """Context manager binding a trace_id and telemetry callback to the current execution thread."""
    t_id = trace_id or f"trace_{uuid.uuid4().hex[:10]}"
    cb = DocMindTelemetryCallback(trace_id=t_id, session_id=session_id)

    token_trace = _current_trace_id_var.set(t_id)
    token_cb = _current_callback_var.set(cb)
    try:
        yield cb
    finally:
        _current_trace_id_var.reset(token_trace)
        _current_callback_var.reset(token_cb)


class TraceManager:
    """Stores, queries, and aggregates execution traces and runtime telemetry."""

    def __init__(self):
        self._traces: Dict[str, ExecutionTrace] = {}

    def record_trace(self, trace: ExecutionTrace) -> None:
        """Indexes an execution trace."""
        self._traces[trace.trace_id] = trace

    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        """Retrieves a single trace by ID."""
        return self._traces.get(trace_id)

    def list_traces(self, limit: int = 50) -> List[ExecutionTrace]:
        """Returns recent execution traces."""
        all_traces = list(self._traces.values())
        return all_traces[-limit:]

    def clear(self) -> None:
        """Clears all stored traces."""
        self._traces.clear()

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Computes aggregate analytics: total traces, token sums, p50/p95 latencies, error rate."""
        traces = list(self._traces.values())
        total_runs = len(traces)
        if total_runs == 0:
            return {
                "total_traces": 0,
                "total_tokens": 0,
                "total_estimated_cost_usd": 0.0,
                "error_rate_pct": 0.0,
                "avg_duration_ms": 0.0,
                "p50_duration_ms": 0.0,
                "p95_duration_ms": 0.0,
            }

        total_tokens = sum(t.total_tokens for t in traces)
        total_cost = sum(t.estimated_cost_usd for t in traces)
        error_count = sum(1 for t in traces if t.status == "error")
        durations = sorted(t.total_duration_ms for t in traces)

        p50_idx = int(0.50 * total_runs)
        p95_idx = min(total_runs - 1, int(0.95 * total_runs))

        return {
            "total_traces": total_runs,
            "total_tokens": total_tokens,
            "total_estimated_cost_usd": round(total_cost, 6),
            "error_rate_pct": round((error_count / total_runs) * 100.0, 1),
            "avg_duration_ms": round(sum(durations) / total_runs, 2),
            "p50_duration_ms": round(durations[p50_idx], 2),
            "p95_duration_ms": round(durations[p95_idx], 2),
        }


def configure_langsmith(project_name: str = "DocMind-Document-QA") -> bool:
    """Configures LangSmith environment variables if API key is provided."""
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        return True
    return False
