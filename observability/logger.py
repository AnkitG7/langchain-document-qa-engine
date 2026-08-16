"""Structured JSON Audit Logger & Disk-Backed Trace Persistence.

Demonstrates:
- Structured JSON logging formatted for log shippers (Datadog, ElasticSearch, CloudWatch)
- File-based persistence of execution traces for offline review
- Global singleton TraceManager accessor
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .callbacks import ExecutionTrace
from .tracing import TraceManager

logger = logging.getLogger("DocMind.Observability")

_global_trace_manager: Optional[TraceManager] = None


def get_global_trace_manager() -> TraceManager:
    """Singleton accessor for the global TraceManager."""
    global _global_trace_manager
    if _global_trace_manager is None:
        _global_trace_manager = TraceManager()
    return _global_trace_manager


class JSONTraceLogger:
    """Emits structured JSON execution events to loggers."""

    @staticmethod
    def format_trace_json(trace: ExecutionTrace) -> str:
        """Serializes an ExecutionTrace into a single-line compact JSON string."""
        return json.dumps(trace.model_dump(), ensure_ascii=False)

    @classmethod
    def log_trace(cls, trace: ExecutionTrace) -> None:
        """Logs an execution trace as a structured JSON record."""
        payload = cls.format_trace_json(trace)
        if trace.status == "error":
            logger.error(f"[DocMind Trace Event] {payload}")
        else:
            logger.info(f"[DocMind Trace Event] {payload}")


class FileTraceStorage:
    """Stores and retrieves traces from disk as JSON files."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir or "data/traces")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trace: ExecutionTrace) -> Path:
        """Saves an execution trace to disk."""
        target_path = self.storage_dir / f"{trace.trace_id}.json"
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(trace.model_dump(), f, indent=2, ensure_ascii=False)
        return target_path

    def load(self, trace_id: str) -> Optional[ExecutionTrace]:
        """Loads an execution trace from disk."""
        target_path = self.storage_dir / f"{trace_id}.json"
        if not target_path.exists():
            return None
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExecutionTrace(**data)
