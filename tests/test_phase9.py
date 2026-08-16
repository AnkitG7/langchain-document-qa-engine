"""Unit and integration tests for Phase 9: Observability & Tracing.

Covers:
- Custom LangChain Telemetry Callback (Chain, LLM, Tool, Retriever spans)
- Token counting and cost estimation
- Distributed Trace Context and correlation
- TraceManager aggregation (p50/p95 latencies, error rates)
- JSON Logger and Disk-Backed FileTraceStorage
- End-to-end execution tracing with LangChain runnables
"""

import time
import uuid
import pytest
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.outputs import LLMResult, Generation

from observability.callbacks import (
    ExecutionSpan,
    ExecutionTrace,
    DocMindTelemetryCallback,
)
from observability.tracing import (
    TraceManager,
    trace_context,
    get_current_trace_id,
    configure_langsmith,
)
from observability.logger import (
    JSONTraceLogger,
    FileTraceStorage,
    get_global_trace_manager,
)


class TestTelemetryCallbacks:
    def test_callback_lifecycle_events(self):
        cb = DocMindTelemetryCallback(trace_id="test_trace_01", session_id="sess_01")
        chain_run_id = uuid.uuid4()
        llm_run_id = uuid.uuid4()
        tool_run_id = uuid.uuid4()
        retriever_run_id = uuid.uuid4()

        # 1. Chain Start/End
        cb.on_chain_start({"name": "TestQAChain"}, {"question": "What is DocMind?"}, run_id=chain_run_id)
        # 2. Retriever Start/End
        cb.on_retriever_start({"name": "FAISSRetriever"}, "What is DocMind?", run_id=retriever_run_id, parent_run_id=chain_run_id)
        cb.on_retriever_end([{"page_content": "Doc 1"}, {"page_content": "Doc 2"}], run_id=retriever_run_id)
        # 3. Tool Start/End
        cb.on_tool_start({"name": "Calculator"}, "5 * 5", run_id=tool_run_id, parent_run_id=chain_run_id)
        cb.on_tool_end("25", run_id=tool_run_id)
        # 4. LLM Start/End
        cb.on_llm_start({"name": "GemmaLLM"}, ["Answer this question"], run_id=llm_run_id, parent_run_id=chain_run_id)
        mock_result = LLMResult(
            generations=[[Generation(text="DocMind is an intelligent document analysis engine.")]],
            llm_output={"token_usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}},
        )
        cb.on_llm_end(mock_result, run_id=llm_run_id)
        cb.on_chain_end({"output": "DocMind is an intelligent document analysis engine."}, run_id=chain_run_id)

        trace = cb.get_trace()
        assert trace.trace_id == "test_trace_01"
        assert trace.session_id == "sess_01"
        assert len(trace.spans) == 4
        assert trace.total_tokens == 20
        assert trace.prompt_tokens == 12
        assert trace.completion_tokens == 8
        assert trace.estimated_cost_usd > 0.0
        assert trace.status == "success"

    def test_callback_error_interception(self):
        cb = DocMindTelemetryCallback(trace_id="error_trace_01")
        run_id = uuid.uuid4()
        cb.on_chain_start({"name": "FailingChain"}, {}, run_id=run_id)
        cb.on_chain_error(ValueError("Connection to vector store timed out"), run_id=run_id)

        trace = cb.get_trace()
        assert trace.status == "error"
        assert "Connection to vector store timed out" in trace.error_message


class TestTraceManagerAndContext:
    def test_trace_context_propagation(self):
        with trace_context(trace_id="custom_trace_123") as cb:
            assert get_current_trace_id() == "custom_trace_123"
            assert cb.trace_id == "custom_trace_123"

        assert get_current_trace_id() is None

    def test_trace_manager_metrics_aggregation(self):
        mgr = TraceManager()
        t1 = ExecutionTrace(
            trace_id="t1",
            total_duration_ms=100.0,
            total_tokens=50,
            estimated_cost_usd=0.001,
            status="success",
        )
        t2 = ExecutionTrace(
            trace_id="t2",
            total_duration_ms=200.0,
            total_tokens=150,
            estimated_cost_usd=0.003,
            status="error",
        )
        mgr.record_trace(t1)
        mgr.record_trace(t2)

        assert mgr.get_trace("t1") is not None
        assert len(mgr.list_traces()) == 2

        metrics = mgr.get_aggregated_metrics()
        assert metrics["total_traces"] == 2
        assert metrics["total_tokens"] == 200
        assert metrics["error_rate_pct"] == 50.0
        assert metrics["avg_duration_ms"] == 150.0
        assert metrics["p95_duration_ms"] == 200.0


class TestLoggerAndStorage:
    def test_json_trace_logger(self):
        t = ExecutionTrace(trace_id="log_trace_01", total_tokens=100, status="success")
        json_str = JSONTraceLogger.format_trace_json(t)
        assert "log_trace_01" in json_str
        assert '"total_tokens": 100' in json_str

    def test_file_trace_storage(self, tmp_path):
        storage = FileTraceStorage(storage_dir=str(tmp_path / "traces"))
        t = ExecutionTrace(trace_id="disk_trace_01", total_tokens=250, status="success")

        saved_path = storage.save(t)
        assert saved_path.exists()

        loaded = storage.load("disk_trace_01")
        assert loaded is not None
        assert loaded.trace_id == "disk_trace_01"
        assert loaded.total_tokens == 250


class TestLangChainLiveTracing:
    def test_lcel_runnable_with_telemetry_callback(self):
        mock_llm = FakeListChatModel(responses=["Observability gives full execution visibility."])
        prompt = ChatPromptTemplate.from_template("Explain: {topic}")
        chain = prompt | mock_llm | StrOutputParser()

        cb = DocMindTelemetryCallback(trace_id="live_lcel_trace")
        result = chain.invoke({"topic": "Observability"}, config={"callbacks": [cb]})

        assert "Observability gives full execution visibility." in result
        trace = cb.get_trace()
        assert trace.trace_id == "live_lcel_trace"
        assert len(trace.spans) >= 2  # Prompt/Chain + LLM
        assert trace.status == "success"
