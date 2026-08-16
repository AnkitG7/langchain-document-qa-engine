"""Custom LangChain Telemetry Callback Handler & Span Data Models.

Demonstrates:
- BaseCallbackHandler lifecycle interception:
  * on_chain_start / end / error
  * on_llm_start / end / error
  * on_tool_start / end / error
  * on_retriever_start / end / error
- Token counting, model extraction, and duration tracking
- Request correlation and hierarchical execution trace construction
"""

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class ExecutionSpan(BaseModel):
    """Single execution span representing a unit of work (LLM, Chain, Tool, Retriever)."""
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_span_id: Optional[str] = None
    span_type: Literal["chain", "llm", "tool", "retriever"]
    name: str
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    """Complete execution trace for an end-to-end request or agent turn."""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    session_id: Optional[str] = None
    spans: List[ExecutionSpan] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    status: Literal["success", "error"] = "success"
    error_message: Optional[str] = None


class DocMindTelemetryCallback(BaseCallbackHandler):
    """Custom LangChain callback handler collecting fine-grained execution spans and token metrics."""

    def __init__(self, trace_id: Optional[str] = None, session_id: Optional[str] = None):
        super().__init__()
        self.trace_id = trace_id or str(uuid.uuid4())[:12]
        self.session_id = session_id
        self.spans: Dict[str, ExecutionSpan] = {}
        self.completed_spans: List[ExecutionSpan] = []
        self.parent_stack: List[str] = []
        self.start_time = time.time()
        self.has_error = False
        self.error_msg: Optional[str] = None

    # --- Chain Events ---
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "Chain") if serialized else "Chain"
        span = ExecutionSpan(
            span_id=str(run_id),
            parent_span_id=str(parent_run_id) if parent_run_id else None,
            span_type="chain",
            name=name,
            inputs={"inputs_preview": str(inputs)[:300]},
        )
        self.spans[str(run_id)] = span

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        if run_key in self.spans:
            span = self.spans.pop(run_key)
            span.end_time = time.time()
            span.duration_ms = round((span.end_time - span.start_time) * 1000, 2)
            span.outputs = {"outputs_preview": str(outputs)[:300]}
            self.completed_spans.append(span)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self.has_error = True
        self.error_msg = str(error)
        run_key = str(run_id)
        if run_key in self.spans:
            span = self.spans.pop(run_key)
            span.end_time = time.time()
            span.duration_ms = round((span.end_time - span.start_time) * 1000, 2)
            span.error = str(error)
            self.completed_spans.append(span)

    # --- LLM Events ---
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "LLM") if serialized else "LLM"
        span = ExecutionSpan(
            span_id=str(run_id),
            parent_span_id=str(parent_run_id) if parent_run_id else None,
            span_type="llm",
            name=name,
            inputs={"prompts_count": len(prompts), "preview": prompts[0][:200] if prompts else ""},
        )
        self.spans[str(run_id)] = span

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        if run_key in self.spans:
            span = self.spans.pop(run_key)
            span.end_time = time.time()
            span.duration_ms = round((span.end_time - span.start_time) * 1000, 2)

            # Token Extraction
            llm_output = response.llm_output or {}
            usage = llm_output.get("token_usage", {})
            if usage:
                span.prompt_tokens = usage.get("prompt_tokens", 0)
                span.completion_tokens = usage.get("completion_tokens", 0)
                span.total_tokens = usage.get("total_tokens", span.prompt_tokens + span.completion_tokens)
            else:
                # Estimate tokens heuristic if usage is not reported by provider
                total_text = "".join(gen.text for gen_list in response.generations for gen in gen_list)
                span.completion_tokens = max(1, len(total_text) // 4)
                span.prompt_tokens = max(1, len(span.inputs.get("preview", "")) // 4)
                span.total_tokens = span.prompt_tokens + span.completion_tokens

            output_text = response.generations[0][0].text if response.generations and response.generations[0] else ""
            span.outputs = {"output_preview": output_text[:300]}
            self.completed_spans.append(span)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self.has_error = True
        self.error_msg = str(error)
        run_key = str(run_id)
        if run_key in self.spans:
            span = self.spans.pop(run_key)
            span.end_time = time.time()
            span.duration_ms = round((span.end_time - span.start_time) * 1000, 2)
            span.error = str(error)
            self.completed_spans.append(span)

    # --- Tool Events ---
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "Tool") if serialized else "Tool"
        span = ExecutionSpan(
            span_id=str(run_id),
            parent_span_id=str(parent_run_id) if parent_run_id else None,
            span_type="tool",
            name=name,
            inputs={"tool_input": input_str},
        )
        self.spans[str(run_id)] = span

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        if run_key in self.spans:
            span = self.spans.pop(run_key)
            span.end_time = time.time()
            span.duration_ms = round((span.end_time - span.start_time) * 1000, 2)
            span.outputs = {"tool_output": str(output)[:300]}
            self.completed_spans.append(span)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self.has_error = True
        self.error_msg = str(error)
        run_key = str(run_id)
        if run_key in self.spans:
            span = self.spans.pop(run_key)
            span.end_time = time.time()
            span.duration_ms = round((span.end_time - span.start_time) * 1000, 2)
            span.error = str(error)
            self.completed_spans.append(span)

    # --- Retriever Events ---
    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "Retriever") if serialized else "Retriever"
        span = ExecutionSpan(
            span_id=str(run_id),
            parent_span_id=str(parent_run_id) if parent_run_id else None,
            span_type="retriever",
            name=name,
            inputs={"query": query},
        )
        self.spans[str(run_id)] = span

    def on_retriever_end(
        self,
        documents: List[Any],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        if run_key in self.spans:
            span = self.spans.pop(run_key)
            span.end_time = time.time()
            span.duration_ms = round((span.end_time - span.start_time) * 1000, 2)
            span.outputs = {"retrieved_count": len(documents)}
            self.completed_spans.append(span)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self.has_error = True
        self.error_msg = str(error)
        run_key = str(run_id)
        if run_key in self.spans:
            span = self.spans.pop(run_key)
            span.end_time = time.time()
            span.duration_ms = round((span.end_time - span.start_time) * 1000, 2)
            span.error = str(error)
            self.completed_spans.append(span)

    # --- Trace Assembly ---
    def get_trace(self) -> ExecutionTrace:
        """Assembles the final ExecutionTrace with totals and estimated costs."""
        # Flush any remaining unended spans
        now = time.time()
        for span in list(self.spans.values()):
            span.end_time = now
            span.duration_ms = round((now - span.start_time) * 1000, 2)
            self.completed_spans.append(span)
        self.spans.clear()

        total_duration = round((time.time() - self.start_time) * 1000, 2)
        p_tokens = sum(s.prompt_tokens for s in self.completed_spans)
        c_tokens = sum(s.completion_tokens for s in self.completed_spans)
        tot_tokens = sum(s.total_tokens for s in self.completed_spans)

        # Standard OpenAI/Claude blend cost estimation ($0.0015 / 1k prompt, $0.002 / 1k completion)
        estimated_cost = (p_tokens * 0.0000015) + (c_tokens * 0.000002)

        return ExecutionTrace(
            trace_id=self.trace_id,
            session_id=self.session_id,
            spans=self.completed_spans,
            total_duration_ms=total_duration,
            total_tokens=tot_tokens,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            estimated_cost_usd=round(estimated_cost, 6),
            status="error" if self.has_error else "success",
            error_message=self.error_msg,
        )
