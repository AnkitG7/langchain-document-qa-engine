"""Unit and integration tests for Phase 5: Tools and Tool-Calling Agents.

Covers:
- CalculatorTool: Arithmetic, aggregations, safe AST parsing, security isolation
- MetadataCatalogTool: Catalog inspection, filtering by file type, file details
- SearchDocumentsTool: Vector retriever tool integration and formatted citations
- Tool Registry (get_docmind_tools)
- DocMindAgent: Tool calling, intermediate steps tracing, error resilience
- Stateful multi-turn agent conversation
- Negative and edge cases
"""

import pytest
from pathlib import Path
from typing import Any, List, Optional
from pydantic import ValidationError
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

from tools.calculator_tool import calculator_tool, CalculatorInput
from tools.metadata_tool import metadata_catalog_tool, DocumentCatalogInput
from tools.search_tool import create_search_tool, SearchDocumentsInput
from tools import get_docmind_tools
from agent.doc_agent import create_docmind_agent, DocMindAgent
from memory.history_store import SessionHistoryManager
from vectorstore.embedder import get_fake_embeddings
from vectorstore.store import get_or_create_faiss


class MockToolCallingChatModel(BaseChatModel):
    """Deterministic mock chat model supporting tool calls and bind_tools."""
    responses: List[AIMessage]
    _idx: int = 0

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = self.responses[self._idx]
        self._idx = min(self._idx + 1, len(self.responses) - 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "mock-tool-caller"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


@pytest.fixture
def test_vectorstore():
    docs = [
        Document(
            page_content="Project Phoenix budget is $45,000 with 15% contingency reserve.",
            metadata={"source": "budget.txt", "filename": "budget.txt", "file_type": "txt"},
        ),
        Document(
            page_content="Quarterly revenue for Q1 was $120,000 and Q2 was $140,000.",
            metadata={"source": "revenue.csv", "filename": "revenue.csv", "file_type": "csv", "row": 1},
        ),
    ]
    embedder = get_fake_embeddings()
    return get_or_create_faiss(documents=docs, embeddings=embedder)


class TestCalculatorTool:
    def test_basic_arithmetic(self):
        res = calculator_tool.invoke({"expression": "25000 * 0.15"})
        assert res == "3750"

    def test_aggregation_functions(self):
        res_sum = calculator_tool.invoke({"expression": "sum([100, 200, 300, 400])"})
        assert res_sum == "1000"

        res_avg = calculator_tool.invoke({"expression": "avg([10, 20, 30, 40])"})
        assert res_avg == "25"

    def test_division_by_zero(self):
        res = calculator_tool.invoke({"expression": "100 / 0"})
        assert "Division by zero" in res

    def test_empty_expression(self):
        res = calculator_tool.invoke({"expression": ""})
        assert "Error: Empty expression" in res

    def test_security_rejection_of_arbitrary_code(self):
        res = calculator_tool.invoke({"expression": "__import__('os').system('ls')"})
        assert "Error evaluating expression" in res


class TestMetadataCatalogTool:
    def test_list_all_files(self):
        res = metadata_catalog_tool.invoke({"action": "list_all"})
        assert "Document Catalog Inventory:" in res
        assert "sample_doc.txt" in res or "sample_guide.md" in res

    def test_find_by_type(self):
        res = metadata_catalog_tool.invoke({"action": "find_by_type", "file_type": "csv"})
        assert "Documents matching type 'csv':" in res
        assert "sample_data.csv" in res

    def test_get_file_info(self):
        res = metadata_catalog_tool.invoke({"action": "get_file_info", "filename": "sample_data.csv"})
        assert "File Details for 'sample_data.csv':" in res
        assert "Size:" in res


class TestSearchTool:
    def test_search_documents_tool(self, test_vectorstore):
        search_tool = create_search_tool(test_vectorstore)
        res = search_tool.invoke({"query": "Project Phoenix budget", "k": 2})
        assert "Project Phoenix budget is $45,000" in res
        assert "Source: budget.txt" in res

    def test_search_documents_with_filter(self, test_vectorstore):
        search_tool = create_search_tool(test_vectorstore)
        res = search_tool.invoke({"query": "revenue", "k": 2, "file_type": "csv"})
        assert "revenue.csv" in res


class TestToolRegistry:
    def test_get_docmind_tools_without_and_with_store(self, test_vectorstore):
        tools_no_store = get_docmind_tools()
        assert len(tools_no_store) == 2

        tools_with_store = get_docmind_tools(test_vectorstore)
        assert len(tools_with_store) == 3
        tool_names = [t.name for t in tools_with_store]
        assert "calculator" in tool_names
        assert "query_document_catalog" in tool_names
        assert "search_documents" in tool_names


class TestDocMindAgent:
    def test_agent_execution_with_calculator_tool(self):
        mock_llm = MockToolCallingChatModel(responses=[
            # Tool call message
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "calculator",
                    "args": {"expression": "45000 * 0.15"},
                    "id": "call_calc_01",
                }],
            ),
            # Final response
            AIMessage(
                content="The 15% contingency reserve for Project Phoenix is $6,750."
            ),
        ])

        manager = SessionHistoryManager(storage_type="memory")
        agent = DocMindAgent(
            llm=mock_llm,
            tools=[calculator_tool],
            history_manager=manager,
        )

        res = agent.run("Calculate 15% contingency on $45,000", session_id="test_sess_01")
        assert "$6,750" in res["output"]
        assert len(res["intermediate_steps"]) == 1

        # Check history was persisted
        hist = manager.get_session_history("test_sess_01")
        assert len(hist.messages) == 2
        assert hist.messages[0].content == "Calculate 15% contingency on $45,000"

    def test_agent_multi_turn_conversation(self):
        mock_llm = MockToolCallingChatModel(responses=[
            # Turn 1: Catalog list tool call
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "query_document_catalog",
                    "args": {"action": "find_by_type", "file_type": "csv"},
                    "id": "call_cat_01",
                }],
            ),
            # Turn 1: Final answer
            AIMessage(content="I found sample_data.csv in the catalog."),
            # Turn 2: Calculation tool call
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "calculator",
                    "args": {"expression": "5000 + 7000"},
                    "id": "call_calc_02",
                }],
            ),
            # Turn 2: Final answer
            AIMessage(content="The sum is 12000."),
        ])

        manager = SessionHistoryManager(storage_type="memory")
        agent = DocMindAgent(
            llm=mock_llm,
            tools=[metadata_catalog_tool, calculator_tool],
            history_manager=manager,
        )

        # Turn 1
        res1 = agent.run("Find CSV files", session_id="sess_multi_01")
        assert "sample_data.csv" in res1["output"]

        # Turn 2
        res2 = agent.run("Now calculate 5000 + 7000", session_id="sess_multi_01")
        assert "12000" in res2["output"]

        hist = manager.get_session_history("sess_multi_01")
        assert len(hist.messages) == 4
