"""Unit and integration tests for Phase 6: FastAPI Backend and SSE Streaming.

Covers:
- Root and Health Check Endpoints
- Document Upload and Catalog Listing
- Conversational RAG Endpoint (Blocking and SSE Token Streaming)
- Tool-Calling Agent Endpoint (Blocking and SSE Step Streaming)
- Session Persistence and State Management
- Validation and Error Handling
"""

import io
import json
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from api.server import create_app
from api.dependencies import AppState, get_app_state
from vectorstore.embedder import get_fake_embeddings
from vectorstore.store import get_or_create_faiss
from memory.history_store import SessionHistoryManager
from memory.conversational_rag import ConversationalRAGChain
from agent.doc_agent import DocMindAgent
from tools.calculator_tool import calculator_tool


class MockToolCallingChatModel(FakeListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def test_app():
    """Creates a test FastAPI application with mocked LLM and offline vector store."""
    app = create_app()

    test_state = AppState()
    test_state.embedder = get_fake_embeddings()

    # Pre-index mock documents
    sample_docs = [
        Document(
            page_content="DocMind supports PDF, Markdown, CSV, and Web ingestion with dedicated embeddings.",
            metadata={"source": "overview.txt", "filename": "overview.txt", "file_type": "txt"},
        ),
        Document(
            page_content="The total research budget for 2026 is $150,000.",
            metadata={"source": "finance.csv", "filename": "finance.csv", "file_type": "csv", "row": 1},
        ),
    ]
    test_state.vectorstore = get_or_create_faiss(documents=sample_docs, embeddings=test_state.embedder)
    test_state.history_manager = SessionHistoryManager(storage_type="memory")

    # Mock LLM
    test_state._llm = MockToolCallingChatModel(responses=[
        "DocMind supports multi-format ingestion including PDF, Markdown, and CSV.",
        "The total research budget is $150,000.",
        "I have calculated the answer for you.",
    ])

    app.dependency_overrides[get_app_state] = lambda: test_state
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


class TestRootAndHealth:
    def test_root_endpoint(self, client):
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert "DocMind" in data["name"]
        assert data["version"] == "0.6.0"

    def test_health_check_endpoint(self, client):
        res = client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["total_indexed_chunks"] >= 2


class TestDocumentRoutes:
    def test_list_documents(self, client):
        res = client.get("/api/v1/documents")
        assert res.status_code == 200
        data = res.json()
        assert "total_documents" in data
        assert "documents" in data

    def test_upload_document_success(self, client):
        file_content = b"DocMind Fast API Upload Test: Intelligent Document Analysis Engine."
        files = {"file": ("test_upload.txt", io.BytesIO(file_content), "text/plain")}
        res = client.post("/api/v1/documents/upload", files=files)
        assert res.status_code == 201
        data = res.json()
        assert data["filename"] == "test_upload.txt"
        assert data["chunks_created"] >= 1

    def test_upload_invalid_file_extension(self, client):
        files = {"file": ("malicious.exe", io.BytesIO(b"binary data"), "application/octet-stream")}
        res = client.post("/api/v1/documents/upload", files=files)
        assert res.status_code == 400
        assert "Unsupported file extension" in res.json()["detail"]


class TestChatRoutes:
    def test_chat_blocking(self, client):
        payload = {
            "input": "What document formats are supported?",
            "session_id": "test_session_api_01",
        }
        res = client.post("/api/v1/chat", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == "test_session_api_01"
        assert "DocMind" in data["answer"]
        assert isinstance(data["citations"], list)

    def test_chat_streaming_sse(self, client):
        payload = {
            "input": "What is the budget?",
            "session_id": "test_session_stream_01",
        }
        res = client.post("/api/v1/chat/stream", json=payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        body_text = res.text
        assert "data: " in body_text
        assert '"event": "citations"' in body_text
        assert '"event": "token"' in body_text
        assert '"event": "done"' in body_text

    def test_chat_validation_empty_input(self, client):
        payload = {"input": "", "session_id": "test_sess"}
        res = client.post("/api/v1/chat", json=payload)
        assert res.status_code == 422


class TestAgentRoutes:
    def test_agent_blocking(self, client):
        payload = {
            "input": "Calculate the tax on budget",
            "session_id": "agent_session_api_01",
        }
        res = client.post("/api/v1/agent", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == "agent_session_api_01"
        assert "output" in data
        assert isinstance(data["intermediate_steps"], list)

    def test_agent_streaming_sse(self, client):
        payload = {
            "input": "Search for files and sum numbers",
            "session_id": "agent_stream_sess_01",
        }
        res = client.post("/api/v1/agent/stream", json=payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        body_text = res.text
        assert "data: " in body_text
        assert '"event": "token"' in body_text
        assert '"event": "done"' in body_text
